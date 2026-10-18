"""Memory-efficient aggregation and smoothing for UNI high-resolution features.

The key design choice is to keep the global embedding at one row per 224x224
source patch.  During grouped CPU reduction it is expanded only in small
channel chunks, so no full ``num_valid_tokens x 1536`` repeated global matrix
and no full ``num_valid_tokens x 3072`` concatenated matrix are materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class AggregatedGrid:
    """Feature sums and contribution counts on a regular integer grid.

    ``grid_positions`` are row/column indices in units of ``base_patch_size``.
    ``sums`` are float32 sums of concatenated ``(global || local)`` features;
    means are intentionally delayed until output so coarser scales can be
    derived exactly from sums and counts.
    """

    grid_positions: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    base_patch_size: int = 14

    def __post_init__(self) -> None:
        self.grid_positions = np.asarray(self.grid_positions, dtype=np.int32)
        self.sums = np.asarray(self.sums, dtype=np.float32)
        self.counts = np.asarray(self.counts, dtype=np.int64)

        if self.grid_positions.ndim != 2 or self.grid_positions.shape[1] != 2:
            raise ValueError(
                "grid_positions must have shape (num_positions, 2), got "
                f"{self.grid_positions.shape}"
            )
        if self.sums.ndim != 2:
            raise ValueError(f"sums must be two-dimensional, got {self.sums.shape}")
        if self.counts.ndim != 1:
            raise ValueError(f"counts must be one-dimensional, got {self.counts.shape}")
        if not (
            self.grid_positions.shape[0]
            == self.sums.shape[0]
            == self.counts.shape[0]
        ):
            raise ValueError(
                "grid_positions, sums, and counts must have the same first dimension"
            )
        if self.base_patch_size <= 0:
            raise ValueError("base_patch_size must be positive")
        if self.counts.size and np.any(self.counts <= 0):
            raise ValueError("all aggregated counts must be positive")


class SplitGridFeatureAccumulator:
    """Chunked accumulation of separate global and spatial feature sources.

    Each merge receives one batch's global embeddings ``[B, G]`` plus either:

    * dense spatial tokens ``[B, N, S]`` and ``token_indices``; or
    * pre-gathered valid spatial tokens ``[M, S]`` with ``token_indices=None``.

    Global vectors are indexed and reduced in small channel chunks.  No full
    repeated-global matrix and no full per-token ``(global || spatial)`` input
    matrix are materialized.
    """

    def __init__(
        self,
        global_dim: int,
        spatial_dim: int,
        base_patch_size: int = 14,
        chunk_budget: int = 100_000,
        channel_chunk: int = 64,
    ) -> None:
        if global_dim <= 0:
            raise ValueError("global_dim must be positive")
        if spatial_dim <= 0:
            raise ValueError("spatial_dim must be positive")
        if base_patch_size <= 0:
            raise ValueError("base_patch_size must be positive")
        if chunk_budget <= 0:
            raise ValueError("chunk_budget must be positive")
        if channel_chunk <= 0:
            raise ValueError("channel_chunk must be positive")

        self.global_dim = int(global_dim)
        self.spatial_dim = int(spatial_dim)
        self.feature_dim = self.global_dim + self.spatial_dim
        self.base_patch_size = int(base_patch_size)
        self.chunk_budget = int(chunk_budget)
        self.channel_chunk = int(channel_chunk)
        self._chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._size = 0

    @staticmethod
    def _group_offsets(inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(inverse, kind="stable")
        inverse_sorted = inverse[order]
        boundaries = np.flatnonzero(np.diff(inverse_sorted)) + 1
        starts = np.concatenate(
            [np.array([0], dtype=np.int64), boundaries.astype(np.int64, copy=False)]
        )
        ends = np.concatenate(
            [starts[1:], np.array([inverse.shape[0]], dtype=np.int64)]
        )
        return order, starts, ends

    def merge(
        self,
        grid_positions: np.ndarray,
        global_embeddings: np.ndarray,
        spatial_tokens: np.ndarray,
        batch_indices: np.ndarray,
        token_indices: Optional[np.ndarray],
    ) -> None:
        """Merge one batch without forming repeated or concatenated feature rows."""

        grid_positions = np.asarray(grid_positions, dtype=np.int32)
        batch_indices = np.asarray(batch_indices, dtype=np.int64)
        global_embeddings = np.asarray(global_embeddings)
        spatial_tokens = np.asarray(spatial_tokens)

        if grid_positions.ndim != 2 or grid_positions.shape[1] != 2:
            raise ValueError(
                f"grid_positions must have shape (M, 2), got {grid_positions.shape}"
            )
        num_rows = grid_positions.shape[0]
        if num_rows == 0:
            return
        if batch_indices.shape != (num_rows,):
            raise ValueError(
                f"batch_indices must have shape ({num_rows},), got {batch_indices.shape}"
            )
        if global_embeddings.ndim != 2 or global_embeddings.shape[1] != self.global_dim:
            raise ValueError(
                f"global_embeddings must have shape (B, {self.global_dim}), got "
                f"{global_embeddings.shape}"
            )
        if batch_indices.min() < 0 or batch_indices.max() >= global_embeddings.shape[0]:
            raise IndexError("batch_indices contains an out-of-range source-patch index")

        dense_spatial = token_indices is not None
        if dense_spatial:
            token_indices_array = np.asarray(token_indices, dtype=np.int64)
            if token_indices_array.shape != (num_rows,):
                raise ValueError(
                    f"token_indices must have shape ({num_rows},), got "
                    f"{token_indices_array.shape}"
                )
            if spatial_tokens.ndim != 3 or spatial_tokens.shape[2] != self.spatial_dim:
                raise ValueError(
                    f"dense spatial_tokens must have shape (B, N, {self.spatial_dim}), "
                    f"got {spatial_tokens.shape}"
                )
            if spatial_tokens.shape[0] != global_embeddings.shape[0]:
                raise ValueError(
                    "dense spatial_tokens and global_embeddings must have the same batch size"
                )
            if token_indices_array.min() < 0 or token_indices_array.max() >= spatial_tokens.shape[1]:
                raise IndexError("token_indices contains an out-of-range token index")
        else:
            token_indices_array = None
            if spatial_tokens.ndim != 2 or spatial_tokens.shape != (
                num_rows,
                self.spatial_dim,
            ):
                raise ValueError(
                    "pre-gathered spatial_tokens must have shape "
                    f"({num_rows}, {self.spatial_dim}), got {spatial_tokens.shape}"
                )

        unique_positions, inverse = np.unique(
            grid_positions,
            axis=0,
            return_inverse=True,
        )
        order, starts, ends = self._group_offsets(inverse)
        ordered_batch = batch_indices[order]
        ordered_tokens = (
            token_indices_array[order] if token_indices_array is not None else None
        )

        sums = np.empty(
            (unique_positions.shape[0], self.feature_dim),
            dtype=np.float32,
        )
        for channel_start in range(0, self.global_dim, self.channel_chunk):
            channel_end = min(channel_start + self.channel_chunk, self.global_dim)
            global_chunk = np.asarray(
                global_embeddings[ordered_batch, channel_start:channel_end],
                dtype=np.float32,
            )
            sums[:, channel_start:channel_end] = np.add.reduceat(
                global_chunk,
                starts,
                axis=0,
            )

        for channel_start in range(0, self.spatial_dim, self.channel_chunk):
            channel_end = min(channel_start + self.channel_chunk, self.spatial_dim)
            if ordered_tokens is None:
                local_chunk = np.asarray(
                    spatial_tokens[order, channel_start:channel_end],
                    dtype=np.float32,
                )
            else:
                local_chunk = np.asarray(
                    spatial_tokens[
                        ordered_batch,
                        ordered_tokens,
                        channel_start:channel_end,
                    ],
                    dtype=np.float32,
                )
            output_start = self.global_dim + channel_start
            output_end = self.global_dim + channel_end
            sums[:, output_start:output_end] = np.add.reduceat(
                local_chunk,
                starts,
                axis=0,
            )

        counts = (ends - starts).astype(np.int64, copy=False)
        self._chunks.append((unique_positions, sums, counts))
        self._size += unique_positions.shape[0]
        if self._size >= self.chunk_budget:
            self._compact()

    def _compact(self) -> None:
        if len(self._chunks) <= 1:
            return

        all_positions = np.concatenate([chunk[0] for chunk in self._chunks], axis=0)
        all_counts = np.concatenate([chunk[2] for chunk in self._chunks], axis=0)
        unique_positions, inverse = np.unique(
            all_positions,
            axis=0,
            return_inverse=True,
        )
        order, starts, _ = self._group_offsets(inverse)

        sums = np.empty(
            (unique_positions.shape[0], self.feature_dim),
            dtype=np.float32,
        )
        for channel_start in range(0, self.feature_dim, self.channel_chunk):
            channel_end = min(channel_start + self.channel_chunk, self.feature_dim)
            concatenated_columns = np.concatenate(
                [chunk[1][:, channel_start:channel_end] for chunk in self._chunks],
                axis=0,
            )
            sums[:, channel_start:channel_end] = np.add.reduceat(
                concatenated_columns[order],
                starts,
                axis=0,
            )

        counts = np.add.reduceat(all_counts[order], starts, axis=0).astype(
            np.int64,
            copy=False,
        )
        self._chunks = [(unique_positions, sums, counts)]
        self._size = unique_positions.shape[0]
        if self._size >= self.chunk_budget:
            self.chunk_budget = self._size * 2

    def finalize_sums(self) -> AggregatedGrid:
        """Return row-major unique fine-grid sums/counts without taking means."""

        self._compact()
        if self._size == 0:
            return AggregatedGrid(
                grid_positions=np.empty((0, 2), dtype=np.int32),
                sums=np.empty((0, self.feature_dim), dtype=np.float32),
                counts=np.empty((0,), dtype=np.int64),
                base_patch_size=self.base_patch_size,
            )

        positions, sums, counts = self._chunks[0]
        order = np.lexsort((positions[:, 1], positions[:, 0]))
        if not np.array_equal(order, np.arange(order.size)):
            positions = positions[order]
            sums = sums[order]
            counts = counts[order]
        return AggregatedGrid(
            grid_positions=positions,
            sums=sums,
            counts=counts,
            base_patch_size=self.base_patch_size,
        )

    def finalize(
        self,
        patch_size: int,
        base_patch_size: Optional[int] = None,
        *,
        channel_chunk: int = 128,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finalize at the fine scale or a coarser integer multiple."""

        source_unit = self.base_patch_size if base_patch_size is None else int(base_patch_size)
        if source_unit <= 0:
            raise ValueError("base_patch_size must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if patch_size % source_unit != 0:
            raise ValueError(
                f"patch_size={patch_size} must be a multiple of base_patch_size={source_unit}"
            )

        fine = self.finalize_sums()
        fine = AggregatedGrid(
            grid_positions=fine.grid_positions,
            sums=fine.sums,
            counts=fine.counts,
            base_patch_size=source_unit,
        )
        factor = patch_size // source_unit
        result = coarsen_aggregated_grid(
            fine,
            factor=factor,
            channel_chunk=channel_chunk,
        )
        return aggregated_grid_to_output(result, patch_size=patch_size)


class FineGridFeatureAccumulator(SplitGridFeatureAccumulator):
    """Equal-dimensional UNI global/local accumulator convenience wrapper."""

    def __init__(
        self,
        embedding_dim: int = 1536,
        base_patch_size: int = 14,
        chunk_budget: int = 100_000,
        channel_chunk: int = 64,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        super().__init__(
            global_dim=embedding_dim,
            spatial_dim=embedding_dim,
            base_patch_size=base_patch_size,
            chunk_budget=chunk_budget,
            channel_chunk=channel_chunk,
        )

    def merge(
        self,
        *,
        grid_positions: np.ndarray,
        batch_indices: np.ndarray,
        token_indices: Optional[np.ndarray],
        global_embeddings: np.ndarray,
        spatial_tokens: np.ndarray,
    ) -> None:
        super().merge(
            grid_positions,
            global_embeddings,
            spatial_tokens,
            batch_indices,
            token_indices,
        )

def _ensure_row_major(grid: AggregatedGrid) -> AggregatedGrid:
    if grid.grid_positions.shape[0] <= 1:
        return grid
    order = np.lexsort((grid.grid_positions[:, 1], grid.grid_positions[:, 0]))
    if np.array_equal(order, np.arange(order.size)):
        return grid
    return AggregatedGrid(
        grid_positions=grid.grid_positions[order],
        sums=grid.sums[order],
        counts=grid.counts[order],
        base_patch_size=grid.base_patch_size,
    )


def coarsen_aggregated_grid(
    fine_grid: AggregatedGrid,
    factor: int,
    channel_chunk: int = 128,
) -> AggregatedGrid:
    """Derive a coarser grid by summing fine-grid sums and counts.

    This is algebraically equivalent to directly aggregating every original
    contribution at the coarser scale, while avoiding a second per-batch sort
    and feature reduction.
    """

    if factor <= 0:
        raise ValueError("factor must be positive")
    if channel_chunk <= 0:
        raise ValueError("channel_chunk must be positive")
    fine_grid = _ensure_row_major(fine_grid)
    if fine_grid.grid_positions.shape[0] == 0:
        return AggregatedGrid(
            grid_positions=fine_grid.grid_positions,
            sums=fine_grid.sums,
            counts=fine_grid.counts,
            base_patch_size=fine_grid.base_patch_size * int(factor),
        )
    if factor == 1:
        return AggregatedGrid(
            grid_positions=fine_grid.grid_positions,
            sums=fine_grid.sums,
            counts=fine_grid.counts,
            base_patch_size=fine_grid.base_patch_size,
        )

    coarse_positions_per_row = fine_grid.grid_positions // int(factor)

    # Fine-grid row-major order does not make coarse cells contiguous. For
    # example, every new fine row restarts coarse-column order at zero, so a
    # coarse cell can reappear later in the sequence. Group by explicit ids
    # rather than relying on adjacent change detection.
    coarse_positions, inverse = np.unique(
        coarse_positions_per_row,
        axis=0,
        return_inverse=True,
    )
    order = np.argsort(inverse, kind="stable")
    inverse_sorted = inverse[order]
    boundaries = np.flatnonzero(np.diff(inverse_sorted)) + 1
    starts = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            boundaries.astype(np.int64, copy=False),
        ]
    )

    coarse_sums = np.empty(
        (coarse_positions.shape[0], fine_grid.sums.shape[1]),
        dtype=np.float32,
    )
    for channel_start in range(0, fine_grid.sums.shape[1], channel_chunk):
        channel_end = min(channel_start + channel_chunk, fine_grid.sums.shape[1])
        coarse_sums[:, channel_start:channel_end] = np.add.reduceat(
            fine_grid.sums[order, channel_start:channel_end],
            starts,
            axis=0,
        )
    coarse_counts = np.add.reduceat(
        fine_grid.counts[order],
        starts,
        axis=0,
    ).astype(np.int64, copy=False)

    return AggregatedGrid(
        grid_positions=coarse_positions,
        sums=coarse_sums,
        counts=coarse_counts,
        base_patch_size=fine_grid.base_patch_size * int(factor),
    )


def aggregated_grid_to_output(
    grid: AggregatedGrid,
    patch_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sums/counts into the existing ``(emb, pos, flag)`` contract."""

    output_patch_size = grid.base_patch_size if patch_size is None else int(patch_size)
    if output_patch_size != grid.base_patch_size:
        raise ValueError(
            f"patch_size={output_patch_size} does not match this grid's "
            f"position unit {grid.base_patch_size}"
        )
    if grid.grid_positions.shape[0] == 0:
        return (
            np.empty((0, grid.sums.shape[1]), dtype=np.float16),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=bool),
        )

    embeddings = (grid.sums / grid.counts[:, None]).astype(np.float16)
    positions = (grid.grid_positions * output_patch_size).astype(np.int32)
    flags = np.ones(grid.grid_positions.shape[0], dtype=bool)
    return embeddings, positions, flags


def validate_uni_patch_sizes(
    patch_sizes: Sequence[int],
    base_patch_size: int = 14,
) -> list[int]:
    """Validate scales that can be derived from UNI's finest token grid."""

    if base_patch_size <= 0:
        raise ValueError("base_patch_size must be positive")
    validated: list[int] = []
    for value in patch_sizes:
        patch_size = int(value)
        if patch_size <= 0:
            raise ValueError(f"patch sizes must be positive; got {patch_size}")
        if patch_size % base_patch_size != 0:
            raise ValueError(
                f"UNI patch size {patch_size} must be a multiple of {base_patch_size}"
            )
        validated.append(patch_size)
    if not validated:
        raise ValueError("at least one patch size is required")
    return validated


def smooth_embeddings(
    embeddings: np.ndarray,
    positions: np.ndarray,
    flags: np.ndarray,
    size: int,
    position_unit: int,
    *,
    channel_chunk: int = 128,
    validate_unique: bool = False,
) -> np.ndarray:
    """Smooth sparse-grid embeddings with direct assignment for unique cells.

    The finalized accumulator guarantees unique positions, so production calls
    can skip the extra ``np.unique`` sort.  Set ``validate_unique=True`` while
    validating another caller or during regression testing.
    """

    embeddings = np.asarray(embeddings)
    positions = np.asarray(positions)
    valid_flags = np.asarray(flags).astype(bool, copy=False)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be two-dimensional, got {embeddings.shape}")
    if positions.shape != (embeddings.shape[0], 2):
        raise ValueError(
            f"positions must have shape ({embeddings.shape[0]}, 2), got {positions.shape}"
        )
    if valid_flags.shape != (embeddings.shape[0],):
        raise ValueError(
            f"flags must have shape ({embeddings.shape[0]},), got {valid_flags.shape}"
        )
    if size <= 0:
        raise ValueError("size must be positive")
    if position_unit <= 0:
        raise ValueError("position_unit must be positive")
    if channel_chunk <= 0:
        raise ValueError("channel_chunk must be positive")

    valid_positions = positions[valid_flags]
    if valid_positions.size == 0:
        return embeddings

    spot_positions = np.floor(valid_positions / position_unit).astype(np.int64)
    spot_positions -= spot_positions.min(axis=0)

    if validate_unique:
        unique_spot_positions = np.unique(spot_positions, axis=0)
        if unique_spot_positions.shape[0] != spot_positions.shape[0]:
            raise ValueError(
                "duplicate positions found after conversion to patch units; got "
                f"{spot_positions.shape[0] - unique_spot_positions.shape[0]} duplicates"
            )

    grid_shape = tuple((spot_positions.max(axis=0) + 1).tolist())
    rows = spot_positions[:, 0]
    cols = spot_positions[:, 1]

    occupied = np.zeros(grid_shape, dtype=bool)
    occupied[rows, cols] = True
    smoothed_count = cv2.blur(
        occupied.astype(np.float32),
        (size, size),
        borderType=cv2.BORDER_REFLECT,
    )

    output_indices = np.flatnonzero(valid_flags)
    nonzero = smoothed_count[rows, cols] > 0
    smoothed_embeddings = np.full_like(embeddings, np.nan, dtype=np.float32)

    for channel_start in range(0, embeddings.shape[1], channel_chunk):
        channel_end = min(channel_start + channel_chunk, embeddings.shape[1])
        channel_values = embeddings[
            valid_flags,
            channel_start:channel_end,
        ].astype(np.float32, copy=False)

        grid_sum = np.zeros(
            (*grid_shape, channel_end - channel_start),
            dtype=np.float32,
        )
        grid_sum[rows, cols, :] = channel_values
        smoothed_sum = cv2.blur(
            grid_sum,
            (size, size),
            borderType=cv2.BORDER_REFLECT,
        )
        if smoothed_sum.ndim == 2:
            smoothed_sum = smoothed_sum[..., None]

        smoothed_embeddings[
            output_indices[nonzero],
            channel_start:channel_end,
        ] = (
            smoothed_sum[rows[nonzero], cols[nonzero], :]
            / smoothed_count[rows[nonzero], cols[nonzero], None]
        )

    return smoothed_embeddings


def smooth_embedding_halves(
    embeddings: np.ndarray,
    positions: np.ndarray,
    flags: np.ndarray,
    position_unit: int,
    first_size: int = 16,
    second_size: int = 4,
    *,
    channel_chunk: int = 128,
    validate_unique: bool = False,
) -> np.ndarray:
    """Smooth UNI2-h global and local 1536-D halves independently."""

    if embeddings.ndim != 2 or embeddings.shape[1] != 1536 * 2:
        raise ValueError(
            f"Expected embeddings with shape (num_spots, 3072), got {embeddings.shape}"
        )

    first_half = smooth_embeddings(
        embeddings[:, :1536],
        positions,
        flags,
        size=first_size,
        position_unit=position_unit,
        channel_chunk=channel_chunk,
        validate_unique=validate_unique,
    )
    second_half = smooth_embeddings(
        embeddings[:, 1536:],
        positions,
        flags,
        size=second_size,
        position_unit=position_unit,
        channel_chunk=channel_chunk,
        validate_unique=validate_unique,
    )
    return np.concatenate([first_half, second_half], axis=1)


__all__ = [
    "AggregatedGrid",
    "FineGridFeatureAccumulator",
    "SplitGridFeatureAccumulator",
    "aggregated_grid_to_output",
    "coarsen_aggregated_grid",
    "smooth_embedding_halves",
    "smooth_embeddings",
    "validate_uni_patch_sizes",
]
