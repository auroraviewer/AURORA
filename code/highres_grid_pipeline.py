"""Regular-grid GPU strip accumulation and streaming high-resolution output.

The module keeps repeated UNI token contributions on the model device.  It
accumulates them into bounded horizontal strips on the known 14-pixel lattice,
then emits each completed unique grid row exactly once for downstream
coarsening, smoothing, and float16 output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


BatchPositions = tuple[torch.Tensor, torch.Tensor]


def _canonicalize_device(
    device: torch.device | str | None,
) -> torch.device:
    """Resolve an unindexed CUDA alias to the active concrete CUDA device."""

    resolved = torch.device(device or "cpu")
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


@dataclass
class FineGridStrip:
    """Completed fine-grid feature sums for a contiguous row interval."""

    row_start: int
    global_sums: torch.Tensor
    spatial_sums: torch.Tensor
    counts: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.row_start = int(self.row_start)
        if self.row_start < 0:
            raise ValueError("row_start cannot be negative")
        if self.global_sums.ndim != 3 or self.spatial_sums.ndim != 3:
            raise ValueError("strip sums must have shape (rows, cols, channels)")
        if self.global_sums.shape[:2] != self.spatial_sums.shape[:2]:
            raise ValueError("global and spatial strip grids must match")
        if self.global_sums.dtype != torch.float32:
            raise ValueError("global_sums must be float32")
        if self.spatial_sums.dtype != torch.float32:
            raise ValueError("spatial_sums must be float32")
        if self.global_sums.device != self.spatial_sums.device:
            raise ValueError("global and spatial strip sums must share a device")
        if self.counts is not None:
            if self.counts.shape != self.global_sums.shape[:2]:
                raise ValueError("counts must match the strip row/column grid")
            if self.counts.device != self.global_sums.device:
                raise ValueError("counts must share the strip device")

    @property
    def row_end(self) -> int:
        return self.row_start + int(self.global_sums.shape[0])


@dataclass
class _MutableDeviceStrip:
    row_start: int
    global_sums: torch.Tensor
    spatial_sums: torch.Tensor
    counts: torch.Tensor | None = None

    @property
    def row_end(self) -> int:
        return self.row_start + int(self.global_sums.shape[0])


class RegularGridStripAccumulator:
    """Accumulate valid UNI contributions into bounded regular-grid strips.

    Model outputs are first rounded to float16, matching the legacy transfer
    path, then promoted to float32 only for accumulation.  Destination cells
    are computed directly from source patch origins and token row/column
    offsets, so no coordinate sorting or per-batch unique grouping is needed.
    """

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int],
        global_dim: int,
        spatial_dim: int,
        strip_rows: int,
        token_size: int = 14,
        token_grid_shape: tuple[int, int] = (16, 16),
        contribution_chunk: int = 8192,
        device: torch.device | str | None = None,
        track_counts: bool = False,
    ) -> None:
        grid_rows, grid_cols = (int(grid_shape[0]), int(grid_shape[1]))
        token_grid_rows, token_grid_cols = (
            int(token_grid_shape[0]),
            int(token_grid_shape[1]),
        )
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("grid_shape entries must be positive")
        if global_dim <= 0 or spatial_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if strip_rows <= 0:
            raise ValueError("strip_rows must be positive")
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if token_grid_rows <= 0 or token_grid_cols <= 0:
            raise ValueError("token_grid_shape entries must be positive")
        if contribution_chunk <= 0:
            raise ValueError("contribution_chunk must be positive")

        self.grid_shape = (grid_rows, grid_cols)
        self.global_dim = int(global_dim)
        self.spatial_dim = int(spatial_dim)
        self.strip_rows = int(strip_rows)
        self.token_size = int(token_size)
        self.token_grid_shape = (token_grid_rows, token_grid_cols)
        self.token_count = token_grid_rows * token_grid_cols
        self.contribution_chunk = int(contribution_chunk)
        self.device = _canonicalize_device(device)
        self.track_counts = bool(track_counts)

        token_rows = torch.arange(token_grid_rows, dtype=torch.long)
        token_cols = torch.arange(token_grid_cols, dtype=torch.long)
        mesh_rows, mesh_cols = torch.meshgrid(
            token_rows,
            token_cols,
            indexing="ij",
        )
        self._token_rows = mesh_rows.reshape(-1).to(self.device)
        self._token_cols = mesh_cols.reshape(-1).to(self.device)
        self._strips: dict[int, _MutableDeviceStrip] = {}
        self._flushed_until = 0
        self._closed = False
        self.accumulated_contributions = 0
        self.emitted_strips = 0

    def _allocate_strip(self, strip_id: int) -> _MutableDeviceStrip:
        row_start = strip_id * self.strip_rows
        if row_start >= self.grid_shape[0]:
            raise IndexError(f"strip {strip_id} starts outside the output grid")
        rows = min(self.strip_rows, self.grid_shape[0] - row_start)
        common = {
            "dtype": torch.float32,
            "device": self.device,
        }
        return _MutableDeviceStrip(
            row_start=row_start,
            global_sums=torch.zeros(
                (rows, self.grid_shape[1], self.global_dim),
                **common,
            ),
            spatial_sums=torch.zeros(
                (rows, self.grid_shape[1], self.spatial_dim),
                **common,
            ),
            counts=(
                torch.zeros(
                    (rows, self.grid_shape[1]),
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.track_counts
                else None
            ),
        )

    def _get_strip(self, strip_id: int) -> _MutableDeviceStrip:
        strip = self._strips.get(strip_id)
        if strip is None:
            strip = self._allocate_strip(strip_id)
            self._strips[strip_id] = strip
        return strip

    @staticmethod
    def _validate_positions(
        positions: Sequence[torch.Tensor],
        batch_size: int,
    ) -> BatchPositions:
        if len(positions) != 2:
            raise ValueError("positions must contain row and column tensors")
        rows, cols = positions
        if rows.shape != (batch_size,) or cols.shape != (batch_size,):
            raise ValueError(
                "row and column positions must each have shape "
                f"({batch_size},), got {rows.shape} and {cols.shape}"
            )
        return rows, cols

    def accumulate(
        self,
        *,
        global_features: torch.Tensor,
        spatial_features: torch.Tensor,
        positions: Sequence[torch.Tensor],
        keep_flag: torch.Tensor,
    ) -> int:
        """Add one model batch and return its number of valid token rows."""

        if self._closed:
            raise RuntimeError("cannot accumulate after finish()")
        if global_features.ndim != 2:
            raise ValueError("global_features must have shape (B, G)")
        if spatial_features.ndim != 3:
            raise ValueError("spatial_features must have shape (B, N, S)")
        batch_size = int(global_features.shape[0])
        if spatial_features.shape[0] != batch_size:
            raise ValueError("global and spatial feature batches must match")
        if global_features.shape[1] != self.global_dim:
            raise ValueError(
                f"expected {self.global_dim} global channels, got "
                f"{global_features.shape[1]}"
            )
        if spatial_features.shape[1:] != (self.token_count, self.spatial_dim):
            raise ValueError(
                "spatial_features must have shape "
                f"(B, {self.token_count}, {self.spatial_dim}), got "
                f"{tuple(spatial_features.shape)}"
            )
        if global_features.device != self.device or spatial_features.device != self.device:
            raise ValueError(
                "model features must already be on the accumulator device "
                f"{self.device}"
            )
        if keep_flag.shape != (batch_size, *self.token_grid_shape):
            raise ValueError(
                "keep_flag must have shape "
                f"({batch_size}, {self.token_grid_shape[0]}, "
                f"{self.token_grid_shape[1]}), got {tuple(keep_flag.shape)}"
            )
        row_positions, col_positions = self._validate_positions(
            positions,
            batch_size,
        )

        # Patch origins and token-validity metadata are produced by the CPU
        # DataLoader.  Resolve the small integer routing problem there instead
        # of calling CUDA ``nonzero``/``unique`` and scalar ``any`` checks, all
        # of which synchronize the model stream with the host.  Only compact
        # integer index vectors are copied to the accumulator device.
        row_positions_cpu = row_positions.detach().to(
            device="cpu", dtype=torch.long
        )
        col_positions_cpu = col_positions.detach().to(
            device="cpu", dtype=torch.long
        )
        if bool((row_positions_cpu % self.token_size != 0).any()) or bool(
            (col_positions_cpu % self.token_size != 0).any()
        ):
            raise ValueError("patch positions must align to token_size")

        keep_cpu = keep_flag.detach().to(device="cpu", dtype=torch.bool)
        valid_pairs = torch.nonzero(
            keep_cpu.reshape(batch_size, self.token_count),
            as_tuple=False,
        )
        num_valid = int(valid_pairs.shape[0])
        if num_valid == 0:
            return 0

        batch_indices_cpu = valid_pairs[:, 0].to(dtype=torch.long)
        token_indices_cpu = valid_pairs[:, 1].to(dtype=torch.long)
        token_rows_cpu = torch.div(
            token_indices_cpu,
            self.token_grid_shape[1],
            rounding_mode="floor",
        )
        token_cols_cpu = token_indices_cpu.remainder(self.token_grid_shape[1])
        origin_rows_cpu = torch.div(
            row_positions_cpu, self.token_size, rounding_mode="floor"
        )
        origin_cols_cpu = torch.div(
            col_positions_cpu, self.token_size, rounding_mode="floor"
        )
        destination_rows_cpu = origin_rows_cpu.index_select(
            0, batch_indices_cpu
        ) + token_rows_cpu
        destination_cols_cpu = origin_cols_cpu.index_select(
            0, batch_indices_cpu
        ) + token_cols_cpu

        if bool((destination_rows_cpu < 0).any()) or bool(
            (destination_rows_cpu >= self.grid_shape[0]).any()
        ):
            raise IndexError("valid token row lies outside the regular output grid")
        if bool((destination_cols_cpu < 0).any()) or bool(
            (destination_cols_cpu >= self.grid_shape[1]).any()
        ):
            raise IndexError("valid token column lies outside the regular output grid")
        if bool((destination_rows_cpu < self._flushed_until).any()):
            raise RuntimeError(
                "received an already-flushed regular-grid contribution; "
                "source patches must be processed in row-major order"
            )

        strip_ids_cpu = torch.div(
            destination_rows_cpu,
            self.strip_rows,
            rounding_mode="floor",
        )
        unique_strip_ids = torch.unique(strip_ids_cpu, sorted=True).tolist()
        global_half = global_features.detach().to(dtype=torch.float16)
        flattened_spatial = spatial_features.detach().reshape(
            batch_size * self.token_count,
            self.spatial_dim,
        )
        non_blocking = self.device.type == "cuda"

        for strip_id_value in unique_strip_ids:
            strip_id = int(strip_id_value)
            selected_cpu = torch.nonzero(
                strip_ids_cpu == strip_id,
                as_tuple=False,
            ).flatten()
            strip = self._get_strip(strip_id)
            local_rows_cpu = destination_rows_cpu.index_select(
                0, selected_cpu
            ) - strip.row_start
            local_cols_cpu = destination_cols_cpu.index_select(0, selected_cpu)
            destination_indices = (
                local_rows_cpu * self.grid_shape[1] + local_cols_cpu
            ).to(self.device, dtype=torch.long, non_blocking=non_blocking)
            selected_batches = batch_indices_cpu.index_select(
                0, selected_cpu
            ).to(self.device, dtype=torch.long, non_blocking=non_blocking)
            selected_tokens_cpu = token_indices_cpu.index_select(0, selected_cpu)
            selected_spatial_linear = (
                batch_indices_cpu.index_select(0, selected_cpu) * self.token_count
                + selected_tokens_cpu
            ).to(self.device, dtype=torch.long, non_blocking=non_blocking)

            global_destination = strip.global_sums.reshape(-1, self.global_dim)
            spatial_destination = strip.spatial_sums.reshape(-1, self.spatial_dim)
            if strip.counts is not None:
                strip.counts.add_(
                    torch.bincount(
                        destination_indices,
                        minlength=int(strip.counts.numel()),
                    ).reshape_as(strip.counts).to(dtype=torch.int32)
                )
            selected_count = int(selected_cpu.numel())
            for chunk_start in range(0, selected_count, self.contribution_chunk):
                chunk_end = min(
                    chunk_start + self.contribution_chunk,
                    selected_count,
                )
                chunk_destination = destination_indices[chunk_start:chunk_end]
                chunk_batches = selected_batches[chunk_start:chunk_end]
                global_source = global_half.index_select(
                    0,
                    chunk_batches,
                ).to(dtype=torch.float32)
                global_destination.index_add_(
                    0,
                    chunk_destination,
                    global_source,
                )
                del global_source

                chunk_spatial_indices = selected_spatial_linear[
                    chunk_start:chunk_end
                ]
                spatial_source = flattened_spatial.index_select(
                    0,
                    chunk_spatial_indices,
                ).to(dtype=torch.float16).to(dtype=torch.float32)
                spatial_destination.index_add_(
                    0,
                    chunk_destination,
                    spatial_source,
                )
                del spatial_source

        self.accumulated_contributions += num_valid
        return num_valid

    def flush_before(self, row_exclusive: int) -> list[FineGridStrip]:
        """Emit whole strips ending at or before a proven-safe grid row."""

        if self._closed:
            raise RuntimeError("cannot flush after finish()")
        row_exclusive = int(row_exclusive)
        if row_exclusive < self._flushed_until:
            raise ValueError("row_exclusive cannot move backwards")
        row_exclusive = min(row_exclusive, self.grid_shape[0])

        ready_ids = sorted(
            strip_id
            for strip_id, strip in self._strips.items()
            if strip.row_end <= row_exclusive
        )
        output: list[FineGridStrip] = []
        for strip_id in ready_ids:
            strip = self._strips.pop(strip_id)
            output.append(
                FineGridStrip(
                    row_start=strip.row_start,
                    global_sums=strip.global_sums,
                    spatial_sums=strip.spatial_sums,
                    counts=strip.counts,
                )
            )
            self._flushed_until = max(self._flushed_until, strip.row_end)
            self.emitted_strips += 1
        return output

    def finish(self) -> list[FineGridStrip]:
        """Emit all remaining strips in row order and close the accumulator."""

        if self._closed:
            return []
        output: list[FineGridStrip] = []
        for strip_id in sorted(self._strips):
            strip = self._strips[strip_id]
            output.append(
                FineGridStrip(
                    row_start=strip.row_start,
                    global_sums=strip.global_sums,
                    spatial_sums=strip.spatial_sums,
                    counts=strip.counts,
                )
            )
            self._flushed_until = max(self._flushed_until, strip.row_end)
            self.emitted_strips += 1
        self._strips.clear()
        self._closed = True
        return output


__all__ = [
    "FineGridStrip",
    "RegularGridStripAccumulator",
]
