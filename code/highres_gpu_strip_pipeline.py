"""GPU strip accumulation, sparse row streaming, and direct float16 output.

The low-level overlap-add kernel lives in :mod:`highres_grid_pipeline`.  This
module adds three production concerns around it:

* completed regular-grid strips are coarsened on-device without sorting;
* only occupied float16 means are copied to an append-only sparse row store;
* smoothing reconstructs one row halo and one channel chunk at a time and
  writes directly to a float16 memmap.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch

from highres_grid_pipeline import (
    FineGridStrip,
    RegularGridStripAccumulator as _FineGridStripAccumulator,
)


class SparseRowFeatureStore:
    """Append-only row-major storage for occupied float16 grid cells."""

    def __init__(self, root: str | Path, *, scale: int, feature_dim: int) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.scale = int(scale)
        self.feature_dim = int(feature_dim)
        self.root = Path(root) / f"scale_{self.scale}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.feature_path = self.root / "features.f16"
        self.column_path = self.root / "columns.i32"
        self._feature_file = self.feature_path.open("wb")
        self._column_file = self.column_path.open("wb")
        self._row_ids: list[int] = []
        self._row_offsets: list[int] = [0]
        self._next_row: int | None = None
        self._closed = False

    @property
    def row_ids(self) -> np.ndarray:
        return np.asarray(self._row_ids, dtype=np.int64)

    @property
    def row_offsets(self) -> np.ndarray:
        return np.asarray(self._row_offsets, dtype=np.int64)

    @property
    def num_features(self) -> int:
        return int(self._row_offsets[-1])

    @property
    def num_rows(self) -> int:
        return len(self._row_ids)

    @property
    def closed(self) -> bool:
        return self._closed

    def _append_empty_rows(self, row_start: int, row_end: int) -> None:
        for row in range(row_start, row_end):
            self._row_ids.append(row)
            self._row_offsets.append(self._row_offsets[-1])

    def append(
        self,
        *,
        row_start: int,
        row_lengths: np.ndarray,
        columns: np.ndarray,
        features: np.ndarray,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed sparse row store")
        row_start = int(row_start)
        row_lengths = np.ascontiguousarray(row_lengths, dtype=np.int64)
        columns = np.ascontiguousarray(columns, dtype=np.int32)
        features = np.ascontiguousarray(features, dtype=np.float16)
        if row_lengths.ndim != 1:
            raise ValueError("row_lengths must be one-dimensional")
        if columns.ndim != 1:
            raise ValueError("columns must be one-dimensional")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (N, {self.feature_dim}), got "
                f"{features.shape}"
            )
        occupied = int(row_lengths.sum())
        if columns.shape[0] != occupied or features.shape[0] != occupied:
            raise ValueError(
                "row_lengths, columns, and features disagree about occupied cells"
            )
        if np.any(row_lengths < 0):
            raise ValueError("row_lengths cannot be negative")

        if self._next_row is None:
            self._next_row = row_start
        if row_start < self._next_row:
            raise ValueError(
                f"rows must be appended in order: expected at least "
                f"{self._next_row}, got {row_start}"
            )
        if row_start > self._next_row:
            # Leading empty rows before the first occupied strip are irrelevant
            # to the smoothing bounding box.  Internal gaps must be retained.
            if self._row_ids:
                self._append_empty_rows(self._next_row, row_start)
            self._next_row = row_start

        if occupied:
            features.tofile(self._feature_file)
            columns.tofile(self._column_file)

        running = self._row_offsets[-1]
        for offset, length in enumerate(row_lengths.tolist()):
            self._row_ids.append(row_start + offset)
            running += int(length)
            self._row_offsets.append(running)
        self._next_row = row_start + int(row_lengths.shape[0])

    def close(self) -> None:
        if self._closed:
            return
        self._feature_file.flush()
        self._column_file.flush()
        self._feature_file.close()
        self._column_file.close()
        self._closed = True

    def open_features(self, *, mode: str = "r") -> np.ndarray:
        if not self._closed:
            raise RuntimeError("close the sparse row store before reading it")
        if self.num_features == 0:
            return np.empty((0, self.feature_dim), dtype=np.float16)
        return np.memmap(
            self.feature_path,
            dtype=np.float16,
            mode=mode,
            shape=(self.num_features, self.feature_dim),
        )

    def open_columns(self, *, mode: str = "r") -> np.ndarray:
        if not self._closed:
            raise RuntimeError("close the sparse row store before reading it")
        if self.num_features == 0:
            return np.empty((0,), dtype=np.int32)
        return np.memmap(
            self.column_path,
            dtype=np.int32,
            mode=mode,
            shape=(self.num_features,),
        )

    def output_positions(self) -> np.ndarray:
        if self.num_features == 0:
            return np.empty((0, 2), dtype=np.int32)
        row_lengths = np.diff(self.row_offsets)
        columns = self.open_columns()
        positions = np.empty((self.num_features, 2), dtype=np.int32)
        positions[:, 0] = (
            np.repeat(self.row_ids, row_lengths) * self.scale
        ).astype(np.int32, copy=False)
        positions[:, 1] = (
            np.asarray(columns, dtype=np.int64) * self.scale
        ).astype(np.int32, copy=False)
        return positions

    def read_unsmoothed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.open_features(),
            self.output_positions(),
            np.ones((self.num_features,), dtype=bool),
        )


@dataclass
class _HostRowChunk:
    scale: int
    row_start: int
    row_lengths: np.ndarray
    columns_host: torch.Tensor
    features_host: torch.Tensor
    ready_event: object | None = None
    device_refs: tuple[torch.Tensor, ...] = ()


_WRITE_SENTINEL = object()


class _SparseRowWriter:
    """Bounded ordered writer for asynchronously staged GPU row chunks."""

    def __init__(
        self,
        stores: dict[int, SparseRowFeatureStore],
        *,
        max_pending: int,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.stores = stores
        self._queue: queue.Queue[_HostRowChunk | object] = queue.Queue(
            maxsize=int(max_pending)
        )
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._closed = False
        self.processed_chunks = 0
        self.processed_rows = 0
        self.processed_features = 0
        self._worker = threading.Thread(
            target=self._worker_main,
            name="highres-sparse-row-writer",
            daemon=True,
        )
        self._worker.start()

    def _set_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("sparse row writer failed") from error

    def _put(self, item: _HostRowChunk | object) -> None:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def submit_device_chunk(
        self,
        *,
        scale: int,
        row_start: int,
        row_lengths: np.ndarray,
        columns_device: torch.Tensor,
        features_device: torch.Tensor,
        copy_stream: torch.cuda.Stream | None,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot submit after writer finish")
        row_lengths = np.ascontiguousarray(row_lengths, dtype=np.int64)
        if columns_device.dtype != torch.int32 or columns_device.ndim != 1:
            raise ValueError("columns_device must be one-dimensional int32")
        if features_device.dtype != torch.float16 or features_device.ndim != 2:
            raise ValueError("features_device must be two-dimensional float16")
        if columns_device.shape[0] != features_device.shape[0]:
            raise ValueError("columns and features must have equal row counts")
        if int(row_lengths.sum()) != int(features_device.shape[0]):
            raise ValueError("row_lengths does not match feature rows")
        if columns_device.device != features_device.device:
            raise ValueError("columns and features must be on the same device")

        device = features_device.device
        if device.type != "cuda":
            self._put(
                _HostRowChunk(
                    scale=int(scale),
                    row_start=int(row_start),
                    row_lengths=row_lengths,
                    columns_host=columns_device.detach().cpu().contiguous(),
                    features_host=features_device.detach().cpu().contiguous(),
                )
            )
            return

        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=device)
        columns_host = torch.empty(
            columns_device.shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        features_host = torch.empty(
            features_device.shape,
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        compute_stream = torch.cuda.current_stream(device=device)
        copy_stream.wait_stream(compute_stream)
        event = torch.cuda.Event(blocking=False, interprocess=False)
        with torch.cuda.stream(copy_stream):
            columns_host.copy_(columns_device, non_blocking=True)
            features_host.copy_(features_device, non_blocking=True)
            event.record(copy_stream)
        self._put(
            _HostRowChunk(
                scale=int(scale),
                row_start=int(row_start),
                row_lengths=row_lengths,
                columns_host=columns_host,
                features_host=features_host,
                ready_event=event,
                device_refs=(columns_device, features_device),
            )
        )

    def submit_preindexed_device_chunk(
        self,
        *,
        scale: int,
        row_start: int,
        row_lengths: np.ndarray,
        columns_host: np.ndarray,
        features_device: torch.Tensor,
        copy_stream: torch.cuda.Stream | None,
    ) -> None:
        """Stage features when sparse columns are already known on the CPU.

        Regular-grid datasets precompute contribution counts, so occupied
        columns and row lengths are deterministic before any GPU work.  Keeping
        those indices on the host avoids a CUDA ``nonzero`` synchronization and
        a pointless columns device-to-host round trip for every completed strip.
        """

        if self._closed:
            raise RuntimeError("cannot submit after writer finish")
        row_lengths = np.ascontiguousarray(row_lengths, dtype=np.int64)
        columns_array = np.ascontiguousarray(columns_host, dtype=np.int32)
        if features_device.dtype != torch.float16 or features_device.ndim != 2:
            raise ValueError("features_device must be two-dimensional float16")
        if columns_array.ndim != 1:
            raise ValueError("columns_host must be one-dimensional int32")
        if columns_array.shape[0] != int(features_device.shape[0]):
            raise ValueError("columns and features must have equal row counts")
        if int(row_lengths.sum()) != int(features_device.shape[0]):
            raise ValueError("row_lengths does not match feature rows")

        columns_tensor = torch.from_numpy(columns_array)
        device = features_device.device
        if device.type != "cuda":
            self._put(
                _HostRowChunk(
                    scale=int(scale),
                    row_start=int(row_start),
                    row_lengths=row_lengths,
                    columns_host=columns_tensor,
                    features_host=features_device.detach().cpu().contiguous(),
                )
            )
            return

        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=device)
        features_host = torch.empty(
            features_device.shape,
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        compute_stream = torch.cuda.current_stream(device=device)
        copy_stream.wait_stream(compute_stream)
        event = torch.cuda.Event(blocking=False, interprocess=False)
        with torch.cuda.stream(copy_stream):
            features_host.copy_(features_device, non_blocking=True)
            event.record(copy_stream)
        self._put(
            _HostRowChunk(
                scale=int(scale),
                row_start=int(row_start),
                row_lengths=row_lengths,
                columns_host=columns_tensor,
                features_host=features_host,
                ready_event=event,
                device_refs=(features_device,),
            )
        )

    def _process(self, payload: _HostRowChunk) -> None:
        try:
            if payload.ready_event is not None:
                synchronize = getattr(payload.ready_event, "synchronize", None)
                if synchronize is None:
                    raise TypeError("ready_event must provide synchronize()")
                synchronize()
            self.stores[payload.scale].append(
                row_start=payload.row_start,
                row_lengths=payload.row_lengths,
                columns=payload.columns_host.detach().numpy(),
                features=payload.features_host.detach().numpy(),
            )
            self.processed_chunks += 1
            self.processed_rows += int(payload.row_lengths.shape[0])
            self.processed_features += int(payload.features_host.shape[0])
        finally:
            payload.ready_event = None
            payload.device_refs = ()

    def _worker_main(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _WRITE_SENTINEL:
                    return
                if not isinstance(item, _HostRowChunk):
                    raise TypeError("unexpected sparse writer item")
                self._process(item)
            except BaseException as exc:
                self._set_error(exc)
                return
            finally:
                self._queue.task_done()

    def finish(self) -> None:
        if not self._closed:
            self._closed = True
            if self._error is None:
                try:
                    self._put(_WRITE_SENTINEL)
                except RuntimeError:
                    pass
        self._worker.join()
        self._raise_if_failed()
        for store in self.stores.values():
            store.close()


def _coarsen_count_grid_numpy(counts: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a complete regular count grid without sorting."""

    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must be two-dimensional")
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return counts.astype(np.int32, copy=False)
    rows, cols = counts.shape
    coarse_rows = (rows + factor - 1) // factor
    coarse_cols = (cols + factor - 1) // factor
    padded = np.zeros(
        (coarse_rows * factor, coarse_cols * factor),
        dtype=np.int32,
    )
    padded[:rows, :cols] = counts.astype(np.int32, copy=False)
    return padded.reshape(
        coarse_rows, factor, coarse_cols, factor
    ).sum(axis=(1, 3), dtype=np.int32)


def _coarsen_feature_sums(
    strip: FineGridStrip,
    *,
    factor: int,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Coarsen only feature sums, leaving known count metadata on the CPU."""

    if factor <= 0:
        raise ValueError("factor must be positive")
    if strip.row_start % factor != 0:
        raise ValueError(
            f"strip row {strip.row_start} is not aligned to factor {factor}"
        )
    if factor == 1:
        return strip.row_start, strip.global_sums, strip.spatial_sums

    rows, cols = strip.global_sums.shape[:2]
    coarse_rows = (rows + factor - 1) // factor
    coarse_cols = (cols + factor - 1) // factor
    padded_rows = coarse_rows * factor
    padded_cols = coarse_cols * factor

    def coarsen(values: torch.Tensor) -> torch.Tensor:
        channels = values.shape[2]
        if padded_rows != rows or padded_cols != cols:
            padded = torch.zeros(
                (padded_rows, padded_cols, channels),
                dtype=values.dtype,
                device=values.device,
            )
            padded[:rows, :cols] = values
        else:
            padded = values
        return padded.reshape(
            coarse_rows, factor, coarse_cols, factor, channels
        ).sum(dim=(1, 3), dtype=torch.float32)

    return (
        strip.row_start // factor,
        coarsen(strip.global_sums),
        coarsen(strip.spatial_sums),
    )


def _coarsen_strip(
    strip: FineGridStrip,
    *,
    factor: int,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coarsen one aligned completed fine strip on its current device."""

    if strip.counts is None:
        raise RuntimeError("strip count tracking is required")
    row_start, global_sums, spatial_sums = _coarsen_feature_sums(
        strip, factor=factor
    )
    if factor == 1:
        return row_start, global_sums, spatial_sums, strip.counts

    rows, cols = strip.counts.shape
    coarse_rows = (rows + factor - 1) // factor
    coarse_cols = (cols + factor - 1) // factor
    padded_rows = coarse_rows * factor
    padded_cols = coarse_cols * factor
    if padded_rows != rows or padded_cols != cols:
        padded_counts = torch.zeros(
            (padded_rows, padded_cols),
            dtype=strip.counts.dtype,
            device=strip.counts.device,
        )
        padded_counts[:rows, :cols] = strip.counts
    else:
        padded_counts = strip.counts
    coarse_counts = padded_counts.reshape(
        coarse_rows, factor, coarse_cols, factor
    ).sum(dim=(1, 3))
    return row_start, global_sums, spatial_sums, coarse_counts


def _stage_completed_grid(
    *,
    scale: int,
    row_start: int,
    global_sums: torch.Tensor,
    spatial_sums: torch.Tensor,
    counts: torch.Tensor,
    writer: _SparseRowWriter,
    copy_stream: torch.cuda.Stream | None,
) -> None:
    occupied = counts > 0
    row_lengths = occupied.sum(dim=1, dtype=torch.int64).detach().cpu().numpy()
    occupied_indices = torch.nonzero(occupied, as_tuple=False)
    feature_dim = global_sums.shape[2] + spatial_sums.shape[2]
    if occupied_indices.shape[0]:
        rows = occupied_indices[:, 0]
        columns = occupied_indices[:, 1]
        denominator = counts[rows, columns].to(torch.float32).unsqueeze(1)
        features = torch.empty(
            (occupied_indices.shape[0], feature_dim),
            dtype=torch.float16,
            device=global_sums.device,
        )
        global_dim = global_sums.shape[2]
        features[:, :global_dim] = (
            global_sums[rows, columns] / denominator
        ).to(torch.float16)
        features[:, global_dim:] = (
            spatial_sums[rows, columns] / denominator
        ).to(torch.float16)
        columns_i32 = columns.to(torch.int32)
    else:
        columns_i32 = torch.empty(
            (0,), dtype=torch.int32, device=global_sums.device
        )
        features = torch.empty(
            (0, feature_dim), dtype=torch.float16, device=global_sums.device
        )
    writer.submit_device_chunk(
        scale=scale,
        row_start=row_start,
        row_lengths=row_lengths,
        columns_device=columns_i32,
        features_device=features,
        copy_stream=copy_stream,
    )


def _stage_completed_grid_precomputed(
    *,
    scale: int,
    row_start: int,
    global_sums: torch.Tensor,
    spatial_sums: torch.Tensor,
    counts: np.ndarray,
    writer: _SparseRowWriter,
    copy_stream: torch.cuda.Stream | None,
) -> None:
    """Stage occupied cells using CPU-known regular-grid count metadata."""

    counts = np.asarray(counts, dtype=np.int32)
    if counts.shape != tuple(global_sums.shape[:2]):
        raise ValueError(
            "precomputed count slice must match the completed feature grid"
        )
    occupied_rows, occupied_cols = np.nonzero(counts > 0)
    row_lengths = np.count_nonzero(counts > 0, axis=1).astype(np.int64)
    feature_dim = int(global_sums.shape[2] + spatial_sums.shape[2])
    num_occupied = int(occupied_rows.shape[0])
    if num_occupied:
        linear_cpu = torch.from_numpy(
            (
                occupied_rows.astype(np.int64, copy=False) * counts.shape[1]
                + occupied_cols.astype(np.int64, copy=False)
            )
        )
        linear_device = linear_cpu.to(
            global_sums.device,
            dtype=torch.long,
            non_blocking=global_sums.device.type == "cuda",
        )
        denominator = torch.from_numpy(
            counts[occupied_rows, occupied_cols].astype(np.float32, copy=False)
        ).to(
            global_sums.device,
            dtype=torch.float32,
            non_blocking=global_sums.device.type == "cuda",
        ).unsqueeze(1)
        features = torch.empty(
            (num_occupied, feature_dim),
            dtype=torch.float16,
            device=global_sums.device,
        )
        global_dim = int(global_sums.shape[2])
        features[:, :global_dim] = (
            global_sums.reshape(-1, global_dim).index_select(0, linear_device)
            / denominator
        ).to(torch.float16)
        spatial_dim = int(spatial_sums.shape[2])
        features[:, global_dim:] = (
            spatial_sums.reshape(-1, spatial_dim).index_select(0, linear_device)
            / denominator
        ).to(torch.float16)
        columns_host = occupied_cols.astype(np.int32, copy=False)
    else:
        features = torch.empty(
            (0, feature_dim),
            dtype=torch.float16,
            device=global_sums.device,
        )
        columns_host = np.empty((0,), dtype=np.int32)

    writer.submit_preindexed_device_chunk(
        scale=scale,
        row_start=row_start,
        row_lengths=row_lengths,
        columns_host=columns_host,
        features_device=features,
        copy_stream=copy_stream,
    )


class StreamingStripStorePipeline:
    """Coarsen completed fine strips and stream unique float16 cells.

    ``fine_contribution_counts`` may be precomputed by the regular-grid dataset,
    which removes count atomics from the model-side accumulator.  Submitted
    strips must be row ordered and aligned to every requested coarsening factor.
    """

    def __init__(
        self,
        *,
        fine_contribution_counts: np.ndarray,
        output_scales: Sequence[int],
        token_size: int,
        global_dim: int,
        spatial_dim: int,
        temp_dir: str | Path,
        device: torch.device | str,
        queue_depth: int = 2,
        copy_stream: torch.cuda.Stream | None = None,
    ) -> None:
        counts = np.asarray(fine_contribution_counts)
        if counts.ndim != 2:
            raise ValueError("fine_contribution_counts must be two-dimensional")
        if np.any(counts < 0):
            raise ValueError("fine contribution counts cannot be negative")
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        scales = tuple(dict.fromkeys(int(value) for value in output_scales))
        if not scales or any(
            scale <= 0 or scale % token_size != 0 for scale in scales
        ):
            raise ValueError(
                "output scales must be positive multiples of token_size"
            )
        self.counts = counts.astype(np.int32, copy=False)
        self.output_scales = scales
        self.token_size = int(token_size)
        self._scale_counts = {
            scale: _coarsen_count_grid_numpy(
                self.counts, scale // self.token_size
            )
            for scale in scales
        }
        self.global_dim = int(global_dim)
        self.spatial_dim = int(spatial_dim)
        self.feature_dim = self.global_dim + self.spatial_dim
        self.device = torch.device(device)
        self.copy_stream = copy_stream
        self.stores = {
            scale: SparseRowFeatureStore(
                temp_dir,
                scale=scale,
                feature_dim=self.feature_dim,
            )
            for scale in scales
        }
        self.writer = _SparseRowWriter(self.stores, max_pending=queue_depth)
        self._next_fine_row: int | None = None
        self._finished = False

    def submit(self, strip: FineGridStrip) -> None:
        if self._finished:
            raise RuntimeError("cannot submit after finish")
        if strip.global_sums.shape[2] != self.global_dim:
            raise ValueError("strip global dimension does not match pipeline")
        if strip.spatial_sums.shape[2] != self.spatial_dim:
            raise ValueError("strip spatial dimension does not match pipeline")
        row_start = int(strip.row_start)
        row_end = row_start + int(strip.global_sums.shape[0])
        if row_end > self.counts.shape[0]:
            raise IndexError("strip extends beyond fine contribution count grid")
        if strip.global_sums.shape[1] != self.counts.shape[1]:
            raise ValueError("strip width does not match contribution count grid")
        if self._next_fine_row is not None and row_start < self._next_fine_row:
            raise ValueError("fine strips must be submitted in row order")
        self._next_fine_row = row_end

        for scale in self.output_scales:
            factor = scale // self.token_size
            if row_start % factor != 0 or int(strip.global_sums.shape[0]) % factor != 0:
                raise ValueError(
                    "submitted strip rows must align to every output scale factor"
                )
            coarse_row, global_sums, spatial_sums = _coarsen_feature_sums(
                strip, factor=factor
            )
            coarse_end = coarse_row + int(global_sums.shape[0])
            _stage_completed_grid_precomputed(
                scale=scale,
                row_start=coarse_row,
                global_sums=global_sums,
                spatial_sums=spatial_sums,
                counts=self._scale_counts[scale][coarse_row:coarse_end],
                writer=self.writer,
                copy_stream=self.copy_stream,
            )

    def finish(self) -> dict[int, SparseRowFeatureStore]:
        if self._finished:
            return self.stores
        self._finished = True
        self.writer.finish()
        return self.stores


class RegularGridStripAccumulator:
    """High-level multi-scale strip accumulator with sparse streamed stores."""

    def __init__(
        self,
        *,
        output_scales: Sequence[int],
        fine_grid_cols: int,
        global_dim: int,
        spatial_dim: int,
        token_size: int = 14,
        token_grid_h: int = 16,
        token_grid_w: int = 16,
        device: torch.device | str | None = None,
        temp_dir: str | Path | None = None,
        writer_queue_depth: int = 2,
        fine_grid_rows: int | None = None,
        fine_contribution_counts: np.ndarray | None = None,
        strip_rows: int | None = None,
        contribution_chunk: int = 8192,
        copy_stream: torch.cuda.Stream | None = None,
    ) -> None:
        scales = tuple(dict.fromkeys(int(value) for value in output_scales))
        if not scales:
            raise ValueError("output_scales cannot be empty")
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if any(scale <= 0 or scale % token_size != 0 for scale in scales):
            raise ValueError(
                "all output scales must be positive token-size multiples"
            )
        factors = [scale // token_size for scale in scales]
        # One source patch contributes ``token_grid_h`` consecutive fine rows.
        # Covering at least that height avoids splitting every patch across four
        # tiny strips for the default 14/56 outputs while retaining exact scale
        # alignment for coarsening.
        aligned_strip_rows = math.lcm(int(token_grid_h), *factors)
        chosen_strip_rows = int(strip_rows or aligned_strip_rows)
        if chosen_strip_rows <= 0 or any(
            chosen_strip_rows % factor != 0 for factor in factors
        ):
            raise ValueError(
                "strip_rows must be a positive multiple of every output factor"
            )
        if fine_grid_cols <= 0:
            raise ValueError("fine_grid_cols must be positive")
        precomputed_counts = None
        if fine_contribution_counts is not None:
            precomputed_counts = np.asarray(fine_contribution_counts)
            if precomputed_counts.ndim != 2:
                raise ValueError("fine_contribution_counts must be two-dimensional")
            if precomputed_counts.shape[1] != int(fine_grid_cols):
                raise ValueError("fine contribution count width does not match fine_grid_cols")
            if np.any(precomputed_counts < 0):
                raise ValueError("fine contribution counts cannot be negative")
            precomputed_counts = precomputed_counts.astype(np.int32, copy=False)
            if fine_grid_rows is None:
                fine_grid_rows = int(precomputed_counts.shape[0])
            elif int(fine_grid_rows) != int(precomputed_counts.shape[0]):
                raise ValueError("fine contribution count height does not match fine_grid_rows")
        if fine_grid_rows is None:
            fine_grid_rows = 2**30
        if fine_grid_rows <= 0:
            raise ValueError("fine_grid_rows must be positive")
        if temp_dir is None:
            raise ValueError("temp_dir must be provided")

        self.output_scales = scales
        self.token_size = int(token_size)
        self.global_dim = int(global_dim)
        self.spatial_dim = int(spatial_dim)
        self.feature_dim = self.global_dim + self.spatial_dim
        self.device = torch.device(device or "cpu")
        self.copy_stream = copy_stream
        self._precomputed_counts = precomputed_counts
        self._scale_counts = (
            {
                scale: _coarsen_count_grid_numpy(
                    precomputed_counts, scale // self.token_size
                )
                for scale in scales
            }
            if precomputed_counts is not None
            else None
        )
        self.stores = {
            scale: SparseRowFeatureStore(
                temp_dir,
                scale=scale,
                feature_dim=self.feature_dim,
            )
            for scale in scales
        }
        self.writer = _SparseRowWriter(
            self.stores,
            max_pending=writer_queue_depth,
        )
        self._fine = _FineGridStripAccumulator(
            grid_shape=(int(fine_grid_rows), int(fine_grid_cols)),
            global_dim=self.global_dim,
            spatial_dim=self.spatial_dim,
            strip_rows=chosen_strip_rows,
            token_size=self.token_size,
            token_grid_shape=(int(token_grid_h), int(token_grid_w)),
            contribution_chunk=int(contribution_chunk),
            device=self.device,
            track_counts=self._precomputed_counts is None,
        )
        self._last_patch_row: int | None = None
        self._finished = False
        self.processed_patches = 0
        self.processed_contributions = 0

    def _emit(self, strips: Sequence[FineGridStrip]) -> None:
        for strip in strips:
            if strip.counts is None and self._precomputed_counts is None:
                raise RuntimeError("completed strip is missing contribution counts")
            for scale in self.output_scales:
                factor = scale // self.token_size
                if self._scale_counts is not None:
                    row_start, global_sums, spatial_sums = _coarsen_feature_sums(
                        strip, factor=factor
                    )
                    row_end = row_start + int(global_sums.shape[0])
                    _stage_completed_grid_precomputed(
                        scale=scale,
                        row_start=row_start,
                        global_sums=global_sums,
                        spatial_sums=spatial_sums,
                        counts=self._scale_counts[scale][row_start:row_end],
                        writer=self.writer,
                        copy_stream=self.copy_stream,
                    )
                else:
                    row_start, global_sums, spatial_sums, counts = _coarsen_strip(
                        strip, factor=factor
                    )
                    _stage_completed_grid(
                        scale=scale,
                        row_start=row_start,
                        global_sums=global_sums,
                        spatial_sums=spatial_sums,
                        counts=counts,
                        writer=self.writer,
                        copy_stream=self.copy_stream,
                    )

    def add_batch(
        self,
        *,
        global_features: torch.Tensor,
        spatial_features: torch.Tensor,
        positions: Sequence[torch.Tensor],
        keep_flag: torch.Tensor,
    ) -> int:
        if self._finished:
            raise RuntimeError("cannot add_batch after finish")
        if len(positions) != 2:
            raise ValueError("positions must contain row and column tensors")
        rows = positions[0].detach().cpu().numpy().astype(np.int64, copy=False)
        if rows.ndim != 1 or rows.shape[0] != global_features.shape[0]:
            raise ValueError("row positions must match the model batch")
        if rows.size and np.any(np.diff(rows) < 0):
            raise ValueError("patch rows must be nondecreasing")
        if rows.size == 0:
            return 0
        if self._last_patch_row is not None and int(rows[0]) < self._last_patch_row:
            raise ValueError("patch rows must be globally nondecreasing")

        # A full model batch can span several sparse source rows. Process each
        # contiguous row group separately so rows completed by an earlier group
        # are staged immediately instead of keeping all touched strips live until
        # the next model batch. This preserves fixed-size model inference while
        # bounding accumulation memory by source-patch height rather than batch
        # row span.
        boundaries = np.flatnonzero(np.diff(rows)) + 1
        starts = np.concatenate(
            [np.array([0], dtype=np.int64), boundaries.astype(np.int64)]
        )
        ends = np.concatenate(
            [boundaries.astype(np.int64), np.array([rows.size], dtype=np.int64)]
        )
        count = 0
        for start_value, end_value in zip(starts, ends, strict=True):
            start = int(start_value)
            end = int(end_value)
            safe_row = int(rows[start] // self.token_size)
            self._emit(self._fine.flush_before(safe_row))
            count += self._fine.accumulate(
                global_features=global_features[start:end],
                spatial_features=spatial_features[start:end],
                positions=(positions[0][start:end], positions[1][start:end]),
                keep_flag=keep_flag[start:end],
            )
            self._last_patch_row = int(rows[end - 1])

        self.processed_patches += int(global_features.shape[0])
        self.processed_contributions += int(count)
        return int(count)

    def finish(self) -> dict[int, SparseRowFeatureStore]:
        if self._finished:
            return self.stores
        self._finished = True
        try:
            self._emit(self._fine.finish())
        finally:
            self.writer.finish()
        return self.stores

    def __enter__(self) -> "RegularGridStripAccumulator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is None:
            self.finish()
        else:
            try:
                self.writer.finish()
            except Exception:
                pass
        return False


def _fill_sparse_window(
    destination: np.ndarray,
    *,
    features: np.ndarray | None,
    columns: np.ndarray,
    row_offsets: np.ndarray,
    store_row_start: int,
    read_row_start: int,
    read_row_end: int,
    column_start: int,
    column_end: int | None = None,
    feature_start: int = 0,
    feature_end: int = 0,
) -> None:
    for relative_row, bbox_row in enumerate(range(read_row_start, read_row_end)):
        store_row = store_row_start + bbox_row
        value_start = int(row_offsets[store_row])
        value_end = int(row_offsets[store_row + 1])
        if value_end == value_start:
            continue
        row_columns = np.asarray(
            columns[value_start:value_end], dtype=np.int64
        )
        selected: np.ndarray | None = None
        if column_end is not None:
            selected = (
                (row_columns >= int(column_start))
                & (row_columns < int(column_end))
            )
            if not np.any(selected):
                continue
            row_columns = row_columns[selected]
        dense_columns = row_columns - int(column_start)
        if features is None:
            destination[relative_row, dense_columns] = 1.0
        else:
            row_features = np.asarray(
                features[value_start:value_end, feature_start:feature_end],
                dtype=np.float32,
            )
            if selected is not None:
                row_features = row_features[selected]
            destination[relative_row, dense_columns, :] = row_features


def stream_smooth_sparse_store(
    store: SparseRowFeatureStore,
    *,
    output_path: str | Path,
    global_dim: int,
    first_size: int,
    second_size: int,
    channel_chunk: int = 1536,
    row_chunk: int = 128,
    workers: int = 1,
    backend: str = "cpu",
    device: torch.device | str | None = None,
    column_tile: int = 4096,
    gpu_algorithm: str = "auto",
    diagnostics: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Smooth sparse rows in bounded halos and write float16 directly.

    ``backend='gpu'`` applies the same box filters with CUDA and copies back
    only occupied output cells. ``backend='cpu'`` retains the OpenCV path.

    Channel chunks are independent.  ``workers > 1`` processes them in a
    bounded thread pool while forcing each OpenCV blur to use one native
    thread, avoiding nested oversubscription.  Every worker writes a disjoint
    output-channel slice, so the parallel path preserves the serial result.
    Each additional worker owns one dense float32 smoothing buffer; callers
    should keep the worker count modest when the grid is very wide.
    """

    if not store.closed:
        raise RuntimeError("close the sparse row store before smoothing")
    if global_dim <= 0 or global_dim >= store.feature_dim:
        raise ValueError("global_dim must split the feature dimension")
    if first_size <= 0 or second_size <= 0:
        raise ValueError("smoothing sizes must be positive")
    if channel_chunk <= 0 or row_chunk <= 0:
        raise ValueError("channel_chunk and row_chunk must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if column_tile <= 0:
        raise ValueError("column_tile must be positive")
    if gpu_algorithm not in {"auto", "prefix", "separable", "pool"}:
        raise ValueError(
            "gpu_algorithm must be 'auto', 'prefix', 'separable', or 'pool'"
        )
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    if backend == "gpu":
        device = torch.device("cuda" if device is None else device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPU smoothing requires an available CUDA device")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if store.num_features == 0:
        output_path.write_bytes(b"")
        return np.empty((0, store.feature_dim), dtype=np.float16)

    features = store.open_features()
    columns = store.open_columns()
    row_offsets = store.row_offsets
    row_lengths = np.diff(row_offsets)
    occupied_rows = np.flatnonzero(row_lengths > 0)
    first_occupied_row = int(occupied_rows[0])
    last_occupied_row = int(occupied_rows[-1])
    bbox_height = last_occupied_row - first_occupied_row + 1
    column_start = int(np.min(columns))
    column_end = int(np.max(columns)) + 1
    bbox_width = column_end - column_start

    output = np.memmap(
        output_path,
        dtype=np.float16,
        mode="w+",
        shape=(store.num_features, store.feature_dim),
    )
    if backend == "gpu":
        return _stream_smooth_sparse_store_gpu(
            store,
            output=output,
            global_dim=global_dim,
            first_size=first_size,
            second_size=second_size,
            channel_chunk=channel_chunk,
            row_chunk=row_chunk,
            device=device,
            column_tile=column_tile,
            gpu_algorithm=gpu_algorithm,
            diagnostics=diagnostics,
            progress_callback=progress_callback,
        )
    # OpenCV represents channel count inside its matrix type and rejects arrays
    # with more than 512 channels.  Treat the CLI value as an upper bound so a
    # large request (for example 1536) remains valid and is split safely.
    cv_channel_chunk = min(int(channel_chunk), 512)
    worker_count = int(workers)
    halves = (
        (0, int(global_dim), int(first_size)),
        (int(global_dim), store.feature_dim, int(second_size)),
    )

    executor: ThreadPoolExecutor | None = None
    previous_opencv_threads: int | None = None
    try:
        if worker_count > 1:
            # OpenCV normally parallelizes each blur internally.  Running that
            # inside several Python workers oversubscribes the same CPU cores,
            # so use one native OpenCV thread per independent channel task.
            previous_opencv_threads = cv2.getNumThreads()
            cv2.setNumThreads(1)
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="highres-smoothing",
            )

        for half_start, half_end, kernel_size in halves:
            top_halo = kernel_size // 2
            bottom_halo = kernel_size - 1 - top_halo
            for output_row_start in range(0, bbox_height, row_chunk):
                output_row_end = min(output_row_start + row_chunk, bbox_height)
                output_height = output_row_end - output_row_start
                output_store_row_start = (
                    first_occupied_row + output_row_start
                )
                output_store_row_end = first_occupied_row + output_row_end
                output_value_start = int(
                    row_offsets[output_store_row_start]
                )
                output_value_end = int(row_offsets[output_store_row_end])
                if output_value_end == output_value_start:
                    if progress_callback is not None:
                        progress_callback(
                            output_height * (half_end - half_start)
                        )
                    continue

                output_columns_absolute = np.asarray(
                    columns[output_value_start:output_value_end],
                    dtype=np.int64,
                )
                output_column_start = int(
                    np.min(output_columns_absolute)
                )
                output_column_end = int(
                    np.max(output_columns_absolute)
                ) + 1
                read_column_start = max(
                    column_start, output_column_start - top_halo
                )
                read_column_end = min(
                    column_end, output_column_end + bottom_halo
                )
                read_width = read_column_end - read_column_start

                read_row_start = max(0, output_row_start - top_halo)
                read_row_end = min(
                    bbox_height,
                    output_row_end + bottom_halo,
                )
                read_height = read_row_end - read_row_start
                occupied_grid = np.zeros(
                    (read_height, read_width), dtype=np.float32
                )
                _fill_sparse_window(
                    occupied_grid,
                    features=None,
                    columns=columns,
                    row_offsets=row_offsets,
                    store_row_start=first_occupied_row,
                    read_row_start=read_row_start,
                    read_row_end=read_row_end,
                    column_start=read_column_start,
                    column_end=read_column_end,
                )
                smoothed_count = cv2.blur(
                    occupied_grid,
                    (kernel_size, kernel_size),
                    borderType=cv2.BORDER_REFLECT,
                )
                crop_start = output_row_start - read_row_start

                output_rows: list[
                    tuple[int, int, int, np.ndarray]
                ] = []
                for bbox_row in range(output_row_start, output_row_end):
                    store_row = first_occupied_row + bbox_row
                    value_start = int(row_offsets[store_row])
                    value_end = int(row_offsets[store_row + 1])
                    if value_end == value_start:
                        continue
                    dense_columns = (
                        np.asarray(
                            columns[value_start:value_end],
                            dtype=np.int64,
                        )
                        - read_column_start
                    )
                    local_row = (
                        crop_start + bbox_row - output_row_start
                    )
                    output_rows.append(
                        (
                            value_start,
                            value_end,
                            local_row,
                            dense_columns,
                        )
                    )

                channel_ranges = [
                    (
                        channel_start,
                        min(channel_start + cv_channel_chunk, half_end),
                    )
                    for channel_start in range(
                        half_start,
                        half_end,
                        cv_channel_chunk,
                    )
                ]

                def smooth_channel_range(
                    channel_range: tuple[int, int],
                ) -> int:
                    channel_start, channel_end = channel_range
                    channel_width = channel_end - channel_start
                    dense_values = np.zeros(
                        (read_height, read_width, channel_width),
                        dtype=np.float32,
                    )
                    _fill_sparse_window(
                        dense_values,
                        features=features,
                        columns=columns,
                        row_offsets=row_offsets,
                        store_row_start=first_occupied_row,
                        read_row_start=read_row_start,
                        read_row_end=read_row_end,
                        column_start=read_column_start,
                        column_end=read_column_end,
                        feature_start=channel_start,
                        feature_end=channel_end,
                    )
                    # In-place filtering halves the Python-visible dense-buffer
                    # footprint, which is especially important with >1 worker.
                    smoothed_sum = cv2.blur(
                        dense_values,
                        (kernel_size, kernel_size),
                        dst=dense_values,
                        borderType=cv2.BORDER_REFLECT,
                    )
                    if smoothed_sum.ndim == 2:
                        smoothed_sum = smoothed_sum[..., None]

                    for (
                        value_start,
                        value_end,
                        local_row,
                        dense_columns,
                    ) in output_rows:
                        denominator = smoothed_count[local_row, dense_columns]
                        output[
                            value_start:value_end,
                            channel_start:channel_end,
                        ] = (
                            smoothed_sum[local_row, dense_columns, :]
                            / denominator[:, None]
                        ).astype(np.float16)
                    return (
                        output_height * channel_width
                    )

                if executor is None:
                    completed_work = map(smooth_channel_range, channel_ranges)
                else:
                    completed_work = executor.map(
                        smooth_channel_range,
                        channel_ranges,
                    )
                for work_units in completed_work:
                    if progress_callback is not None:
                        progress_callback(int(work_units))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if previous_opencv_threads is not None:
            cv2.setNumThreads(previous_opencv_threads)

    output.flush()
    return output


def _reflect_positions(indices: torch.Tensor, length: int) -> torch.Tensor:
    """Reflect arbitrary integer positions using OpenCV BORDER_REFLECT rules."""
    period = 2 * int(length)
    reflected = torch.remainder(indices, period)
    return torch.where(
        reflected < length,
        reflected,
        period - reflected - 1,
    )


def _target_reflect_indices(
    *,
    output_start: int,
    output_end: int,
    kernel_size: int,
    full_length: int,
    read_start: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return reflected local indices, or ``None`` for an identity span.

    Most row blocks and column tiles are away from the global bounding-box
    edges.  Their halo is already a contiguous slice, so avoiding an
    ``index_select`` saves one full tensor copy per filtering pass.
    """

    before = kernel_size // 2
    after = kernel_size - 1 - before
    if output_start - before >= 0 and output_end + after <= full_length:
        return None
    positions = torch.arange(
        output_start - before,
        output_end + after,
        dtype=torch.int64,
        device=device,
    )
    return _reflect_positions(positions, full_length) - int(read_start)


def _prefix_window_sum(
    values: torch.Tensor,
    *,
    kernel_size: int,
    dim: int,
) -> torch.Tensor:
    """Sliding window sums in O(N) work using a float32 prefix scan."""

    if kernel_size <= 0 or kernel_size > values.shape[dim]:
        raise ValueError("invalid prefix-window kernel size")
    output_length = int(values.shape[dim]) - int(kernel_size) + 1
    prefix = torch.cumsum(values, dim=dim, dtype=torch.float32)
    result = prefix.narrow(dim, kernel_size - 1, output_length).clone()
    if output_length > 1:
        result.narrow(dim, 1, output_length - 1).sub_(
            prefix.narrow(dim, 0, output_length - 1)
        )
    return result


def _resolve_gpu_box_algorithm(requested: str, kernel_size: int) -> str:
    """Static fallback choice for one embedding half."""

    if requested != "auto":
        return requested
    return "prefix" if kernel_size >= 16 else "pool"


def _autotune_gpu_box_algorithm(
    *,
    device: torch.device,
    kernel_size: int,
    row_chunk: int,
    column_tile: int,
    channel_chunk: int,
    bbox_height: int,
    bbox_width: int,
    diagnostics: bool,
) -> str:
    """Benchmark prefix, separable, and direct pooling on the actual CUDA GPU."""

    if kernel_size < 8 or device.type != "cuda":
        return _resolve_gpu_box_algorithm("auto", kernel_size)
    output_height = max(1, min(int(row_chunk), int(bbox_height)))
    output_width = max(1, min(int(column_tile), int(bbox_width)))
    source_height = output_height + int(kernel_size) - 1
    source_width = output_width + int(kernel_size) - 1
    sample_channels = max(1, min(int(channel_chunk), 64))
    timings: dict[str, float] = {}

    try:
        for candidate in ("prefix", "separable", "pool"):
            dtype = (
                torch.float16 if candidate == "prefix" else torch.float32
            )
            sample = torch.zeros(
                (sample_channels, source_height, source_width),
                dtype=dtype,
                device=device,
            )
            # Warm the allocator and the selected CUDA kernels before timing.
            warm = _gpu_box_filter_target(
                sample,
                kernel_size=kernel_size,
                row_indices=None,
                column_indices=None,
                algorithm=candidate,
            )
            del warm
            torch.cuda.synchronize(device)
            elapsed: list[float] = []
            for _ in range(3):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = _gpu_box_filter_target(
                    sample,
                    kernel_size=kernel_size,
                    row_indices=None,
                    column_indices=None,
                    algorithm=candidate,
                )
                end.record()
                end.synchronize()
                elapsed.append(float(start.elapsed_time(end)))
                del result
            timings[candidate] = float(np.median(elapsed))
            del sample
        chosen = min(timings, key=timings.get)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        # Prefix scans retain more float32 intermediates than pooling.  If the
        # representative tile cannot be benchmarked safely, prefer the lower-
        # memory separable path and release any failed allocation immediately.
        torch.cuda.empty_cache()
        chosen = "separable"
        timings = {}

    if diagnostics:
        if timings:
            detail = ", ".join(
                f"{name}={duration:.2f}ms"
                for name, duration in timings.items()
            )
            print(
                f"    {kernel_size}x{kernel_size} CUDA autotune: "
                f"{detail}; selected={chosen}"
            )
        else:
            print(
                f"    {kernel_size}x{kernel_size} CUDA autotune fell back "
                f"to {chosen}"
            )
    return chosen


def _gpu_box_filter_target(
    values: torch.Tensor,
    *,
    kernel_size: int,
    row_indices: torch.Tensor | None,
    column_indices: torch.Tensor | None,
    algorithm: str,
) -> torch.Tensor:
    """Filter one dense tile and return exactly its requested output core.

    ``prefix`` returns box *sums*. ``pool`` and ``separable`` return box
    averages.  Occupancy and feature values always use the same primitive, so
    their quotient is identical mathematically.
    """

    if values.ndim != 3:
        raise ValueError("values must have shape (channels, rows, columns)")

    if algorithm == "prefix":
        horizontal_source = (
            values
            if column_indices is None
            else values.index_select(2, column_indices)
        )
        horizontal = _prefix_window_sum(
            horizontal_source,
            kernel_size=kernel_size,
            dim=2,
        )
        vertical_source = (
            horizontal
            if row_indices is None
            else horizontal.index_select(1, row_indices)
        )
        return _prefix_window_sum(
            vertical_source,
            kernel_size=kernel_size,
            dim=1,
        )

    if algorithm == "separable":
        horizontal_source = (
            values
            if column_indices is None
            else values.index_select(2, column_indices)
        )
        horizontal = torch.nn.functional.avg_pool2d(
            horizontal_source[None],
            kernel_size=(1, kernel_size),
            stride=1,
        )[0]
        vertical_source = (
            horizontal
            if row_indices is None
            else horizontal.index_select(1, row_indices)
        )
        return torch.nn.functional.avg_pool2d(
            vertical_source[None],
            kernel_size=(kernel_size, 1),
            stride=1,
        )[0]

    if algorithm != "pool":
        raise ValueError(f"unknown GPU box-filter algorithm: {algorithm}")
    padded = values
    if row_indices is not None:
        padded = padded.index_select(1, row_indices)
    if column_indices is not None:
        padded = padded.index_select(2, column_indices)
    return torch.nn.functional.avg_pool2d(
        padded[None], kernel_size=kernel_size, stride=1
    )[0]


def _partition_active_columns(
    output_columns: np.ndarray,
    *,
    maximum_span: int,
    kernel_size: int,
) -> list[tuple[int, int]]:
    """Partition occupied output columns into bounded, gap-aware tiles."""

    unique_columns = np.unique(np.asarray(output_columns, dtype=np.int64))
    if unique_columns.size == 0:
        return []
    # Splitting across a sizeable empty gap avoids filtering zeros.  Tiny gaps
    # remain in one tile to avoid excessive CUDA launch and halo overhead.
    gap_threshold = max(64, 4 * int(kernel_size))
    tiles: list[tuple[int, int]] = []
    tile_start = int(unique_columns[0])
    previous = tile_start
    for column_value in unique_columns[1:]:
        column = int(column_value)
        exceeds_span = column - tile_start + 1 > int(maximum_span)
        crosses_gap = column - previous > gap_threshold
        if exceeds_span or crosses_gap:
            tiles.append((tile_start, previous + 1))
            tile_start = column
        previous = column
    tiles.append((tile_start, previous + 1))
    return tiles


@dataclass
class _GpuSmoothTile:
    output_local_indices: np.ndarray
    output_local_indices_device: torch.Tensor
    source_local_indices: np.ndarray
    sparse_linear: torch.Tensor
    output_rows: torch.Tensor
    output_columns: torch.Tensor
    inverse_counts: torch.Tensor
    read_width: int
    column_indices: torch.Tensor | None
    output_span: int


def _stream_smooth_sparse_store_gpu(
    store: SparseRowFeatureStore,
    *,
    output: np.ndarray,
    global_dim: int,
    first_size: int,
    second_size: int,
    channel_chunk: int,
    row_chunk: int,
    device: torch.device,
    column_tile: int,
    gpu_algorithm: str,
    diagnostics: bool,
    progress_callback: Callable[[int], None] | None,
) -> np.ndarray:
    """CUDA smoother with tile-major staging and row-major half writeback.

    The 16x16 pass no longer samples every element 16 times in each direction.
    It performs two float32 prefix scans and four differences.  Output-column
    tiles also prevent disconnected tissue islands from forcing work across a
    slide-wide empty span.
    """

    features = store.open_features()
    columns = store.open_columns()
    row_offsets = store.row_offsets
    row_lengths = np.diff(row_offsets)
    occupied_rows = np.flatnonzero(row_lengths > 0)
    first_occupied_row = int(occupied_rows[0])
    last_occupied_row = int(occupied_rows[-1])
    bbox_height = last_occupied_row - first_occupied_row + 1
    column_start = int(np.min(columns))
    bbox_width = int(np.max(columns)) + 1 - column_start
    halves = (
        (0, int(global_dim), int(first_size)),
        (int(global_dim), store.feature_dim, int(second_size)),
    )

    if diagnostics:
        bbox_cells = max(1, bbox_height * bbox_width)
        density = float(store.num_features) / float(bbox_cells)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            "    GPU smoothing geometry: "
            f"bbox={bbox_height}x{bbox_width}, "
            f"occupied={store.num_features:,} ({density:.2%}), "
            f"row_chunk={row_chunk}, column_tile={column_tile}, "
            f"channel_chunk={channel_chunk}, "
            f"free_cuda={free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB"
        )

    total_started = time.perf_counter()
    total_tiles = 0
    total_untiled_span = 0
    total_tiled_span = 0

    with torch.inference_mode():
        for half_start, half_end, kernel_size in halves:
            algorithm = (
                _autotune_gpu_box_algorithm(
                    device=device,
                    kernel_size=int(kernel_size),
                    row_chunk=int(row_chunk),
                    column_tile=int(column_tile),
                    channel_chunk=int(channel_chunk),
                    bbox_height=int(bbox_height),
                    bbox_width=int(bbox_width),
                    diagnostics=bool(diagnostics),
                )
                if gpu_algorithm == "auto"
                else _resolve_gpu_box_algorithm(
                    gpu_algorithm, int(kernel_size)
                )
            )
            half_started = time.perf_counter()
            half_tiles = 0
            half_untiled_span = 0
            half_tiled_span = 0
            before = kernel_size // 2
            after = kernel_size - 1 - before

            for output_row_start in range(0, bbox_height, row_chunk):
                output_row_end = min(output_row_start + row_chunk, bbox_height)
                output_height = output_row_end - output_row_start
                output_store_row_start = first_occupied_row + output_row_start
                output_store_row_end = first_occupied_row + output_row_end
                value_start = int(row_offsets[output_store_row_start])
                value_end = int(row_offsets[output_store_row_end])
                if value_end == value_start:
                    if progress_callback is not None:
                        progress_callback(
                            output_height * (half_end - half_start)
                        )
                    continue

                output_row_lengths = np.diff(
                    row_offsets[
                        output_store_row_start : output_store_row_end + 1
                    ]
                )
                output_rows_host = np.repeat(
                    np.arange(output_height, dtype=np.int64),
                    output_row_lengths,
                )
                output_columns_relative = (
                    np.asarray(
                        columns[value_start:value_end], dtype=np.int64
                    )
                    - column_start
                )
                output_min = int(np.min(output_columns_relative))
                output_max = int(np.max(output_columns_relative)) + 1
                untiled_span = output_max - output_min

                tile_bounds = _partition_active_columns(
                    output_columns_relative,
                    maximum_span=int(column_tile),
                    kernel_size=int(kernel_size),
                )

                read_row_start = max(0, output_row_start - before)
                read_row_end = min(bbox_height, output_row_end + after)
                read_height = read_row_end - read_row_start
                row_indices = _target_reflect_indices(
                    output_start=output_row_start,
                    output_end=output_row_end,
                    kernel_size=kernel_size,
                    full_length=bbox_height,
                    read_start=read_row_start,
                    device=device,
                )

                read_store_row_start = first_occupied_row + read_row_start
                read_store_row_end = first_occupied_row + read_row_end
                read_value_start = int(row_offsets[read_store_row_start])
                read_value_end = int(row_offsets[read_store_row_end])
                read_row_lengths = np.diff(
                    row_offsets[
                        read_store_row_start : read_store_row_end + 1
                    ]
                )
                sparse_rows_all = np.repeat(
                    np.arange(read_height, dtype=np.int64),
                    read_row_lengths,
                )
                sparse_columns_all = (
                    np.asarray(
                        columns[read_value_start:read_value_end],
                        dtype=np.int64,
                    )
                    - column_start
                )

                tiles: list[_GpuSmoothTile] = []
                for tile_start, tile_end in tile_bounds:
                    output_mask = (
                        (output_columns_relative >= tile_start)
                        & (output_columns_relative < tile_end)
                    )
                    output_local_indices = np.flatnonzero(output_mask)
                    if output_local_indices.size == 0:
                        continue

                    read_column_start = max(0, tile_start - before)
                    read_column_end = min(bbox_width, tile_end + after)
                    read_width = read_column_end - read_column_start
                    source_mask = (
                        (sparse_columns_all >= read_column_start)
                        & (sparse_columns_all < read_column_end)
                    )
                    source_local_indices = np.flatnonzero(source_mask)
                    sparse_linear_host = np.ascontiguousarray(
                        sparse_rows_all[source_mask] * read_width
                        + sparse_columns_all[source_mask]
                        - read_column_start,
                        dtype=np.int64,
                    )
                    sparse_linear = torch.from_numpy(
                        sparse_linear_host
                    ).to(device)

                    column_indices = _target_reflect_indices(
                        output_start=tile_start,
                        output_end=tile_end,
                        kernel_size=kernel_size,
                        full_length=bbox_width,
                        read_start=read_column_start,
                        device=device,
                    )

                    occupied_flat = torch.zeros(
                        read_height * read_width,
                        dtype=torch.float32,
                        device=device,
                    )
                    occupied_flat.index_fill_(0, sparse_linear, 1.0)
                    smoothed_count = _gpu_box_filter_target(
                        occupied_flat.view(1, read_height, read_width),
                        kernel_size=kernel_size,
                        row_indices=row_indices,
                        column_indices=column_indices,
                        algorithm=algorithm,
                    )[0]
                    tile_output_rows = torch.from_numpy(
                        np.ascontiguousarray(
                            output_rows_host[output_local_indices]
                        )
                    ).to(device)
                    tile_output_columns = torch.from_numpy(
                        np.ascontiguousarray(
                            output_columns_relative[output_local_indices]
                            - tile_start
                        )
                    ).to(device)
                    inverse_counts = smoothed_count[
                        tile_output_rows, tile_output_columns
                    ].reciprocal()
                    output_local_indices_device = torch.from_numpy(
                        np.ascontiguousarray(output_local_indices)
                    ).to(device)
                    tiles.append(
                        _GpuSmoothTile(
                            output_local_indices=output_local_indices,
                            output_local_indices_device=(
                                output_local_indices_device
                            ),
                            source_local_indices=source_local_indices,
                            sparse_linear=sparse_linear,
                            output_rows=tile_output_rows,
                            output_columns=tile_output_columns,
                            inverse_counts=inverse_counts,
                            read_width=read_width,
                            column_indices=column_indices,
                            output_span=tile_end - tile_start,
                        )
                    )

                tiled_span = sum(tile.output_span for tile in tiles)
                half_tiles += len(tiles)
                half_untiled_span += output_height * untiled_span
                half_tiled_span += output_height * tiled_span

                # The sparse store and output memmap are row-major.  Iterating
                # over 128-channel slices causes every 4 KiB file page to be
                # revisited many times and forces one synchronous D2H copy per
                # slice.  Stage all channels for one spatial tile once, compute
                # channel chunks entirely on CUDA, then copy/write one complete
                # embedding half for this row block.
                half_width = half_end - half_start
                half_result_device = torch.empty(
                    (value_end - value_start, half_width),
                    dtype=torch.float16,
                    device=device,
                )
                dense_dtype = (
                    torch.float16
                    if algorithm == "prefix"
                    else torch.float32
                )
                full_feature_block = features[
                    read_value_start:read_value_end,
                    half_start:half_end,
                ]

                for tile in tiles:
                    if (
                        tile.source_local_indices.size
                        == read_value_end - read_value_start
                    ):
                        feature_slice = full_feature_block
                    else:
                        feature_slice = full_feature_block[
                            tile.source_local_indices
                        ]

                    # One sequential row-major memmap gather and one H2D transfer
                    # per spatial tile, rather than one of each per channel chunk.
                    sparse_features_host = np.array(
                        np.asarray(feature_slice, dtype=np.float16),
                        dtype=np.float16,
                        order="C",
                        copy=True,
                    )
                    sparse_features_all = torch.from_numpy(
                        sparse_features_host
                    ).to(device=device, dtype=torch.float16)
                    del sparse_features_host, feature_slice

                    for channel_start in range(
                        half_start, half_end, channel_chunk
                    ):
                        channel_end = min(
                            channel_start + channel_chunk, half_end
                        )
                        channel_width = channel_end - channel_start
                        local_start = channel_start - half_start
                        local_end = channel_end - half_start

                        # Filtering expects channel-major dense values.  Perform
                        # this transpose on the GPU, where it is much cheaper than
                        # repeatedly transposing strided memmap slices on the CPU.
                        sparse_features = (
                            sparse_features_all[:, local_start:local_end]
                            .T.contiguous()
                        )
                        if dense_dtype != torch.float16:
                            sparse_features = sparse_features.to(dense_dtype)

                        dense_values = torch.zeros(
                            (
                                channel_width,
                                read_height * tile.read_width,
                            ),
                            dtype=dense_dtype,
                            device=device,
                        )
                        dense_values.index_copy_(
                            1, tile.sparse_linear, sparse_features
                        )
                        smoothed_values = _gpu_box_filter_target(
                            dense_values.view(
                                channel_width,
                                read_height,
                                tile.read_width,
                            ),
                            kernel_size=kernel_size,
                            row_indices=row_indices,
                            column_indices=tile.column_indices,
                            algorithm=algorithm,
                        )
                        values_at_cells = smoothed_values[
                            :, tile.output_rows, tile.output_columns
                        ].T
                        tile_result = (
                            values_at_cells
                            * tile.inverse_counts[:, None]
                        ).to(torch.float16)
                        half_result_device[
                            :, local_start:local_end
                        ].index_copy_(
                            0,
                            tile.output_local_indices_device,
                            tile_result,
                        )
                        del (
                            sparse_features,
                            dense_values,
                            smoothed_values,
                            values_at_cells,
                            tile_result,
                        )
                    del sparse_features_all

                # A complete half is contiguous on the source side and is written
                # with one NumPy assignment.  This cuts 12 D2H synchronizations and
                # 12 strided memmap writes to one when channel_chunk=128.
                result_block = half_result_device.cpu().numpy()
                output[
                    value_start:value_end,
                    half_start:half_end,
                ] = result_block
                del half_result_device, result_block, full_feature_block
                if progress_callback is not None:
                    progress_callback(output_height * half_width)

                del tiles

            half_elapsed = time.perf_counter() - half_started
            total_tiles += half_tiles
            total_untiled_span += half_untiled_span
            total_tiled_span += half_tiled_span
            if diagnostics:
                reduction = (
                    1.0 - half_tiled_span / max(1, half_untiled_span)
                )
                print(
                    f"    {kernel_size}x{kernel_size} half: "
                    f"algorithm={algorithm}, tiles={half_tiles:,}, "
                    f"empty-column work removed={reduction:.1%}, "
                    f"elapsed={half_elapsed:.1f}s"
                )

    output.flush()
    if diagnostics:
        elapsed = time.perf_counter() - total_started
        reduction = 1.0 - total_tiled_span / max(1, total_untiled_span)
        print(
            "    GPU smoothing total: "
            f"tiles={total_tiles:,}, "
            f"empty-column work removed={reduction:.1%}, "
            f"elapsed={elapsed:.1f}s"
        )
    return output

def smoothing_work_units(
    store: SparseRowFeatureStore,
    *,
    global_dim: int,
) -> int:
    """Exact progress total used by ``stream_smooth_sparse_store``."""

    if store.num_features == 0:
        return 0
    occupied_rows = np.flatnonzero(np.diff(store.row_offsets) > 0)
    bbox_height = int(occupied_rows[-1] - occupied_rows[0] + 1)
    return bbox_height * store.feature_dim


def save_sparse_npz(
    store: SparseRowFeatureStore,
    output_path: str | Path,
    *,
    embeddings: np.ndarray,
    pad: np.ndarray | Sequence[int],
) -> None:
    """Write the existing ``emb``, ``pos``, ``flag``, and ``pad`` contract."""

    embeddings = np.asarray(embeddings)
    if embeddings.shape != (store.num_features, store.feature_dim):
        raise ValueError("embeddings shape does not match the sparse store")
    if embeddings.dtype != np.float16:
        raise ValueError("embeddings must already be float16")
    pad_array = np.asarray(pad, dtype=np.int32)
    if pad_array.shape != (2,):
        raise ValueError("pad must contain row and column padding")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        emb=embeddings,
        pos=store.output_positions(),
        flag=np.ones((store.num_features,), dtype=bool),
        pad=pad_array,
    )


__all__ = [
    "RegularGridStripAccumulator",
    "StreamingStripStorePipeline",
    "SparseRowFeatureStore",
    "save_sparse_npz",
    "smoothing_work_units",
    "stream_smooth_sparse_store",
]
