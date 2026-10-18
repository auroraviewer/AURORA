"""Inference-side batching, transfer, and CPU accumulation helpers.

The helpers in this module deliberately separate three responsibilities:

* remove source patches that have no valid UNI tokens and repack the survivors
  into fixed-size inference batches;
* gather only valid spatial tokens on the model device before host transfer;
* feed compact host payloads to a bounded, ordered CPU accumulation worker.

Each component is usable independently, while the extraction loop composes all
three into one bounded inference pipeline.
"""

from __future__ import annotations

from collections import deque
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch


BatchPositions = tuple[torch.Tensor, torch.Tensor]
SourceBatch = tuple[torch.Tensor, Sequence[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ActivePatchBatch:
    """One inference batch containing only patches with valid spatial tokens."""

    patches: torch.Tensor
    positions: BatchPositions
    keep_flag: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.patches.shape[0])


@dataclass
class ActiveBatchStats:
    """Counters collected while filtering and repacking source batches."""

    source_patches: int = 0
    active_patches: int = 0
    skipped_patches: int = 0
    emitted_batches: int = 0
    full_batches: int = 0
    tail_batches: int = 0


@dataclass
class _BufferedChunk:
    patches: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    keep_flag: torch.Tensor
    offset: int = 0

    @property
    def remaining(self) -> int:
        return int(self.patches.shape[0]) - self.offset


class ActivePatchBatcher:
    """Filter empty source patches and repack survivors into full batches.

    The source is normally a ``DataLoader`` whose batches may become short
    after empty-patch removal.  This iterator buffers surviving rows across
    source-batch boundaries so every emitted batch except the final tail has
    exactly ``batch_size`` rows.  Source order is preserved.
    """

    def __init__(
        self,
        source: Iterable[SourceBatch],
        *,
        batch_size: int,
        pin_memory: bool = False,
        on_skipped_patches: Callable[[int], None] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if on_skipped_patches is not None and not callable(on_skipped_patches):
            raise TypeError("on_skipped_patches must be callable")
        self.source = source
        self.batch_size = int(batch_size)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.on_skipped_patches = on_skipped_patches
        self.stats = ActiveBatchStats()

    @staticmethod
    def _validate_source_batch(
        patches: torch.Tensor,
        positions: Sequence[torch.Tensor],
        keep_flag: torch.Tensor,
    ) -> BatchPositions:
        if patches.ndim < 1:
            raise ValueError(f"patches must be batched, got shape {patches.shape}")
        if keep_flag.ndim < 2:
            raise ValueError(
                f"keep_flag must be batched, got shape {keep_flag.shape}"
            )
        if len(positions) != 2:
            raise ValueError("positions must contain row and column tensors")
        rows, cols = positions
        batch_size = int(patches.shape[0])
        if keep_flag.shape[0] != batch_size:
            raise ValueError("patches and keep_flag must have the same batch size")
        if rows.shape != (batch_size,) or cols.shape != (batch_size,):
            raise ValueError(
                "row and column positions must each have shape "
                f"({batch_size},), got {rows.shape} and {cols.shape}"
            )
        if patches.device.type != "cpu" or keep_flag.device.type != "cpu":
            raise ValueError("ActivePatchBatcher expects CPU DataLoader tensors")
        if rows.device.type != "cpu" or cols.device.type != "cpu":
            raise ValueError("ActivePatchBatcher expects CPU position tensors")
        return rows, cols

    @staticmethod
    def _filter_chunk(
        patches: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        keep_flag: torch.Tensor,
        active: torch.Tensor,
    ) -> _BufferedChunk:
        if bool(active.all()):
            return _BufferedChunk(patches, rows, cols, keep_flag)
        indices = torch.nonzero(active, as_tuple=False).flatten()
        return _BufferedChunk(
            patches.index_select(0, indices),
            rows.index_select(0, indices),
            cols.index_select(0, indices),
            keep_flag.index_select(0, indices),
        )

    def _allocate_output(
        self,
        first: _BufferedChunk,
        size: int,
    ) -> ActivePatchBatch:
        common = {"device": "cpu", "pin_memory": self.pin_memory}
        patches = torch.empty(
            (size, *first.patches.shape[1:]),
            dtype=first.patches.dtype,
            **common,
        )
        rows = torch.empty((size,), dtype=first.rows.dtype, **common)
        cols = torch.empty((size,), dtype=first.cols.dtype, **common)
        keep_flag = torch.empty(
            (size, *first.keep_flag.shape[1:]),
            dtype=first.keep_flag.dtype,
            **common,
        )
        return ActivePatchBatch(patches, (rows, cols), keep_flag)

    def _pop_batch(
        self,
        chunks: deque[_BufferedChunk],
        size: int,
    ) -> ActivePatchBatch:
        if size <= 0:
            raise ValueError("output batch size must be positive")
        if not chunks:
            raise RuntimeError("cannot pop from an empty active-patch buffer")

        first = chunks[0]
        if (
            first.offset == 0
            and first.remaining == size
            and (not self.pin_memory or first.patches.is_pinned())
            and (not self.pin_memory or first.rows.is_pinned())
            and (not self.pin_memory or first.cols.is_pinned())
            and (not self.pin_memory or first.keep_flag.is_pinned())
        ):
            chunks.popleft()
            return ActivePatchBatch(
                first.patches,
                (first.rows, first.cols),
                first.keep_flag,
            )

        output = self._allocate_output(first, size)
        write_start = 0
        while write_start < size:
            chunk = chunks[0]
            take = min(size - write_start, chunk.remaining)
            source = slice(chunk.offset, chunk.offset + take)
            destination = slice(write_start, write_start + take)
            output.patches[destination].copy_(chunk.patches[source])
            output.positions[0][destination].copy_(chunk.rows[source])
            output.positions[1][destination].copy_(chunk.cols[source])
            output.keep_flag[destination].copy_(chunk.keep_flag[source])
            chunk.offset += take
            write_start += take
            if chunk.remaining == 0:
                chunks.popleft()
        return output

    def __iter__(self) -> Iterator[ActivePatchBatch]:
        self.stats = ActiveBatchStats()
        chunks: deque[_BufferedChunk] = deque()
        buffered = 0

        for patches, positions, keep_flag in self.source:
            rows, cols = self._validate_source_batch(
                patches,
                positions,
                keep_flag,
            )
            source_size = int(patches.shape[0])
            active = keep_flag.bool().reshape(source_size, -1).any(dim=1)
            active_size = int(active.sum().item())

            self.stats.source_patches += source_size
            self.stats.active_patches += active_size
            skipped_size = source_size - active_size
            self.stats.skipped_patches += skipped_size
            if skipped_size and self.on_skipped_patches is not None:
                self.on_skipped_patches(skipped_size)
            if active_size == 0:
                continue

            chunks.append(
                self._filter_chunk(
                    patches,
                    rows,
                    cols,
                    keep_flag,
                    active,
                )
            )
            buffered += active_size

            while buffered >= self.batch_size:
                yield self._pop_batch(chunks, self.batch_size)
                buffered -= self.batch_size
                self.stats.emitted_batches += 1
                self.stats.full_batches += 1

        if buffered:
            yield self._pop_batch(chunks, buffered)
            self.stats.emitted_batches += 1
            self.stats.tail_batches += 1


@dataclass
class CpuFeatureBatch:
    """Compact model output staged for ordered CPU accumulation.

    ``spatial_host`` contains one row per valid token rather than the dense
    ``B x N`` token grid.  CUDA callers receive a readiness event and retained
    device tensors so the worker can wait without synchronizing the inference
    thread.
    """

    global_host: torch.Tensor
    spatial_host: torch.Tensor
    batch_indices: np.ndarray
    pixel_positions: np.ndarray
    ready_event: object | None = None
    device_refs: tuple[torch.Tensor, ...] = ()


def stage_valid_feature_batch(
    *,
    global_features: torch.Tensor,
    spatial_features: torch.Tensor,
    batch_indices: np.ndarray,
    token_indices: np.ndarray,
    pixel_positions: np.ndarray,
    copy_stream: torch.cuda.Stream | None,
) -> CpuFeatureBatch:
    """Gather valid spatial tokens on-device and stage compact host tensors.

    CUDA copies are enqueued on ``copy_stream`` after it waits for the current
    compute stream.  The returned event is synchronized by the CPU worker, so
    the inference thread can immediately launch the next model batch.
    """

    if global_features.ndim != 2:
        raise ValueError(
            "global_features must have shape (B, G), got "
            f"{tuple(global_features.shape)}"
        )
    if spatial_features.ndim != 3:
        raise ValueError(
            "spatial_features must have shape (B, N, S), got "
            f"{tuple(spatial_features.shape)}"
        )
    if global_features.shape[0] != spatial_features.shape[0]:
        raise ValueError(
            "global_features and spatial_features must have the same batch size"
        )
    if global_features.device != spatial_features.device:
        raise ValueError(
            "global_features and spatial_features must be on the same device"
        )

    batch_indices_array = np.ascontiguousarray(batch_indices, dtype=np.int64)
    token_indices_array = np.ascontiguousarray(token_indices, dtype=np.int64)
    pixel_positions_array = np.ascontiguousarray(pixel_positions, dtype=np.int32)
    num_valid = int(batch_indices_array.shape[0])
    if batch_indices_array.shape != (num_valid,):
        raise ValueError("batch_indices must be one-dimensional")
    if token_indices_array.shape != (num_valid,):
        raise ValueError(
            f"token_indices must have shape ({num_valid},), got "
            f"{token_indices_array.shape}"
        )
    if pixel_positions_array.shape != (num_valid, 2):
        raise ValueError(
            f"pixel_positions must have shape ({num_valid}, 2), got "
            f"{pixel_positions_array.shape}"
        )

    batch_size, token_count, spatial_dim = spatial_features.shape
    if num_valid:
        if batch_indices_array.min() < 0 or batch_indices_array.max() >= batch_size:
            raise IndexError("batch_indices contains an out-of-range patch index")
        if token_indices_array.min() < 0 or token_indices_array.max() >= token_count:
            raise IndexError("token_indices contains an out-of-range token index")

    device = spatial_features.device
    batch_index_device = torch.as_tensor(
        batch_indices_array,
        dtype=torch.long,
        device=device,
    )
    token_index_device = torch.as_tensor(
        token_indices_array,
        dtype=torch.long,
        device=device,
    )
    linear_indices = batch_index_device * token_count + token_index_device
    flattened_spatial = spatial_features.reshape(batch_size * token_count, spatial_dim)
    valid_spatial_device = flattened_spatial.index_select(0, linear_indices)

    global_half = global_features.detach().to(dtype=torch.float16).contiguous()
    spatial_half = valid_spatial_device.detach().to(dtype=torch.float16).contiguous()

    if device.type != "cuda":
        return CpuFeatureBatch(
            global_host=global_half.cpu(),
            spatial_host=spatial_half.cpu(),
            batch_indices=batch_indices_array,
            pixel_positions=pixel_positions_array,
            ready_event=None,
            device_refs=(),
        )

    if copy_stream is None:
        copy_stream = torch.cuda.Stream(device=device)
    global_host = torch.empty(
        global_half.shape,
        dtype=torch.float16,
        device="cpu",
        pin_memory=True,
    )
    spatial_host = torch.empty(
        spatial_half.shape,
        dtype=torch.float16,
        device="cpu",
        pin_memory=True,
    )
    compute_stream = torch.cuda.current_stream(device=device)
    copy_stream.wait_stream(compute_stream)
    ready_event = torch.cuda.Event(blocking=False, interprocess=False)
    with torch.cuda.stream(copy_stream):
        global_host.copy_(global_half, non_blocking=True)
        spatial_host.copy_(spatial_half, non_blocking=True)
        ready_event.record(copy_stream)

    return CpuFeatureBatch(
        global_host=global_host,
        spatial_host=spatial_host,
        batch_indices=batch_indices_array,
        pixel_positions=pixel_positions_array,
        ready_event=ready_event,
        device_refs=(global_half, spatial_half),
    )


_QUEUE_SENTINEL = object()


class CpuAccumulatorPipeline:
    """Bounded ordered worker for compact CPU feature accumulation.

    CUDA readiness is waited on by the worker thread rather than the inference
    thread.  Because one worker owns all accumulator mutation, no locks are
    needed inside ``SplitGridFeatureAccumulator`` and merge order remains the
    same as submission order.
    """

    def __init__(
        self,
        *,
        fine_state: object | None,
        direct_states: dict[int, object],
        token_size: int,
        max_pending: int = 2,
        on_processed_patches: Callable[[int], None] | None = None,
    ) -> None:
        if token_size <= 0:
            raise ValueError("token_size must be positive")
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        if any(int(size) <= 0 for size in direct_states):
            raise ValueError("direct-state patch sizes must be positive")
        if on_processed_patches is not None and not callable(on_processed_patches):
            raise TypeError("on_processed_patches must be callable")

        self.fine_state = fine_state
        self.direct_states = dict(direct_states)
        self.token_size = int(token_size)
        self.max_pending = int(max_pending)
        self.on_processed_patches = on_processed_patches
        self.processed_payloads = 0
        self.processed_patches = 0
        self.processed_contributions = 0

        self._queue: queue.Queue[CpuFeatureBatch | object] = queue.Queue(
            maxsize=self.max_pending
        )
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_main,
            name="highres-cpu-accumulator",
            daemon=True,
        )
        self._worker.start()

    def _set_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _get_error(self) -> BaseException | None:
        with self._error_lock:
            return self._error

    def _raise_if_failed(self) -> None:
        error = self._get_error()
        if error is not None:
            raise RuntimeError("CPU accumulation worker failed") from error

    def _put_with_failure_check(self, item: CpuFeatureBatch | object) -> None:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def submit(self, payload: CpuFeatureBatch) -> None:
        if self._closed:
            raise RuntimeError("cannot submit after the CPU pipeline is finished")
        if not isinstance(payload, CpuFeatureBatch):
            raise TypeError("payload must be a CpuFeatureBatch")
        self._put_with_failure_check(payload)

    def _process(self, payload: CpuFeatureBatch) -> None:
        try:
            if payload.ready_event is not None:
                synchronize = getattr(payload.ready_event, "synchronize", None)
                if synchronize is None:
                    raise TypeError("ready_event must provide synchronize()")
                synchronize()

            if payload.global_host.device.type != "cpu":
                raise ValueError("global_host must be a CPU tensor")
            if payload.spatial_host.device.type != "cpu":
                raise ValueError("spatial_host must be a CPU tensor")

            global_embeddings = payload.global_host.detach().numpy()
            spatial_tokens = payload.spatial_host.detach().numpy()
            batch_indices = np.asarray(payload.batch_indices, dtype=np.int64)
            pixel_positions = np.asarray(payload.pixel_positions, dtype=np.int32)

            if self.fine_state is not None:
                self.fine_state.merge(
                    pixel_positions // self.token_size,
                    global_embeddings,
                    spatial_tokens,
                    batch_indices,
                    None,
                )
            for patch_size, state in self.direct_states.items():
                state.merge(
                    pixel_positions // int(patch_size),
                    global_embeddings,
                    spatial_tokens,
                    batch_indices,
                    None,
                )

            self.processed_payloads += 1
            processed_patches = int(global_embeddings.shape[0])
            self.processed_patches += processed_patches
            self.processed_contributions += int(batch_indices.shape[0])
            if self.on_processed_patches is not None:
                self.on_processed_patches(processed_patches)
        finally:
            # Retaining these tensors until after event synchronization prevents
            # the CUDA caching allocator from reusing copy sources prematurely.
            payload.device_refs = ()
            payload.ready_event = None

    def _worker_main(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _QUEUE_SENTINEL:
                    return
                if not isinstance(item, CpuFeatureBatch):
                    raise TypeError("unexpected item in CPU accumulation queue")
                self._process(item)
            except BaseException as exc:  # propagated by submit()/finish()
                self._set_error(exc)
                return
            finally:
                self._queue.task_done()

    def finish(self) -> None:
        if not self._closed:
            self._closed = True
            if self._get_error() is None:
                try:
                    self._put_with_failure_check(_QUEUE_SENTINEL)
                except RuntimeError:
                    # The worker failed while the sentinel was being queued.
                    pass
        self._worker.join()
        self._raise_if_failed()

    def __enter__(self) -> "CpuAccumulatorPipeline":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.finish()
        except RuntimeError:
            if exc is None:
                raise
        return False


__all__ = [
    "ActiveBatchStats",
    "ActivePatchBatch",
    "ActivePatchBatcher",
    "CpuAccumulatorPipeline",
    "CpuFeatureBatch",
    "stage_valid_feature_batch",
]
