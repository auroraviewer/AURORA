# from pyexpat import model
import cv2
import numpy as np
import h5py
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from typing import Tuple
from torch import nn

def plot_image_with_spot(image, loc, spot_radius_pixel, mask_matrix):
    # Create a copy of the image to draw spots with transparency
    image_with_spots = image.copy()
    # Draw spots on the image
    for (x, y) in loc:
        #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
        #! For cv2, (x,y) is (col_index, row_index)
        image_with_spots = cv2.circle(image_with_spots, (int(x), int(y)), int(spot_radius_pixel), (0, 255, 0), -1)
    # Create a mask for the spots
    # spots_mask = np.zeros_like(image_with_spots, dtype=np.uint8)
    # for (x, y) in loc:
    #     spots_mask = cv2.circle(spots_mask, (int(x), int(y)), int(spot_radius_pixel), (0, 255, 0), -1)
    # Blend the spots with the original image
    image_with_spots = cv2.addWeighted(image, 0.7, image_with_spots, 0.3, 0)
    image_with_spots[mask_matrix == 0] = [255, 255, 255]  # Color the masked area as white
    image_with_spots = cv2.resize(image_with_spots, (0,0), fx=0.1, fy=0.1)
    return image_with_spots


def cut_and_rescale_image(image, mask, loc, raw_pixel_size, target_pixel_size = 0.5):
    scale_factor = raw_pixel_size/target_pixel_size
    # Cut the white border of the image
    # Find the bounding box of the region of interest in the mask
    coords = np.column_stack(np.where(mask > 0))
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    image = image[row_min:row_max, col_min:col_max]
    mask = mask[row_min:row_max, col_min:col_max]
    #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
    if loc is not None:
        loc = loc - [col_min, row_min]
    # Rescale the image
    image = cv2.resize(image, (0,0), fx=scale_factor, fy=scale_factor)
    mask = cv2.resize(mask.astype('float32'), (0,0), fx=scale_factor, fy=scale_factor)
    if loc is not None:
        loc = loc * scale_factor
        loc = loc.round().astype(int)
    return image, mask, loc, (row_min, row_max, col_min, col_max, scale_factor)

def get_UNI_model(token, version = "UNI2-h", from_local = False):
    if not from_local:
        login(token=token)
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        print(f"Loading model: {version}")
        if version == "UNI":
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        elif version == "UNI2-h":
            timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
                }
            model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        print(f"Loading model from local path: {token}")
        if version == "UNI":
            model = timm.create_model(f'local-dir:{token}', pretrained=True, init_values=1e-5, dynamic_img_size=True)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        elif version == "UNI2-h":
            timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
                }
            # Not token is the local path to the model weights (.bin)
            model = timm.create_model(f'local-dir:{token}', pretrained=True, **timm_kwargs)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    print(f"Finished loading model: {version}")
    print("global_pool:", getattr(model, "global_pool", None))
    print("token-level final norm:", getattr(model, "norm", None))
    print("post-pooling norm:", getattr(model, "fc_norm", None))
    fused_attention_flags = [
        getattr(block.attn, "fused_attn", False)
        for block in model.blocks
        if hasattr(block, "attn")
    ]
    print(
        "Fused SDPA enabled:",
        bool(fused_attention_flags) and all(fused_attention_flags),
    )
    return model, transform

# @torch.inference_mode()
# def extract_highres_features(
#     model: torch.nn.Module,
#     batch: torch.Tensor,
#     amp_dtype: torch.dtype,
#     device_type: str
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     with torch.autocast(device_type="cuda",dtype=amp_dtype):
#         final_tokens, intermediate_features = model.forward_intermediates(
#             batch,
#             indices=1,                    # take the final transformer block
#             return_prefix_tokens=False,   # remove CLS/register tokens from intermediates
#             norm=True,                    # normalize patch embeddings
#             output_fmt="NCHW",            # [B, embedding_dim, grid_h, grid_w]
#         )
#         # Global embedding before any classification layer.
#         global_emb = model.forward_head(
#             final_tokens,
#             pre_logits=True,
#         )
#     # Spatial features from the final transformer block.
#     patch_emb = intermediate_features[-1]

#     return global_emb, patch_emb

HEST_PATCH_META_KEYS = ['downsample', 'patch_size_src', 'patch_size_target', 'pixel_size']


def _hest_patch_meta(handle, h5_file_path, img, meta_key):
    """Read the patch metadata, tolerating dumps that do not carry it.

    HEST's `dump_patches` does not always store these four values as attributes on
    `img` -- the AURORA training patches have none at all -- and the consumer only
    writes them to `locs/<name>-meta.pickle`, which nothing reads back. So fall back
    through the other places they can live instead of raising KeyError:

      1. attributes on the `img` dataset (the original behaviour);
      2. attributes on the file itself;
      3. a sibling ``metadata/<name>.json`` that HEST's preprocessor also writes;
      4. values derivable from the data, with a printed warning.
    """
    meta = {}
    file_attrs = dict(handle.attrs)
    sidecar = {}
    try:
        import json
        import os
        stem = os.path.basename(h5_file_path).rsplit('.h5', 1)[0]
        json_path = os.path.join(os.path.dirname(os.path.dirname(h5_file_path)),
                                 'metadata', f'{stem}.json')
        if os.path.isfile(json_path):
            with open(json_path) as fh:
                sidecar = json.load(fh)
    except Exception:
        sidecar = {}

    for key in meta_key:
        if key in handle['img'].attrs:
            meta[key] = handle['img'].attrs[key]
        elif key in file_attrs:
            meta[key] = file_attrs[key]
        elif isinstance(sidecar, dict) and key in sidecar:
            meta[key] = sidecar[key]
        else:
            meta[key] = None

    if any(value is None for value in meta.values()):
        # Derive what can be derived; the rest are never read back, so a default is
        # honest as long as it is announced.
        patch_size = int(img.shape[1]) if getattr(img, 'ndim', 0) >= 2 else None
        for key, fallback in (
            ('downsample', 1.0),
            ('patch_size_src', patch_size),
            ('patch_size_target', patch_size),
            ('pixel_size', 0.5),
        ):
            if meta.get(key) is None:
                meta[key] = fallback
        print(f"    note: {h5_file_path} has no {', '.join(meta_key)} metadata; "
              f"derived what could be: {meta}")
    return meta


def get_HEST1k_patch(h5_file_path):
    # Open the .h5 file
    print(f"Fetching patches in {h5_file_path.replace('.h5', '')}")
    img_meta = {}
    meta_key = ['downsample', 'patch_size_src', 'patch_size_target', 'pixel_size']
    with h5py.File(h5_file_path, 'r') as f:
        # HEST's dump_patches writes 'barcodes' (plural); older dumps and the rest of
        # this codebase use 'barcode'. Accept either rather than requiring one.
        # barcode = f['barcode'][:]
        barcode = f['barcode'][:] if 'barcode' in f else f['barcodes'][:]
        img = f['img'][:]
        coords = f['coords'][:]
        # for key in meta_key:
        #     img_meta[key] = f['img'].attrs[key]
        img_meta = _hest_patch_meta(f, h5_file_path, img, meta_key)
    print(f"Finished fetching patches in {h5_file_path.replace('.h5', '')}")
    return img, img_meta, barcode, coords

def smooth_uniform_cv(x, size):
    kernel = np.ones((size, size), np.float32) / (size ** 2)
    return cv2.filter2D(x, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)

def smooth_embeddings(embeddings, positions, flags, size, position_unit):
    valid_flags = np.asarray(flags).astype(bool)
    valid_positions = np.asarray(positions)[valid_flags]

    if valid_positions.size == 0:
        return embeddings

    spot_pos = np.floor(valid_positions / position_unit).astype(np.int64)
    offset = spot_pos.min(axis=0)
    spot_pos = spot_pos - offset

    unique_spot_pos = np.unique(spot_pos, axis=0)
    if unique_spot_pos.shape[0] != spot_pos.shape[0]:
        raise ValueError(
            f"spot_pos must be unique after conversion to patch units; got "
            f"{spot_pos.shape[0] - unique_spot_pos.shape[0]} duplicate positions"
        )

    grid_shape = tuple(spot_pos.max(axis=0) + 1)
    grid_count = np.zeros(grid_shape, dtype=np.float32)

    rows = spot_pos[:, 0]
    cols = spot_pos[:, 1]
    np.add.at(grid_count, (rows, cols), 1.0)

    occupied = grid_count > 0
    valid_mask = occupied.astype(np.float32)
    smoothed_count = cv2.blur(valid_mask, (size, size), borderType=cv2.BORDER_REFLECT)
    output_indices = np.flatnonzero(valid_flags)
    valid_rows = spot_pos[:, 0]
    valid_cols = spot_pos[:, 1]
    nonzero = smoothed_count[valid_rows, valid_cols] > 0

    smoothed_embeddings = np.full_like(embeddings, np.nan, dtype=np.float32)
    channel_chunk = 64
    for start in range(0, embeddings.shape[1], channel_chunk):
        end = min(start + channel_chunk, embeddings.shape[1])
        chunk_embeddings = embeddings[valid_flags, start:end].astype(np.float32, copy=False)
        grid_sum_chunk = np.zeros((*grid_shape, end - start), dtype=np.float32)
        np.add.at(grid_sum_chunk, (rows, cols), chunk_embeddings)
        smoothed_sum_chunk = cv2.blur(
            grid_sum_chunk,
            (size, size),
            borderType=cv2.BORDER_REFLECT,
        )
        smoothed_embeddings[output_indices[nonzero], start:end] = (
            smoothed_sum_chunk[valid_rows[nonzero], valid_cols[nonzero], :] /
            smoothed_count[valid_rows[nonzero], valid_cols[nonzero]][..., np.newaxis]
        )
    return smoothed_embeddings


def smooth_embedding_halves(embeddings, positions, flags, position_unit, first_size=16, second_size=4):
    if embeddings.ndim != 2 or embeddings.shape[1] != 1536*2:
        raise ValueError(f"Expected embeddings with shape (num_spots, 3072), got {embeddings.shape}")

    first_half = smooth_embeddings(
        embeddings[:, :1536], positions, flags, size=first_size, position_unit=position_unit)
    second_half = smooth_embeddings(
        embeddings[:, 1536:], positions, flags, size=second_size, position_unit=position_unit)
    return np.concatenate([first_half, second_half], axis=1)

def aggregate_shifted_npz_files(npz_paths, output_path, position_unit):
    if len(npz_paths) == 0:
        raise FileNotFoundError(f"No shifted embedding files found for {output_path}")
    global_min = None
    global_max = None
    feature_dim = None
    for npz_path in npz_paths:
        with np.load(npz_path) as result:
            positions = result['pos'].astype(np.int32, copy=False)
            if feature_dim is None:
                feature_dim = int(result['emb'].shape[1])
            if positions.size == 0:
                continue
            pos_min = positions.min(axis=0)
            pos_max = positions.max(axis=0)
            global_min = pos_min if global_min is None else np.minimum(global_min, pos_min)
            global_max = pos_max if global_max is None else np.maximum(global_max, pos_max)

    if global_min is None or global_max is None or feature_dim is None:
        raise ValueError(f"No valid shifted embeddings found for {output_path}")
    if position_unit is None:
        raise ValueError(f"Unable to infer position unit for {output_path}")

    grid_origin = np.rint(global_min / position_unit).astype(np.int64)
    grid_limit = np.rint(global_max / position_unit).astype(np.int64)
    grid_shape = tuple((grid_limit - grid_origin + 1).tolist())

    sum_grid = np.zeros((*grid_shape, feature_dim), dtype=np.float32)
    count_grid = np.zeros(grid_shape, dtype=np.float32)
    flag_grid = np.zeros(grid_shape, dtype=bool)

    for npz_path in npz_paths:
        with np.load(npz_path) as result:
            embeddings = result['emb'].astype(np.float32, copy=False)
            positions = result['pos'].astype(np.int32, copy=False)
            flags = result['flag'].astype(bool, copy=False)
            if positions.size == 0:
                continue
            grid_positions = np.rint(positions / position_unit).astype(np.int64) - grid_origin
            rows = grid_positions[:, 0]
            cols = grid_positions[:, 1]
            np.add.at(sum_grid, (rows, cols), embeddings)
            np.add.at(count_grid, (rows, cols), 1.0)
            np.logical_or.at(flag_grid, (rows, cols), flags)

    occupied = count_grid > 0
    occupied_indices = np.argwhere(occupied)
    num_positions = occupied_indices.shape[0]
    emb_mean = np.empty((num_positions, feature_dim), dtype=np.float32)
    positions = np.empty((num_positions, 2), dtype=np.int32)
    flags = np.empty(num_positions, dtype=bool)

    for idx, (row, col) in enumerate(occupied_indices):
        positions[idx] = ((np.array([row, col], dtype=np.int64) + grid_origin) * position_unit).astype(np.int32)
        emb_mean[idx] = sum_grid[row, col] / count_grid[row, col]
        flags[idx] = flag_grid[row, col]

    emb_mean = emb_mean.astype(np.float16)
    # np.savez(output_path, emb=emb_mean, pos=positions, flag=flags, pad=np.array([0, 0], dtype=np.int32))
    return emb_mean, positions, flags

class _PatchAccumulator:
    """Chunked accumulation of subpixel embeddings per grid cell.

    Positions are grid coordinates (subpixel position // patch_size).
    ``merge`` appends each batch's (deduplicated) contribution as a chunk in
    O(batch) time -- no growing flat arrays, so the per-batch cost stays flat
    as an image progresses (avoids the O(N^2) slowdown of re-copying the whole
    state on every batch). Chunks are periodically compacted (duplicates
    merged) with an amortized budget that doubles, bounding total work to
    ~O(N log N) and peak memory to ~2x the deduplicated state.
    ``finalize`` produces the same (emb, pos, flag) contract as the original
    ``_finalize_patch_state`` (row-major sorted by (row, col), means averaged).
    """

    def __init__(self, feature_dim: int, chunk_budget: int = 100_000):
        self.feature_dim = feature_dim
        self.chunk_budget = chunk_budget
        self._chunks = []   # list of (positions int32 (N,2), sums float32 (N,D), counts int32 (N,))
        self._size = 0

    @staticmethod
    def _group_offsets(inverse: np.ndarray):
        """Stable-sort order and group start/end boundaries for inverse ids."""
        order = np.argsort(inverse, kind="stable")
        inv_sorted = inverse[order]
        boundaries = np.flatnonzero(np.diff(inv_sorted)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([starts[1:], [inverse.shape[0]]])
        return order, starts, ends

    @staticmethod
    def _reduceat_chunks(colcat: np.ndarray, order: np.ndarray, starts: np.ndarray,
                         n_uniq: int, chunk: int = 64) -> np.ndarray:
        """Grouped sums over columns of ``colcat`` (N, n_cols), channel-chunked
        so peak memory is bounded by (N, chunk) instead of (N, n_cols)."""
        n_cols = colcat.shape[1]
        sums = np.empty((n_uniq, n_cols), dtype=np.float32)
        for c0 in range(0, n_cols, chunk):
            c1 = min(c0 + chunk, n_cols)
            sums[:, c0:c1] = np.add.reduceat(colcat[order, c0:c1], starts, axis=0)
        return sums

    def merge(self, grid_positions: np.ndarray, embeddings: np.ndarray) -> None:
        """Accumulate float16 embeddings keyed by integer grid positions (N, 2)."""
        if grid_positions.shape[0] == 0:
            return
        emb_f32 = embeddings.astype(np.float32, copy=False)
        uniq, inverse = np.unique(grid_positions, axis=0, return_inverse=True)
        order, starts, ends = self._group_offsets(inverse)
        chunk_sum = self._reduceat_chunks(emb_f32, order, starts, uniq.shape[0])
        chunk_count = (ends - starts).astype(np.int32)
        self._chunks.append((uniq, chunk_sum, chunk_count))
        self._size += uniq.shape[0]
        if self._size >= self.chunk_budget:
            self._compact()

    def _compact(self) -> None:
        """Merge all buffered chunks into one deduplicated chunk."""
        if len(self._chunks) <= 1:
            return
        all_pos = np.concatenate([c[0] for c in self._chunks], axis=0)
        all_cnt = np.concatenate([c[2] for c in self._chunks], axis=0)
        uniq, inverse = np.unique(all_pos, axis=0, return_inverse=True)
        order, starts, ends = self._group_offsets(inverse)
        sums = np.empty((uniq.shape[0], self.feature_dim), dtype=np.float32)
        for c0 in range(0, self.feature_dim, 64):
            c1 = min(c0 + 64, self.feature_dim)
            # colcat is already the [c0:c1] column slice of every chunk, so only
            # reorder rows (colcat[order]) -- do NOT slice columns again.
            colcat = np.concatenate([c[1][:, c0:c1] for c in self._chunks], axis=0)
            sums[:, c0:c1] = np.add.reduceat(colcat[order], starts, axis=0)
        counts = np.add.reduceat(all_cnt[order], starts, axis=0).astype(np.int32)
        self._chunks = [(uniq, sums, counts)]
        self._size = uniq.shape[0]
        # If still at/above budget (positions mostly unique), double the budget
        # so we do not re-compact on every following merge (amortized O(N log N)).
        if self._size >= self.chunk_budget:
            self.chunk_budget = self._size * 2

    def finalize(self, patch_size: int):
        self._compact()
        if self._size == 0:
            return (
                np.empty((0, self.feature_dim), dtype=np.float16),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=bool),
            )
        if len(self._chunks) == 1:
            positions, sums, counts = self._chunks[0]
        else:
            positions = np.concatenate([c[0] for c in self._chunks], axis=0)
            sums = np.concatenate([c[1] for c in self._chunks], axis=0)
            counts = np.concatenate([c[2] for c in self._chunks], axis=0)
        order = np.lexsort((positions[:, 1], positions[:, 0]))
        positions = positions[order]
        sums = sums[order]
        counts = counts[order]
        flags = np.ones(positions.shape[0], dtype=bool)
        emb = (sums / counts[:, None]).astype(np.float16)
        pos = (positions * patch_size).astype(np.int32)
        return emb, pos, flags

# def _merge_embeddings_into_state(state, embeddings, positions, patch_size):
#     if embeddings.shape[0] == 0:
#         return
#     grid_positions = positions // patch_size
#     unique_positions, inverse_indices = np.unique(grid_positions, axis=0, return_inverse=True)
#     chunk_embeddings = embeddings.astype(np.float32, copy=False)
#     chunk_sum = np.zeros((unique_positions.shape[0], chunk_embeddings.shape[1]), dtype=np.float32)
#     chunk_count = np.zeros(unique_positions.shape[0], dtype=np.int32)
#     np.add.at(chunk_sum, inverse_indices, chunk_embeddings)
#     np.add.at(chunk_count, inverse_indices, 1)

#     for idx, pos in enumerate(unique_positions):
#         key = (int(pos[0]), int(pos[1]))
#         if key in state["sum"]:
#             state["sum"][key] += chunk_sum[idx]
#             state["count"][key] += int(chunk_count[idx])
#         else:
#             state["sum"][key] = chunk_sum[idx]
#             state["count"][key] = int(chunk_count[idx])

# def _finalize_patch_state(state, patch_size):
#     if len(state["sum"]) == 0:
#         feature_dim = None
#         output_dim = feature_dim if feature_dim is not None else 3072
#         return (
#             np.empty((0, output_dim), dtype=np.float16),
#             np.empty((0, 2), dtype=np.int32),
#             np.empty((0,), dtype=bool),
#         )

#     keys = np.array(list(state["sum"].keys()), dtype=np.int32)
#     order = np.lexsort((keys[:, 1], keys[:, 0]))
#     keys = keys[order]
#     embeddings = np.stack([state["sum"][(int(row), int(col))] for row, col in keys], axis=0)
#     counts = np.array([state["count"][(int(row), int(col))] for row, col in keys], dtype=np.int32)
#     flags = np.ones(keys.shape[0], dtype=bool)
#     patch_embeddings_curr = (embeddings / counts[:, None]).astype(np.float16)
#     patch_positions_curr = (keys * patch_size).astype(np.int32)
#     return patch_embeddings_curr, patch_positions_curr, flags

class HighResFeatureExtractor(nn.Module):
    """
    Optimized extractor for timm VisionTransformer-like models.

    Set contiguous_nchw=False to avoid materializing the NCHW tensor when
    the downstream consumer can accept a non-contiguous tensor.
    """

    def __init__(
        self,
        model: nn.Module,
        contiguous_nchw: bool = False,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.model = model
        self.contiguous_nchw = contiguous_nchw
        # Resolve the spatial grid once here so forward() has no per-call Python
        # work / print() inside the compiled (torch.compile) graph.
        self.grid_h, self.grid_w = self.model.patch_embed.dynamic_feat_size(input_size)
        print(f"Model prefix tokens: {self.model.num_prefix_tokens}; grid: {self.grid_h} x {self.grid_w}")

    def forward(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, prefix_tokens + grid_h * grid_w, embedding_dim]
        # This already includes the model's final token normalization.
        final_tokens = self.model.forward_features(batch)

        # Global embedding before the classification layer.
        global_emb = self.model.forward_head(
            final_tokens,
            pre_logits=True,
        )

        # Remove CLS and register tokens.
        spatial_tokens = final_tokens[:, self.model.num_prefix_tokens:]

        # [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        patch_emb = spatial_tokens.reshape(
            batch.shape[0],
            self.grid_h,
            self.grid_w,
            spatial_tokens.shape[-1],
        ).permute(0, 3, 1, 2)

        # This is a potentially expensive full-tensor copy.
        # Only request it if the downstream consumer requires contiguous NCHW.
        if self.contiguous_nchw:
            patch_emb = patch_emb.contiguous()

        return global_emb, patch_emb