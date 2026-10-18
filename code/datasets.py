import numpy as np
import cv2
from torch.utils.data import Dataset, Sampler
from typing import Iterator
from PIL import Image
import torch
from train_utils import SuperpatchCollection
class PatchDataset(Dataset):
    def __init__(self, image, mask, transform, loc, spot_radius_pixel, patch_size=224, stride=224, filter_prop = 0.2, high_res=False):
        self.stride = stride
        if spot_radius_pixel is not None:
            self.spot_radius_pixel = int(spot_radius_pixel)
        else:
            self.spot_radius_pixel = None
        # Padding the image and mask to make sure that all pixels are covered by some superpixels
        self.row_pad = self.stride - (image.shape[0] % self.stride)
        self.col_pad = self.stride - (image.shape[1] % self.stride)
        self.loc = loc
        image = np.pad(image.copy(), ((self.row_pad//2, self.row_pad-self.row_pad//2), (self.col_pad//2, self.col_pad-self.col_pad//2), (0, 0)), mode='constant', constant_values=255)
        mask = np.pad(mask.copy(), ((self.row_pad//2, self.row_pad-self.row_pad//2), (self.col_pad//2, self.col_pad-self.col_pad//2)), mode='constant', constant_values=0)
        # print(f"    Row padding: {self.row_pad}; Column padding: {self.col_pad}; Padded image shape: {image.shape}")
        self.image = image
        self.image[mask == 0] = [255, 255, 255]
        self.mask = mask
        self.patch_size = patch_size
        self.transform = transform
        self.shape_ori = np.array(image.shape[:2])
        self.num_patches = ((self.shape_ori + patch_size -1) // self.stride)
        self.total_patches = self.num_patches[0] * self.num_patches[1]
        self.filter_prop = filter_prop
        self.high_res = high_res
    def __len__(self):
        return self.total_patches
    def __getitem__(self, idx):
        # coordinate of the top left corner of the metapixel
        i = (idx // self.num_patches[1]) * self.stride
        j = (idx % self.num_patches[1]) * self.stride
        start_i, start_j = i, j
        end_i, end_j = min(self.shape_ori[0], i + self.patch_size), min(self.shape_ori[1], j + self.patch_size)
        patch = self.image[start_i:end_i, start_j:end_j, :]
        # Pad if necessary to ensure 224x224 size
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=patch.dtype)
            padded_patch[(self.patch_size-patch.shape[0])//2:(self.patch_size-patch.shape[0])//2+patch.shape[0], 
                         (self.patch_size-patch.shape[1])//2:(self.patch_size-patch.shape[1])//2+patch.shape[1]] = patch
            patch = padded_patch
        patch = Image.fromarray(patch.astype('uint8')).convert('RGB')
        metapixel_mask = self.mask[(i):(i+self.patch_size), (j):(j+self.patch_size)]
        if not self.high_res:
            # NOTE (aurora 2.0 reorganisation): wrap the comparison in bool().
            # ``np.sum(...) > ...`` returns a NumPy scalar (``np.bool_``) on many
            # NumPy versions rather than a Python bool.  A batch of NumPy scalars
            # is dispatched by ``torch.utils.data.default_collate`` to
            # ``collate_numpy_scalar_fn``, which calls ``torch.as_tensor`` on the
            # list and fails with
            #   TypeError: 'numpy.bool' object cannot be interpreted as an integer
            # because ``np.bool_`` has no ``__index__`` (unlike ``np.int64``).
            # Returning a real ``bool`` keeps the saved ``flag`` array a boolean
            # array while collating correctly on every NumPy/PyTorch pair.
            keep_flag = bool(np.sum(metapixel_mask) > self.patch_size*self.patch_size * self.filter_prop)
        else:
            block_size = 14
            keep_flag = np.zeros((16, 16), dtype=bool)
            for r in range(16):
                for c in range(16):
                    r0, r1 = r * block_size, min((r + 1) * block_size, self.patch_size)
                    c0, c1 = c * block_size, min((c + 1) * block_size, self.patch_size)
                    r1 = min(r1, metapixel_mask.shape[0])
                    c1 = min(c1, metapixel_mask.shape[1])
                    sub_mask = metapixel_mask[r0:r1, c0:c1]
                    keep_flag[r, c] = np.sum(sub_mask) > sub_mask.size * self.filter_prop
        return self.transform(patch), (i, j), keep_flag
    def plot_mask(self):
        mask_plot = self.image.copy()
        mask_plot[self.mask == 0] = [0, 255, 0]
        mask_plot = cv2.addWeighted(self.image, 0.7, mask_plot, 0.3, 0)
        mask_plot = cv2.resize(mask_plot, (0,0), fx=0.1, fy=0.1)
        return mask_plot
    def plot_patch_loc(self, plot_spot = True):
        patch_plot = self.image.copy()
        for idx in range(self.total_patches):
            i = (idx // self.num_patches[1]) * self.stride
            j = (idx % self.num_patches[1]) * self.stride
            start_i, start_j = i, j
            end_i, end_j = min(self.shape_ori[0], i + self.patch_size), min(self.shape_ori[1], j + self.patch_size)
            metapixel_mask = self.mask[(i):(i+self.patch_size), (j):(j+self.patch_size)]
            keep_flag = np.sum(metapixel_mask) > self.patch_size*self.patch_size * self.filter_prop
            if keep_flag:
                if (idx // self.num_patches[1] + idx % self.num_patches[1]) % 2 == 0:
                    patch_plot[start_i:end_i, start_j:end_j, :] = [0, 0, 255]
                else:
                    patch_plot[start_i:end_i, start_j:end_j, :] = [0, 255, 255]
            else:
                patch_plot[start_i:end_i, start_j:end_j, :] = [0, 255, 0]
            # Color two consecutive patch to check overlapping
            # if idx == 500:
            #     patch_plot[start_i:end_i, start_j:end_j, :] = [0, 0, 255]
            # elif idx == 501:
            #     patch_plot[start_i:end_i, start_j:end_j, :] = [0, 255, 255]
        patch_plot = cv2.addWeighted(self.image, 0.7, patch_plot, 0.3, 0)
        ## Plot delimiters
        for idx in range(self.total_patches):
            i = (idx // self.num_patches[1]) * self.stride
            j = (idx % self.num_patches[1]) * self.stride
            start_i, start_j = i, j
            end_i, end_j = min(self.shape_ori[0], i + self.patch_size), min(self.shape_ori[1], j + self.patch_size)
            if (idx // self.num_patches[1] + idx % self.num_patches[1]) % 4 == 0:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [0, 0, 0]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [0, 0, 0]
            elif (idx // self.num_patches[1] + idx % self.num_patches[1]) % 4 == 1:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [255,255,255]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [255,255,255]
            elif (idx // self.num_patches[1] + idx % self.num_patches[1]) % 4 == 2:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [255, 0, 0]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [255, 0, 0]
            else:
                patch_plot[start_i:(start_i+self.patch_size//5), start_j:(start_j+self.patch_size//5), :] = [255, 0, 255]
                patch_plot[(end_i-self.patch_size//5):end_i, (end_j-self.patch_size//5):end_j, :] = [255, 0, 255]
        
        if plot_spot is not None and self.spot_radius_pixel is not None and self.loc is not None:
            spot_plot = patch_plot.copy()
            #! Notice that this the first column of loc is the col index for the image matrix, which is the x axis direction on the plot!
            loc = self.loc + [self.col_pad//2, self.row_pad//2]
            for (x, y) in loc:
                spot_plot = cv2.circle(spot_plot, (int(x), int(y)), self.spot_radius_pixel, (255, 0, 0), -1)
            patch_plot = cv2.addWeighted(patch_plot, 0.5, spot_plot, 0.5, 0)
        patch_plot[self.mask == 0] = [255, 255, 255]  # Color the masked area as white
        patch_plot = cv2.resize(patch_plot, (0,0), fx=0.1, fy=0.1)
        return patch_plot


class PrefilteredHighResPatchDataset(PatchDataset):
    """High-resolution patch dataset that indexes only active source patches.

    The parent constructor is intentionally reused without modification.  This
    preserves the original row/column padding, white mask fill, patch count,
    and coordinate system exactly.  The only behavioral change is that token
    validity is computed once on the padded 14-pixel grid and patches with no
    valid tokens are omitted before image transforms are executed.
    """

    def __init__(
        self,
        image,
        mask,
        transform,
        loc,
        spot_radius_pixel,
        patch_size=224,
        stride=224,
        filter_prop=0.2,
        high_res=True,
        *,
        token_size=14,
    ):
        if not high_res:
            raise ValueError("PrefilteredHighResPatchDataset requires high_res=True")
        super().__init__(
            image,
            mask,
            transform,
            loc,
            spot_radius_pixel,
            patch_size=patch_size,
            stride=stride,
            filter_prop=filter_prop,
            high_res=True,
        )
        self.filter_prop = float(self.filter_prop)
        if not (0.0 < self.filter_prop <= 1.0):
            raise ValueError(
                f"filter_prop must be in (0, 1]; got {self.filter_prop}"
            )
        self.token_size = int(token_size)
        if self.token_size <= 0:
            raise ValueError("token_size must be positive")
        if self.patch_size % self.token_size != 0:
            raise ValueError(
                f"patch_size={self.patch_size} must be divisible by "
                f"token_size={self.token_size}"
            )
        if self.stride % self.token_size != 0:
            raise ValueError(
                f"stride={self.stride} must be divisible by token_size={self.token_size}"
            )
        if np.any(self.shape_ori % self.token_size != 0):
            raise ValueError(
                "the padded image shape must be divisible by token_size; got "
                f"{tuple(self.shape_ori.tolist())} and token_size={self.token_size}"
            )

        self.token_grid_size = self.patch_size // self.token_size
        self.stride_tokens = self.stride // self.token_size
        self.fine_grid_shape = tuple((self.shape_ori // self.token_size).tolist())
        grid_rows, grid_cols = self.fine_grid_shape

        # Patch origins and token cells are aligned because stride and patch
        # size are integer multiples of token_size.  Computing validity once on
        # this absolute grid is therefore exactly equivalent to the nested
        # 16x16 slicing loop in PatchDataset.__getitem__.
        mask_blocks = self.mask.reshape(
            grid_rows,
            self.token_size,
            grid_cols,
            self.token_size,
        )
        token_sums = mask_blocks.sum(axis=(1, 3))
        self.fine_valid = np.asarray(
            token_sums > (self.token_size * self.token_size * self.filter_prop),
            dtype=bool,
        )

        source_indices = np.arange(int(self.total_patches), dtype=np.int64)
        patch_cols = int(self.num_patches[1])
        source_patch_rows = source_indices // patch_cols
        source_patch_cols = source_indices % patch_cols
        origin_rows = source_patch_rows * self.stride_tokens
        origin_cols = source_patch_cols * self.stride_tokens

        # Integral-image window sums identify active patches without loading or
        # transforming any image pixels.  Origins at or beyond the padded edge
        # naturally receive a zero-sized/zero-sum window, matching the parent.
        integral = np.pad(
            self.fine_valid.astype(np.int32).cumsum(0).cumsum(1),
            ((1, 0), (1, 0)),
            mode="constant",
        )
        row_ends = np.minimum(origin_rows + self.token_grid_size, grid_rows)
        col_ends = np.minimum(origin_cols + self.token_grid_size, grid_cols)
        in_bounds = (origin_rows < grid_rows) & (origin_cols < grid_cols)
        window_sums = np.zeros(source_indices.shape[0], dtype=np.int32)
        valid_origins = np.flatnonzero(in_bounds)
        if valid_origins.size:
            r0 = origin_rows[valid_origins]
            c0 = origin_cols[valid_origins]
            r1 = row_ends[valid_origins]
            c1 = col_ends[valid_origins]
            window_sums[valid_origins] = (
                integral[r1, c1]
                - integral[r0, c1]
                - integral[r1, c0]
                + integral[r0, c0]
            )
        active = window_sums > 0
        self.source_total_patches = int(self.total_patches)
        self.source_indices = source_indices[active]
        self.active_source_indices = self.source_indices
        self.active_patch_count = int(self.source_indices.shape[0])
        self.skipped_patch_count = self.source_total_patches - self.active_patch_count
        self.active_positions = np.column_stack(
            [origin_rows[active], origin_cols[active]]
        ).astype(np.int32, copy=False) * self.token_size

        # The number of source-patch contributions at a valid absolute token
        # cell is separable on this regular lattice.  Every source patch that
        # covers a valid cell is necessarily active, so this is also the exact
        # denominator used by feature aggregation.
        row_coverage = self._axis_coverage(
            grid_rows,
            int(self.num_patches[0]),
            self.stride_tokens,
            self.token_grid_size,
        )
        col_coverage = self._axis_coverage(
            grid_cols,
            int(self.num_patches[1]),
            self.stride_tokens,
            self.token_grid_size,
        )
        coverage = row_coverage[:, None] * col_coverage[None, :]
        self.fine_contribution_counts = (
            coverage * self.fine_valid.astype(np.int32)
        ).astype(np.int16, copy=False)

    @staticmethod
    def _axis_coverage(
        grid_size: int,
        num_origins: int,
        stride_tokens: int,
        patch_tokens: int,
    ) -> np.ndarray:
        difference = np.zeros(grid_size + 1, dtype=np.int32)
        for origin_index in range(num_origins):
            start = origin_index * stride_tokens
            if start >= grid_size:
                continue
            end = min(start + patch_tokens, grid_size)
            difference[start] += 1
            difference[end] -= 1
        return np.cumsum(difference[:-1], dtype=np.int32)

    def __len__(self):
        return int(self.source_indices.shape[0])

    def _keep_flag_at_position(self, row: int, col: int) -> np.ndarray:
        token_row = row // self.token_size
        token_col = col // self.token_size
        keep_flag = np.zeros(
            (self.token_grid_size, self.token_grid_size),
            dtype=bool,
        )
        row_end = min(token_row + self.token_grid_size, self.fine_grid_shape[0])
        col_end = min(token_col + self.token_grid_size, self.fine_grid_shape[1])
        if row_end > token_row and col_end > token_col:
            keep_flag[: row_end - token_row, : col_end - token_col] = self.fine_valid[
                token_row:row_end,
                token_col:col_end,
            ]
        return keep_flag

    def __getitem__(self, idx):
        source_idx = int(self.source_indices[idx])
        i = (source_idx // int(self.num_patches[1])) * self.stride
        j = (source_idx % int(self.num_patches[1])) * self.stride
        end_i = min(int(self.shape_ori[0]), i + self.patch_size)
        end_j = min(int(self.shape_ori[1]), j + self.patch_size)
        patch = self.image[i:end_i, j:end_j, :]

        # Keep the parent's centered zero-padding logic byte-for-byte.
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros(
                (self.patch_size, self.patch_size, 3),
                dtype=patch.dtype,
            )
            row_offset = (self.patch_size - patch.shape[0]) // 2
            col_offset = (self.patch_size - patch.shape[1]) // 2
            padded_patch[
                row_offset : row_offset + patch.shape[0],
                col_offset : col_offset + patch.shape[1],
            ] = patch
            patch = padded_patch

        patch = Image.fromarray(patch.astype("uint8")).convert("RGB")
        keep_flag = self._keep_flag_at_position(i, j)
        return self.transform(patch), (i, j), keep_flag

    def next_origin_token_row(self, processed_count: int) -> int:
        """Fine-grid row of the next unprocessed active patch origin."""
        processed_count = int(processed_count)
        if processed_count < 0 or processed_count > len(self):
            raise IndexError(
                f"processed_count must be in [0, {len(self)}], got {processed_count}"
            )
        if processed_count == len(self):
            return int(self.fine_grid_shape[0])
        return int(self.active_positions[processed_count, 0] // self.token_size)


class RegularGridPatchDataset(PrefilteredHighResPatchDataset):
    """Compatibility name for the prefiltered regular-grid dataset."""



class RowSpanBatchSampler(Sampler[list[int]]):
    """Consecutive batches whose source-patch origins stay within a row span.

    Bounding a batch's source-row span keeps the number of simultaneously live
    GPU accumulation strips bounded on sparse images while still producing full
    fixed-shape batches wherever the row-major active patch sequence permits.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        max_row_span_tokens: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_row_span_tokens <= 0:
            raise ValueError("max_row_span_tokens must be positive")
        if not hasattr(dataset, "active_positions") or not hasattr(
            dataset, "token_size"
        ):
            raise TypeError(
                "dataset must provide active_positions and token_size attributes"
            )

        positions = np.asarray(dataset.active_positions)
        if positions.shape != (len(dataset), 2):
            raise ValueError(
                "dataset.active_positions must have shape "
                f"({len(dataset)}, 2), got {positions.shape}"
            )
        token_size = int(dataset.token_size)
        if token_size <= 0:
            raise ValueError("dataset.token_size must be positive")
        row_tokens = positions[:, 0].astype(np.int64, copy=False) // token_size
        if row_tokens.size > 1 and np.any(np.diff(row_tokens) < 0):
            raise ValueError("dataset.active_positions must be row-major ordered")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.max_row_span_tokens = int(max_row_span_tokens)
        self._bounds: list[tuple[int, int]] = []

        start = 0
        total = len(dataset)
        while start < total:
            end = start + 1
            first_row = int(row_tokens[start])
            while end < total and end - start < self.batch_size:
                if int(row_tokens[end]) - first_row >= self.max_row_span_tokens:
                    break
                end += 1
            self._bounds.append((start, end))
            start = end

    def __iter__(self) -> Iterator[list[int]]:
        for start, end in self._bounds:
            yield list(range(start, end))

    def __len__(self) -> int:
        return len(self._bounds)


# Modify the Dataset class to output data on the correct device
class HistologyGeneExpressionDataset(Dataset):
    def __init__(self, data, device, highres=False, adj_mat=False):
        self.data = []
        self.highres = highres
        self.adj_mat = adj_mat
        if self.highres or self.adj_mat:
            self.device = 'cpu' # Always store data on CPU for highres mode to save GPU memory, and move to GPU during training/testing
        else:
            self.device = device
        if isinstance(data, SuperpatchCollection):
            # Lazy mode: superpatches are loaded from disk on demand (with a
            # small LRU cache inside the collection), so the whole training set
            # is never held in RAM at once.
            self.lazy_source = data
        else:
            self.lazy_source = None
            if not self.adj_mat:
                for embs, windows_mask, spot_weight, gene_expressions, gene_mask, bulk_filtered, bulk_rank, max_exp, father_patches, major_prop in data:
                    self.data.append((embs,windows_mask,spot_weight, gene_expressions, gene_mask, bulk_filtered, bulk_rank, max_exp, father_patches, major_prop))
            else:
                for embs, windows_mask, spot_weight, gene_expressions, gene_mask, bulk_filtered, bulk_rank, max_exp, father_patches, major_prop, adj_mat in data:
                    self.data.append((embs,windows_mask,spot_weight, gene_expressions, gene_mask, bulk_filtered, bulk_rank, max_exp, father_patches, major_prop, adj_mat))
    #
    def __len__(self):
        if self.lazy_source is not None:
            return len(self.lazy_source)
        return len(self.data)
    #
    def __getitem__(self, idx):
        dtype_switch = torch.float32 if not self.highres else torch.int32
        if self.lazy_source is not None:
            item = self.lazy_source[idx]
        else:
            item = self.data[idx]
        if not self.adj_mat:
            emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop = item
        else:
            emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop, adj_mat = item
        # Shallow-copy the per-scale dicts so we never mutate the stored/cached
        # tuples (important for the lazy path where items are shared/cached).
        emb = dict(emb)
        windows_mask = dict(windows_mask)
        spot_weight = dict(spot_weight)
        father_patches = dict(father_patches)
        for key in emb.keys():
            emb[key] = torch.tensor(emb[key], dtype=torch.float32).to(self.device)
        for key in windows_mask.keys():
            windows_mask[key] = torch.tensor(windows_mask[key], dtype=torch.float32).to(self.device)
        for key in spot_weight.keys():
            if isinstance(spot_weight[key], dict):
                # Compact sparse form {'indices', 'values'}.
                spot_weight[key] = {
                    'indices': torch.tensor(spot_weight[key]['indices'], dtype=torch.long).to(self.device),
                    'values': torch.tensor(spot_weight[key]['values'], dtype=torch.float32).to(self.device),
                }
            else:
                spot_weight[key] = torch.tensor(spot_weight[key], dtype=torch.float32).to(self.device)
        for key in father_patches.keys():
            father_patches[key] = torch.tensor(father_patches[key], dtype=torch.int).to(self.device)
        if self.adj_mat:
            # check if adj_mat is a dictionary, if so, convert each value to tensor
            if isinstance(adj_mat, dict):
                for key in adj_mat.keys():
                    adj_mat[key] = torch.tensor(adj_mat[key], dtype=torch.int32).to(self.device)
            else:
                adj_mat = torch.tensor(adj_mat, dtype=torch.int32).to(self.device)
        if not self.adj_mat:
            return emb, windows_mask, spot_weight, torch.tensor(
                gene_expression, dtype=torch.float32
            ).to(self.device), torch.tensor(gene_mask).to(self.device), torch.tensor(
                bulk, dtype=torch.float32
            ).to(self.device), torch.tensor(
                bulk_rank, dtype=torch.float32
            ).to(self.device), torch.tensor(
                max_exp, dtype=dtype_switch
            ).to(self.device), father_patches, torch.tensor(
                major_prop, dtype=torch.float32
            ).to(self.device)
        else:
            return emb, windows_mask, spot_weight, torch.tensor(
                gene_expression, dtype=torch.float32
            ).to(self.device), torch.tensor(gene_mask).to(self.device), torch.tensor(
                bulk, dtype=torch.float32
            ).to(self.device), torch.tensor(
                bulk_rank, dtype=torch.float32
            ).to(self.device), torch.tensor(
                max_exp, dtype=dtype_switch
            ).to(self.device), father_patches, torch.tensor(
                major_prop, dtype=torch.float32
            ).to(self.device), adj_mat
        
    
def my_collate(batch):
    emb = [item[0] for item in batch]
    windows_mask = [item[1] for item in batch]
    spot_weight = [item[2] for item in batch]
    gene_expression = [item[3] for item in batch]
    gene_mask = [item[4] for item in batch]
    bulk = [item[5] for item in batch]
    bulk_rank = [item[6] for item in batch]
    max_exp = [item[7] for item in batch]
    father_patches = [item[8] for item in batch]
    major_prop = [item[9] for item in batch]
    return emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop

def my_collate_pred(batch):
    emb = [item[0] for item in batch]
    bulk = [item[1] for item in batch]
    bulk_rank = [item[2] for item in batch]
    father_patches = [item[3] for item in batch]
    return emb, bulk, bulk_rank, father_patches

def my_collate_adjmat(batch):
    emb = [item[0] for item in batch]
    windows_mask = [item[1] for item in batch]
    spot_weight = [item[2] for item in batch]
    gene_expression = [item[3] for item in batch]
    gene_mask = [item[4] for item in batch]
    bulk = [item[5] for item in batch]
    bulk_rank = [item[6] for item in batch]
    max_exp = [item[7] for item in batch]
    father_patches = [item[8] for item in batch]
    major_prop = [item[9] for item in batch]
    adj_mat = [item[10] for item in batch]
    return emb, windows_mask, spot_weight, gene_expression, gene_mask, bulk, bulk_rank, max_exp, father_patches, major_prop, adj_mat
