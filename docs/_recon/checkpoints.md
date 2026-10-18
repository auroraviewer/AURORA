# Checkpoint Reconnaissance — AURORA LUAD (low-res + high-res)

**Method:** pure-stdlib Python (no `torch` installed, none used). Script:
`docs/_recon/scripts/ckpt_recon/recon.py`; machine-readable dump: `docs/_recon/scripts/ckpt_recon/recon.json`.
Targets were opened **read-only**; nothing was modified or moved.

| Artifact | Size on disk |
|---|---|
| `trained_model/AURORA_LUAD_V2.pth` | 13,854,757 B |
| `trained_model/AURORA_highres_model_weights.pth` | 20,146,232 B |

---

## 1. ZIP container layout

Both files are **new-style (`zipfile`) `torch.save` archives**, format `version = '3\n'`, `byteorder = 'little'`.
No legacy `pickle`-only container, no `_use_new_zipfile_serialization=False` fallback, **no compression** (every entry stored, method 0).

A useful fingerprint: the internal archive prefix is the *original* filename at save time, so
`AURORA_LUAD_V2.pth` is a **renamed copy of `AURORA_model_weights_smooth.pth`**
(`code/train.py:362`), not a file that was ever written under its current name.

### `AURORA_LUAD_V2.pth` (prefix `AURORA_model_weights_smooth/`)

| Entry | Uncompressed bytes |
|---|---|
| `data.pkl` | 2,280 |
| `byteorder` | 6 |
| `version` | 2 |
| `.data/serialization_id` | 40 |
| `data/0` | 3,074,048 |
| `data/1` | 1,024 |
| `data/2` | 7,348,224 |
| `data/3` | 131,072 |
| `data/4` | 512 |
| `data/5` | 131,072 |
| `data/6` | 1,024 |
| `data/7` | 3,074,048 |
| `data/8` | 12,008 |
| `data/9` | 65,536 |
| `data/10` | 256 |
| `data/11` | 8,192 |
| `data/12` | 128 |
| `data/13` | 1,280 |
| `data/14` | 40 |
| **total** | **13,850,792** |

### `AURORA_highres_model_weights.pth` (prefix `AURORA_highres_model_weights/`)

Identical entry list and identical sizes **except `data/2`**:

| Entry | Uncompressed bytes |
|---|---|
| `data.pkl` | 2,280 |
| `byteorder` / `version` / `.data/serialization_id` | 6 / 2 / 40 |
| `data/0` | 3,074,048 |
| `data/1` | 1,024 |
| **`data/2`** | **13,639,680** |
| `data/3` … `data/14` | identical to low-res (131,072 / 512 / 131,072 / 1,024 / 3,074,048 / 12,008 / 65,536 / 256 / 8,192 / 128 / 1,280 / 40) |
| **total** | **20,142,248** |

`data.pkl` is **byte-length identical (2,280) in both files**, and the opcode census is identical
(`REDUCE 38`, `BINPERSID 18`, `BINUNICODE 73`, `SETITEMS 2`, `SETITEM 20`, `BUILD 1`, `STOP 1`) — the two
pickles encode the same structure and differ only in integer literals (shapes + storage sizes).

---

## 2. Reconstruction method (what was done)

Two independent passes over `data.pkl`, because the archive uses the standard torch idiom of
`torch._utils._rebuild_tensor_v2` + **out-of-band storage bytes referenced by persistent id**:

**Pass A — opcode census via `pickletools.genops`.** Enumerated every opcode and every
`GLOBAL`/`STACK_GLOBAL` pair without resolving any class. Result for **both** files — exactly three globals:

```
1  torch._utils  _rebuild_tensor_v2
1  torch          FloatStorage
1  collections    OrderedDict
```

**Pass B — real object-graph reconstruction via `pickle.Unpickler` subclass.** `find_class` was
overridden to return:
* genuine classes for a safelist — `collections.OrderedDict`, `builtins.*`, and `torch.Size` → `tuple`
  (so the top level and all nested dicts come back as real dicts with real `str` keys);
* a **dynamically generated stub class** for everything else (`torch._utils._rebuild_tensor_v2`,
  `torch.storage.*`, dtypes, numpy). Each stub records its qualified name plus the positional/keyword
  arguments of the call, and implements `__setstate__`/`__new__` so the `BUILD`, `NEWOBJ` and
  `NEWOBJ_EX` opcodes cannot break the walk. Stub classes are cached per qualified name.

`persistent_load(pid)` received the out-of-band storage descriptors (torch emits
`('storage', StorageClass, key, location, numel)`) and returned a `StorageRef` recording class,
key, device, numel. This is the **only** way to recover storage sizes without torch, since the tensor
bytes are never inlined in the pickle.

Rebuild signatures then interpreted positionally:

```
_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)
_rebuild_parameter(data, requires_grad, backward_hooks)          # not present in these files
```

Every value in both files uses `_rebuild_tensor_v2`. **No `_rebuild_parameter`, no `_rebuild_from_type_v2`,
no `_rebuild_wrapper_subclass`, no numpy** appears — these are plain tensors, not `nn.Parameter`-tagged,
even though the keys are parameter names.

### Validation that the reconstruction is correct

| Check | Low-res | High-res |
|---|---|---|
| Σ(zip `data/N` bytes) | 13,848,464 | 20,139,920 |
| Σ(tensor numel × itemsize) | 13,848,464 | 20,139,920 |
| Σ(unique storage numel × itemsize) | 13,848,464 | 20,139,920 |
| disk = zip payload + 3,965 B container overhead | 13,854,757 ✓ | 20,146,232 ✓ |

The three sums agree **exactly**, which independently confirms every recovered shape, dtype and storage numel.

---

## 3. Key tables

### 3.1 `AURORA_LUAD_V2.pth` — 18 keys, 3,462,116 params, 13,848,464 B

| # | key | type | shape | dtype | numel | bytes |
|---|---|---|---|---|---|---|
| 0 | `bulk_encoder.0.weight` | `_rebuild_tensor_v2` | `[256, 3002]` | `torch.float32` | 768,512 | 3,074,048 |
| 1 | `bulk_encoder.0.bias` | `_rebuild_tensor_v2` | `[256]` | `torch.float32` | 256 | 1,024 |
| 2 | `lstm.weight_ih_l0` | `_rebuild_tensor_v2` | **`[1024, 1536]`** | `torch.float32` | 1,572,864 | 6,291,456 |
| 3 | `lstm.weight_hh_l0` | `_rebuild_tensor_v2` | `[1024, 256]` | `torch.float32` | 262,144 | 1,048,576 |
| 4 | `lstm.bias_ih_l0` | `_rebuild_tensor_v2` | `[1024]` | `torch.float32` | 1,024 | 4,096 |
| 5 | `lstm.bias_hh_l0` | `_rebuild_tensor_v2` | `[1024]` | `torch.float32` | 1,024 | 4,096 |
| 6 | `decoder.0.weight` | `_rebuild_tensor_v2` | `[128, 256]` | `torch.float32` | 32,768 | 131,072 |
| 7 | `decoder.0.bias` | `_rebuild_tensor_v2` | `[128]` | `torch.float32` | 128 | 512 |
| 8 | `decoder.2.weight` | `_rebuild_tensor_v2` | `[256, 128]` | `torch.float32` | 32,768 | 131,072 |
| 9 | `decoder.2.bias` | `_rebuild_tensor_v2` | `[256]` | `torch.float32` | 256 | 1,024 |
| 10 | `decoder.4.weight` | `_rebuild_tensor_v2` | `[3002, 256]` | `torch.float32` | 768,512 | 3,074,048 |
| 11 | `decoder.4.bias` | `_rebuild_tensor_v2` | `[3002]` | `torch.float32` | 3,002 | 12,008 |
| 12 | `celltype_decoder.0.weight` | `_rebuild_tensor_v2` | `[64, 256]` | `torch.float32` | 16,384 | 65,536 |
| 13 | `celltype_decoder.0.bias` | `_rebuild_tensor_v2` | `[64]` | `torch.float32` | 64 | 256 |
| 14 | `celltype_decoder.2.weight` | `_rebuild_tensor_v2` | `[32, 64]` | `torch.float32` | 2,048 | 8,192 |
| 15 | `celltype_decoder.2.bias` | `_rebuild_tensor_v2` | `[32]` | `torch.float32` | 32 | 128 |
| 16 | `celltype_decoder.4.weight` | `_rebuild_tensor_v2` | `[10, 32]` | `torch.float32` | 320 | 1,280 |
| 17 | `celltype_decoder.4.bias` | `_rebuild_tensor_v2` | `[10]` | `torch.float32` | 10 | 40 |

### 3.2 `AURORA_highres_model_weights.pth` — 18 keys, 5,034,980 params, 20,139,920 B

**Byte-for-byte identical to §3.1 in every row except row 2:**

| # | key | type | shape | dtype | numel | bytes |
|---|---|---|---|---|---|---|
| 2 | `lstm.weight_ih_l0` | `_rebuild_tensor_v2` | **`[1024, 3072]`** | `torch.float32` | 3,145,728 | 12,582,912 |

All other 17 keys are identical in name, shape, dtype and byte count. `lstm.weight_ih_l0` is the
**only** shape that differs between the two checkpoints.

### 3.3 Storage → key map (both files; storage ids and counts identical)

`nn.LSTM` packs its four `_flat_weights` into **one** contiguous storage, which is why 18 tensors
occupy only 15 zip `data/N` entries. Nothing is oversized or dead — the three sums in §2 agree exactly.

| storage | numel (low-res) | numel (high-res) | keys sharing it |
|---|---|---|---|
| `data/0` | 768,512 | 768,512 | `bulk_encoder.0.weight` |
| `data/1` | 256 | 256 | `bulk_encoder.0.bias` |
| **`data/2`** | **1,837,056** | **3,409,920** | **`lstm.weight_ih_l0`, `lstm.weight_hh_l0`, `lstm.bias_ih_l0`, `lstm.bias_hh_l0`** |
| `data/3` | 32,768 | 32,768 | `decoder.0.weight` |
| `data/4` | 128 | 128 | `decoder.0.bias` |
| `data/5` | 32,768 | 32,768 | `decoder.2.weight` |
| `data/6` | 256 | 256 | `decoder.2.bias` |
| `data/7` | 768,512 | 768,512 | `decoder.4.weight` |
| `data/8` | 3,002 | 3,002 | `decoder.4.bias` |
| `data/9` | 16,384 | 16,384 | `celltype_decoder.0.weight` |
| `data/10` | 64 | 64 | `celltype_decoder.0.bias` |
| `data/11` | 2,048 | 2,048 | `celltype_decoder.2.weight` |
| `data/12` | 32 | 32 | `celltype_decoder.2.bias` |
| `data/13` | 320 | 320 | `celltype_decoder.4.weight` |
| `data/14` | 10 | 10 | `celltype_decoder.4.bias` |

Check: `data/2` low-res `1,837,056 = 1,572,864 + 262,144 + 1,024 + 1,024` ✓ ;
high-res `3,409,920 = 3,145,728 + 262,144 + 1,024 + 1,024` ✓.

The recovered `storage_offset` values prove the packing rather than merely suggesting it. The **only**
non-zero offsets in either file are the four LSTM entries, and they tile `data/2` exactly with no gap
and no overlap:

| entry | low-res offset | high-res offset |
|---|---|---|
| `lstm.weight_ih_l0` | 0 | 0 |
| `lstm.weight_hh_l0` | 1,572,864 | 3,145,728 |
| `lstm.bias_ih_l0` | 1,835,008 | 3,407,872 |
| `lstm.bias_hh_l0` | 1,836,032 | 3,408,896 |
| end of last | 1,837,056 = numel ✓ | 3,409,920 = numel ✓ |

All 14 other tensors sit at offset 0 in their own private storage, and all recovered `stride` values
are contiguous (`[n, 1]` / `[1]`), so every tensor is a dense, non-overlapping, C-contiguous view —
no slicing, transposition or aliasing tricks to worry about. Recovered key order equals pickled order
in both files (`OrderedDict`).

---

## 4. Answers to the specific questions

### 4.1 Do the two state_dicts share key names? Are the key sets disjoint?

**They are not disjoint — they are *identical*.** Intersection = **18**, low-res-only = **0**,
high-res-only = **0**. The two checkpoints have exactly the same 18 fully-qualified parameter names
in exactly the same order. They are the same architecture at two different input widths.

### 4.2 Top level: bare state_dict, or a dict with metadata?

**Bare state_dict.** In both files the top-level unpickled object is a
`collections.OrderedDict` whose 18 keys are *directly* the parameter names — there is **no**
`state_dict`, `model`, `epoch`, `optimizer`, `args`, `lr_scheduler` or `best_*` wrapper key, and no
non-tensor metadata of any kind. This matches the writers:
`code/train.py:362` and `code/train_highres.py:387` both call `torch.save(state_dict, …)`.

Because the container is `OrderedDict`, key order is preserved and matches the `nn.Module`
registration order.

### 4.3 Exact key set for a `strict=True` load

Both files, verbatim and in pickled order (18 keys):

```
bulk_encoder.0.weight
bulk_encoder.0.bias
lstm.weight_ih_l0
lstm.weight_hh_l0
lstm.bias_ih_l0
lstm.bias_hh_l0
decoder.0.weight
decoder.0.bias
decoder.2.weight
decoder.2.bias
decoder.4.weight
decoder.4.bias
celltype_decoder.0.weight
celltype_decoder.0.bias
celltype_decoder.2.weight
celltype_decoder.2.bias
celltype_decoder.4.weight
celltype_decoder.4.bias
```

This is exactly the parameter set of `AURORA` (`code/model.py:4`) built with
`finetune=False` (→ `self.decoder` exists, `self.new_decoder` does not),
`bidirectional=False` (→ single-direction LSTM, `_l0` only, no `_reverse`),
`use_bulk=True` (→ `bulk_encoder` present) and `pred_celltype=True` (→ `celltype_decoder` present).
`strict=True` will succeed for both files with the constructor calls below.

### 4.4 Constructor calls that reproduce each state_dict (verified against `code/model.py`)

Derived from `code/train.py:53-54,109-112` and `code/train_highres.py:46,122-123`:

```python
# AURORA_LUAD_V2.pth        (low-resolution)
AURORA(input_size=1536, output_size=3002, device=..., celltype_num=10,
       bulk_size=3002, latent_size=256, bidirectional=False,
       use_bulk=True, pred_celltype=True, finetune=False)

# AURORA_highres_model_weights.pth   (high-resolution)
AURORA(input_size=3072, output_size=3002, device=..., celltype_num=10,
       bulk_size=3002, latent_size=256, bidirectional=False,
       use_bulk=True, pred_celltype=True, finetune=False)
```

Every shape follows from `code/model.py`:

| module | definition | derived shape | observed |
|---|---|---|---|
| `bulk_encoder.0` | `nn.Linear(bulk_size, latent_size)` | `[256, 3002]` | ✓ |
| `lstm` | `nn.LSTM(input_size, 256, bidirectional=False)` | `weight_ih_l0 [4·256, input_size]`, `weight_hh_l0 [1024, 256]` | ✓ 1536 / 3072 |
| `decoder.0` | `nn.Linear(256, 256//2)` | `[128, 256]` | ✓ |
| `decoder.2` | `nn.Linear(128, 256)` | `[256, 128]` | ✓ |
| `decoder.4` | `nn.Linear(256, output_size)` | `[3002, 256]` | ✓ |
| `celltype_decoder.0` | `nn.Linear(256, max(256//4, 32))` | `[64, 256]` | ✓ |
| `celltype_decoder.2` | `nn.Linear(64, 32)` | `[32, 64]` | ✓ |
| `celltype_decoder.4` | `nn.Linear(32, major_num)` | `[10, 32]` | ✓ |

---

## 5. Companion config and CSV files

### 5.1 `trained_model/model_parameter_LUAD.json` (717 B)

Keys: `use_bulk=true`, `pred_celltype=true`, `bidirectional=false`, `retrain_cellprop_only=false`,
`use_context_loss=true`, **`img_emb_dim=1536`**, `pixel_size=0.5`, `num_workers=4`, `batch_size=4`,
`patch_size=[3584, 896, 224]`, `mse_weight=10`, `prop_weight=0.1`, `focal_gamma=0.5`,
`num_epochs=600`, `exp_epochs=400`, `lr=0.001`, `lr_prop=0.001`, `weight_decay=1e-05`,
`stepLR_step_exp=200`, `stepLR_gamma_exp=0.1`, `stepLR_step_prop=50`, `stepLR_gamma_prop=0.5`,
`gene_list_file="LUAD_gene.csv"`, `gene_norm_factor_file="LUAD_gene_normalize_factor.csv"`,
`celltype_file="LUAD_cell_types.csv"`.

`img_emb_dim = 1536` matches `lstm.weight_ih_l0` → `[1024, 1536]`. ✓

### 5.2 `trained_model/model_parameter_LUAD_highres.json` (840 B)

Same 22 shared keys with these differences:
**`img_emb_dim=3072`**, `batch_size=8`, `patch_size=[56, 14]`, `num_epochs=200`, `exp_epochs=100`,
`stepLR_step_exp=50`, plus four extra keys not present in the low-res config:
`superpatch_size=3000`, `superpatch_stride=2000`, `superpatch_minsize=3000`,
`superpatch_minlowres_patch=50`. The three file references are prefixed with a leading slash
(`"/LUAD_gene.csv"` etc.).

`img_emb_dim = 3072` matches `lstm.weight_ih_l0` → `[1024, 3072]`. ✓

> **This is the only architectural difference between the two checkpoints, and it is fully explained
> by `img_emb_dim`.** Both configs request `bidirectional=false`, matching the absence of any
> `_reverse` LSTM keys.

### 5.3 CSV inventories (row counts and headers only)

| file | size | data rows | columns / header |
|---|---|---|---|
| `trained_model/LUAD_gene.csv` | 32,574 B | **3,002** | 2 cols; header `["", "Genes"]` (unnamed index column + `Genes`) |
| `trained_model/LUAD_cell_types.csv` | 92 B | **10** | 1 col, **no header row** — values are the class names |
| `trained_model/LUAD_gene_normalize_factor.csv` | 234,342 B | **17,860** | 2 cols; header `["gene_name", "0"]` |

`LUAD_cell_types.csv` in full, in file order (10 labels, no header):

```
B_cell, Dendritic, Endothelial, Epithelial, Fibroblast,
Macrophage, Malignant, Mast, NK_cell, T_cell
```

**Cross-checks — three independent confirmations of the constructor arguments:**

* `output_size = len(gene_list) = 3002` (`code/train.py:54`) → `decoder.4.weight [3002, 256]` ✓
* `bulk_size = len(hvg_list) = len(gene_list) = 3002` (`code/train.py:51,109`) → `bulk_encoder.0.weight [256, 3002]` ✓
* `celltype_num = len(major_cell_type) = 10` (`code/train.py:110`) → `celltype_decoder.4.weight [10, 32]` ✓

The cell-types file is read with `header=None` (`code/train.py:88`), so the **first line `B_cell` is a
class label, not a column header** — there are 10 classes, not 9. Reading it with a default
`pd.read_csv` (header inferred) silently yields 9 classes and would produce a `celltype_decoder.4`
shape mismatch. See §6.

Note `LUAD_gene_normalize_factor.csv` lists **17,860** genes while the model predicts **3,002**;
`code/train.py:52` selects the model's rows via `.loc[gene_list, '0']`, so the panel is a superset.

---

## 6. Surprises / gotchas for a loader

1. **Saved from CUDA.** Every persistent id carries location `cuda:0`. `torch.load(path)` on a
   CPU-only machine will raise unless you pass `map_location`. Use
   `torch.load(path, map_location="cpu", weights_only=True)`.
2. **`AURORA_LUAD_V2.pth` is a renamed `AURORA_model_weights_smooth.pth`.** The internal ZIP prefix
   exposes the original save name, which no longer matches the file on disk. Do not key any logic off
   the filename.
3. **`LUAD_cell_types.csv` has no header.** It must be read with `header=None` (as `train.py` does) or
   you lose one of the 10 classes. Cell-type index order is the file order in §5.3, and the model's
   `celltype_decoder.4` output activations are `LogSoftmax(dim=-1)` (`code/model.py:72`) — so logits
   should be exponentiated before being read as proportions.
4. **The two checkpoints differ by exactly one shape.** `lstm.weight_ih_l0` is `[1024, 1536]` vs
   `[1024, 3072]`. They are *not* interchangeable: 17 of 18 tensors are shape-identical, so a naive
   transfer will fail only on that one key, and `strict=True` will report exactly one size mismatch.
   Feeding 1536-d embeddings into the high-res model (or vice versa) fails at the LSTM input.
5. **All tensors are plain tensors, not `nn.Parameter`.** Both pickles contain only
   `_rebuild_tensor_v2`; there is no `_rebuild_parameter`. So the archive cannot tell you which
   tensors were frozen — every one of the 18 entries was saved with its `requires_grad` flag as a
   plain argument (all recorded as `False`). Whether a parameter was trainable at save time is
   therefore *not* recoverable from these files.
6. **No optimizer / epoch / args state was saved.** Both files are pure weight dicts, so training
   cannot be resumed and no run metadata (best epoch, LR schedule position, loss) is available.
7. **No compression, `version = 3`, `byteorder = little`.** These are host-endian little-endian CUDA
   float32 archives; a big-endian host would need the `byteorder` entry honoured, but any normal
   x86/ARM little-endian box loads them directly.
8. **`bulk_encoder` and `decoder.4` are not weight-tied.** They happen to have identical shapes
   (`[256, 3002]` and `[3002, 256]` — transposes of each other, 768,512 elements each) and live in
   separate storages `data/0` and `data/7`, so they are independent matrices despite the suspicious
   symmetry.

---

## 7. Reproduction

```
C:\Users\14791\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe \
  C:\Edward_Chen\appeal\AURORA-main\docs\_recon\scripts\ckpt_recon\recon.py
```

Requires only the standard library (`zipfile`, `pickle`, `pickletools`, `io`, `json`, `os`, `sys`,
`collections`). Emits the full per-file dump to stdout and a machine-readable
`docs/_recon/scripts/ckpt_recon/recon.json` (key tables, storage map, opcode census, persistent ids).
