#!/usr/bin/env python
"""Regression test for the DataLoader collate failure in ``code/datasets.py``.

The low-resolution branch of ``PatchDataset.__getitem__`` used to return
``np.sum(mask) > threshold``, which is a NumPy scalar (``np.bool_``) on many
NumPy versions.  ``torch.utils.data.default_collate`` routes a batch of NumPy
scalars to its numpy-scalar handler, which calls ``torch.as_tensor`` on the list
and raises::

    TypeError: 'numpy.bool' object cannot be interpreted as an integer

because ``np.bool_`` implements ``__int__`` but not ``__index__`` -- unlike
``np.int64``, which is why the neighbouring ``positions`` tuple never failed.
The dataset now returns a real ``bool``.

The test has two parts:

* **mechanism** -- no PyTorch needed.  Reproduces the exact CPython error from a
  ``np.bool_`` and shows a Python ``bool`` is fine, so the diagnosis is verified
  even in an environment without torch.
* **dataset** -- builds the smallest possible ``PatchDataset`` and collates a
  batch the way the DataLoader does.  Skipped when torch is absent.

    python tests/test_dataloader_collate.py
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "code"))

#: Returned when the environment cannot run any part of the test.
SKIP = 2

failures: list[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def _transform(_image):  # module level so DataLoader workers can pickle it
    import torch

    return torch.zeros(3, 8, 8)


def check_mechanism() -> None:
    """Show precisely why a NumPy boolean scalar cannot be collated."""
    import numpy as np

    print("mechanism (no torch needed)")
    numeric = np.int64(224)
    boolean = np.bool_(True)
    print(f"  numeric numpy scalar     : {type(numeric).__module__}.{type(numeric).__name__}")
    print(f"  boolean numpy scalar     : {type(boolean).__module__}.{type(boolean).__name__}")

    expect(hasattr(numeric, "__index__"), "np.int64 has __index__, so torch can build a tensor")
    expect(not hasattr(boolean, "__index__"), "np.bool_ has no __index__")

    # operator.index is exactly the conversion whose failure produces the message
    # in the reported traceback.
    index_error = None
    try:
        operator.index(boolean)
    except TypeError as exc:
        index_error = str(exc)
    print(f"  operator.index(np.bool_) : {index_error}")
    expect(
        index_error is not None and "cannot be interpreted as an integer" in index_error,
        "np.bool_ raises the error seen in the traceback",
    )
    expect(operator.index(True) == 1, "a Python bool converts cleanly")

    # The comparison that used to leak a NumPy scalar.
    mask = np.ones((8, 8), dtype=np.float32)
    raw = np.sum(mask) > 8 * 8 * 0.2
    wrapped = bool(np.sum(mask) > 8 * 8 * 0.2)
    print(f"  np.sum(mask) > t          : {type(raw).__module__}.{type(raw).__name__}")
    print(f"  bool(np.sum(mask) > t)    : {type(wrapped).__module__}.{type(wrapped).__name__}")
    expect(type(wrapped) is bool, "bool(...) always yields a Python bool")
    expect(
        not hasattr(raw, "__index__") or type(raw) is bool,
        "the raw comparison is either a Python bool or a non-indexable NumPy scalar",
    )
    # Simulate the NumPy versions that still hand back np.bool_.
    forced = np.bool_(np.float32(5.0) > 1.0)
    expect(type(bool(forced)) is bool, "wrapping a genuine np.bool_ also yields a Python bool")
    expect(not hasattr(forced, "__index__"), "and the unwrapped one is not indexable")


def check_dataset() -> str:
    """Collate a real PatchDataset batch, exactly as the stage does."""
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, default_collate
    except ImportError as exc:
        return f"SKIP: {exc}"

    import datasets  # from code/, requires torch

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.float32)

    dataset = datasets.PatchDataset(
        image, mask, _transform, None, None, patch_size=32, stride=32
    )
    print()
    print("dataset (low resolution)")
    print(f"  dataset length           : {len(dataset)}")

    flag = dataset[0][2]
    print(f"  type(keep_flag)          : {type(flag).__module__}.{type(flag).__name__}")
    expect(isinstance(flag, bool), "keep_flag is a Python bool")
    expect(type(flag) is bool, "and not merely an int subclass")

    batch = default_collate([dataset[i] for i in range(len(dataset))])
    print(f"  default_collate          : flag dtype {batch[2].dtype}, shape {tuple(batch[2].shape)}")
    expect(batch[2].dtype == torch.bool, f"collated flags are bool, got {batch[2].dtype}")

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    for patches, positions, keep_flag in loader:
        print(
            f"  DataLoader(workers=0)    : patches {tuple(patches.shape)}, "
            f"positions {[tuple(p.shape) for p in positions]}, flags {tuple(keep_flag.shape)}"
        )
        expect(keep_flag.dtype == torch.bool, "DataLoader flags are bool")

    try:
        workers = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)
        for _p, _pos, keep_flag in workers:
            print(f"  DataLoader(workers=2)    : flags {tuple(keep_flag.shape)} {keep_flag.dtype}")
            expect(keep_flag.dtype == torch.bool, "worker-process flags are bool")
    except (RuntimeError, OSError) as exc:
        print(f"  note: worker-process check skipped ({type(exc).__name__}: {exc})")

    print()
    print("dataset (high resolution)")
    highres = datasets.PatchDataset(
        image, mask, _transform, None, None, patch_size=224, stride=112, high_res=True
    )
    item = highres[0][2]
    print(f"  type(keep_flag)          : {type(item).__module__}.{type(item).__name__}")
    expect(isinstance(item, np.ndarray), "the high-resolution branch already returns an array")
    batch = default_collate([highres[i] for i in range(min(2, len(highres)))])
    print(f"  default_collate          : flag shape {tuple(batch[2].shape)} {batch[2].dtype}")
    expect(batch[2].dtype == torch.bool, "high-resolution flags collate to bool")
    return ""


def main() -> int:
    check_mechanism()
    skipped = check_dataset()
    if skipped:
        print()
        print(skipped)

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("collate regression test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
