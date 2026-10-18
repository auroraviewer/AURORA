"""Inference-only entry points of the new code system.

Modules in this package are the *only* ones that read a combined checkpoint to
produce predictions:

* :mod:`aurora.inference.predict_lowres`  -- the low-resolution model,
* :mod:`aurora.inference.predict_highres` -- the high-resolution refinement.

They are importable as libraries and runnable as modules
(``python -m aurora.inference.predict_lowres``), and they depend on the
``inference`` dependency group only.
"""
