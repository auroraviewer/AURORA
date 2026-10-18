"""Single source of truth for the AURORA unified code-system version.

This module is deliberately dependency-free (standard library only) so that
``import aurora`` never pulls in numpy/torch/pandas.
"""

from __future__ import annotations

#: Version of the *new* unified code system (the ``aurora`` package).
__version__ = "2.0.0"

#: Version of the on-disk combined-checkpoint container format.
#: See :mod:`aurora.bundle`.
BUNDLE_FORMAT = "aurora.combined"
BUNDLE_FORMAT_VERSION = 1

#: Version of the per-run state file written by :mod:`aurora.state`.
RUN_FORMAT = "aurora.run"
RUN_FORMAT_VERSION = 1

#: Version of the run-directory layout produced by :mod:`aurora.paths`.
LAYOUT_FORMAT = "aurora.layout"
LAYOUT_FORMAT_VERSION = 1
