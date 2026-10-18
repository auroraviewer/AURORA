"""Typed exceptions for the AURORA unified code system.

Every error carries a ``hint`` describing the concrete next action, so the CLI
can print something actionable instead of a bare traceback.  Keeping the
exception types in one dependency-free module lets every other module raise
them without creating import cycles.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence


class AuroraError(Exception):
    """Base class for every error raised by the ``aurora`` package."""

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def format(self) -> str:
        """Render ``message`` plus, when present, the suggested next action."""
        if self.hint:
            return f"{self.message}\n\nHow to fix:\n{self.hint}"
        return self.message


class ConfigError(AuroraError):
    """A run configuration value is missing, malformed or inconsistent."""


class MissingDependencyError(AuroraError):
    """A required third-party package is not installed.

    Raised by :func:`aurora.deps.require` with a message naming the requirement
    group (``inference`` / ``training`` / ...) that provides the package.
    """


class CheckpointFormatError(AuroraError):
    """A checkpoint file is not a combined AURORA checkpoint.

    The new code system is *only* compatible with the combined checkpoint
    produced by ``tools/merge_aurora_checkpoints.py``; legacy single-stage
    ``.pth`` files raise this error with instructions on how to merge them.
    """


class StagePrerequisiteError(AuroraError):
    """A pipeline stage was requested before its prerequisites completed.

    This is the guard that requirement #2 of the reorganisation asks for: the
    high-resolution model cannot be run without the low-resolution model.

    ``missing`` lists the prerequisite stages in the order they must run and
    ``stage`` is the stage that was requested.
    """

    def __init__(
        self,
        stage: str,
        missing: Sequence[str],
        reason: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> None:
        self.stage = stage
        self.missing: tuple[str, ...] = tuple(missing)
        self.reason = reason
        if reason is None:
            reason = (
                f"stage '{stage}' cannot run because the following prerequisite "
                f"stage(s) have not completed for this run: {', '.join(self.missing)}"
            )
        super().__init__(reason, hint=hint)


class StageFailedError(AuroraError):
    """A stage started but did not finish successfully."""

    def __init__(
        self,
        stage: str,
        returncode: int,
        log_path: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> None:
        self.stage = stage
        self.returncode = returncode
        self.log_path = log_path
        message = f"stage '{stage}' failed with exit code {returncode}"
        if log_path:
            message += f" (log: {log_path})"
        super().__init__(message, hint=hint)


class LayoutError(AuroraError):
    """A run directory does not match the expected layout."""


class SampleSheetError(AuroraError):
    """The sample sheet is missing a required column or has duplicate samples."""


class NotInferenceEnvironmentError(AuroraError):
    """An inference-only command was asked to do something that needs training extras."""


def format_error_list(items: Iterable[str], bullet: str = "  - ") -> str:
    """Small helper used by several ``hint`` builders."""
    return "\n".join(f"{bullet}{item}" for item in items)
