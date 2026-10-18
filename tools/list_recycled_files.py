#!/usr/bin/env python
"""List what the Recycle Bin holds for this project, with original paths.

Read-only: it parses the ``$I*`` metadata records (original path, size and
deletion time) and prints them. Nothing is restored or deleted.

    python tools/list_recycled_files.py                 # entries from this project
    python tools/list_recycled_files.py --all           # every recycled entry
    python tools/list_recycled_files.py --since 2026-10-15

Windows only; on other platforms it prints a short explanation.
"""

from __future__ import annotations

import argparse
import datetime
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Windows FILETIME epoch.
_FILETIME_EPOCH = datetime.datetime(1601, 1, 1, tzinfo=datetime.timezone.utc)


def recycle_bin_dirs() -> List[Path]:
    """Every per-user Recycle Bin directory on the system drive."""
    if os.name != "nt":  # pragma: no cover - platform guard
        return []
    root = Path(os.environ.get("SystemDrive", "C:") + "\\$Recycle.Bin")
    if not root.is_dir():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def parse_record(meta: Path) -> Optional[Tuple[datetime.datetime, int, str]]:
    """Return ``(deleted_utc, size, original_path)`` for an ``$I`` record."""
    try:
        data = meta.read_bytes()
    except OSError:
        return None
    if len(data) < 28:
        return None
    size = struct.unpack_from("<Q", data, 8)[0]
    filetime = struct.unpack_from("<Q", data, 16)[0]
    when = _FILETIME_EPOCH + datetime.timedelta(microseconds=filetime / 10) if filetime else None
    name_length = struct.unpack_from("<I", data, 24)[0]
    raw = data[28 : 28 + name_length * 2]
    try:
        original = raw.decode("utf-16-le").rstrip("\x00")
    except UnicodeDecodeError:
        original = "<?>"
    return (when, size, original)  # type: ignore[return-value]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Show every recycled entry, not just this project's")
    parser.add_argument("--since", type=str, default=None, help="Only entries deleted on/after this date (YYYY-MM-DD)")
    parser.add_argument("--project", type=str, default=str(REPO_ROOT), help="Project path to filter on")
    args = parser.parse_args(argv)

    if os.name != "nt":
        print("The Recycle Bin is a Windows feature; nothing to list on this platform.")
        return 0

    since: Optional[datetime.datetime] = None
    if args.since:
        since = datetime.datetime.fromisoformat(args.since).replace(tzinfo=datetime.timezone.utc)

    needles = {args.project.replace("/", "\\")}
    try:
        needles.add(str(Path(args.project).resolve()))
    except OSError:
        pass

    rows: List[Tuple[datetime.datetime, int, str]] = []
    for directory in recycle_bin_dirs():
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            # Other users' Recycle Bins are not readable; skip them.
            continue
        for entry in entries:
            if not entry.name.startswith("$I"):
                continue
            record = parse_record(entry)
            if record is None or record[0] is None:
                continue
            when, size, original = record
            if since is not None and when < since:
                continue
            if not args.all and not any(needle.lower() in original.lower() for needle in needles):
                continue
            rows.append((when, size, original))

    rows.sort(key=lambda row: row[0])
    if not rows:
        print("Nothing matching was found in the Recycle Bin.")
        return 0

    print(f"{len(rows)} recycled entr{'y' if len(rows) == 1 else 'ies'}:")
    print()
    print(f"{'deleted (local)':<20} {'size':>14}  original path")
    print("-" * 100)
    for when, size, original in rows:
        local = when.astimezone()
        print(f"{local:%Y-%m-%d %H:%M:%S}  {size:>14,}  {original}")
    print()
    print("These can be restored with Explorer (Recycle Bin -> Restore).")
    print("This tool never restores or deletes anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
