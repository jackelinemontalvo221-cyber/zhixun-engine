"""Small filesystem helpers shared by atomic writers."""

from __future__ import annotations

import os
import time
from pathlib import Path


TRANSIENT_WINDOWS_REPLACE_ERRORS = frozenset({5, 32, 33})


def replace_with_retry(
    source: str | Path,
    destination: str | Path,
    *,
    attempts: int = 8,
    base_delay: float = 0.02,
) -> None:
    """Atomically replace a path, retrying transient Windows sharing errors.

    WinError 5/32/33 can be raised briefly while antivirus, indexing, or a
    recently closed file handle releases the destination.  Other filesystem
    errors are propagated immediately so real permission and contract failures
    remain visible.
    """

    source = Path(source)
    destination = Path(destination)
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient = getattr(exc, "winerror", None) in TRANSIENT_WINDOWS_REPLACE_ERRORS
            if not transient or attempt + 1 >= attempts:
                raise
            time.sleep(min(float(base_delay) * (2 ** attempt), 0.25))
