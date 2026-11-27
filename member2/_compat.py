"""Find the official ``baselines`` package without depending on member-1 internals.

Member 1's ``zhixun_harness.official.import_baselines`` is used when it exists.
Otherwise the official package is looked up the same way: an importable
``baselines`` module, then ``<repo>/vendor/baselines``, then the directory named by
the ``ZHIXUN_BASELINES`` environment variable (the folder that contains ``baselines/``).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def import_baselines():
    try:
        from zhixun_harness.official import import_baselines as official
    except ImportError:
        official = None
    if official is not None:
        return official()
    candidates = [Path(__file__).resolve().parents[1] / "vendor"]
    if os.environ.get("ZHIXUN_BASELINES"):
        candidates.append(Path(os.environ["ZHIXUN_BASELINES"]))
    try:
        package = importlib.import_module("baselines")
    except ImportError:
        for folder in candidates:
            if (folder / "baselines").is_dir() and str(folder) not in sys.path:
                sys.path.insert(0, str(folder))
        package = importlib.import_module("baselines")
    for name in ("model", "data", "predict"):
        importlib.import_module(f"baselines.{name}")
    return package
