"""Build member 5 acceptance summaries from local round JSON files.

The implementation lives in ``member5.report_cli`` so it can also be run with
``python -m member5.report_cli`` in restricted contest environments.
"""

from __future__ import annotations

from member5.report_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
