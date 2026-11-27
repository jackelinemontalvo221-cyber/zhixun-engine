"""Platform adapters.

``WorkspaceV2Source`` is the implemented adapter for contest_harness 1.0.4.
``PlatformStageSource`` is a deliberate stub for a future contract.
"""

from .sources import PlatformStageSource, WorkspaceV2Source

__all__ = ["PlatformStageSource", "WorkspaceV2Source"]
