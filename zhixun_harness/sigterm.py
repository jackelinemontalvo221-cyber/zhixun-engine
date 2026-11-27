"""Exit promptly on SIGTERM.

In a container the harness process is often PID 1, and the kernel does not apply the
default SIGTERM action to PID 1: without an explicit handler a platform stop is ignored
until SIGKILL.  The SDK only imports ``contest_submission`` after it has read the start
frame, so the handler must exist earlier: the image installs ``zhixun_sigterm.pth`` in
site-packages, which imports this module when the interpreter starts.

SDK 1.0.5 forbids continuing RPC after an interruption, so the handler just exits.
"""

from __future__ import annotations

import signal
import threading


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


def install() -> bool:
    if threading.current_thread() is not threading.main_thread():
        return False
    if signal.getsignal(signal.SIGTERM) is _terminate:
        return True
    signal.signal(signal.SIGTERM, _terminate)
    return True


install()
