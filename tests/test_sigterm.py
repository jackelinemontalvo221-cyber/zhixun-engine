"""The SIGTERM hook must turn a platform stop into a prompt exit."""

import signal
import subprocess
import sys
import time


def test_handler_raises_system_exit():
    from zhixun_harness import sigterm

    assert sigterm.install() is True
    assert signal.getsignal(signal.SIGTERM) is sigterm._terminate
    try:
        sigterm._terminate(signal.SIGTERM, None)
    except SystemExit as exc:
        assert exc.code == 128 + signal.SIGTERM
    else:  # pragma: no cover
        raise AssertionError("handler did not exit")


def test_blocked_reader_exits_on_sigterm():
    code = "import zhixun_harness.sigterm, sys; sys.stdin.readline()"
    process = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE)
    time.sleep(1.0)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=10) == 128 + signal.SIGTERM
