"""
Print to the console and to a report file at the same time.

cmd has no `tee`, and redirecting a long-running script to a file leaves the
window blank for as long as it runs -- which for a multi-gigabyte download
looks exactly like a hang. So the scripts duplicate their own output instead
of being redirected by the .bat that launches them.
"""

import atexit
import sys
import time


class _Tee:
    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()          # so the console keeps up with a slow step
        try:
            self._handle.write(text)
            self._handle.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        self._stream.flush()
        try:
            self._handle.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._stream, 'isatty', lambda: False)()


def tee_to(path):
    """Send stdout and stderr to `path` as well as the console."""
    try:
        handle = open(path, 'w', encoding='utf-8', errors='replace')
    except Exception:
        return None
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    atexit.register(handle.close)
    return path


_started = time.time()


def elapsed():
    """mm:ss since the script started, for steps that take a while."""
    seconds = int(time.time() - _started)
    return f"{seconds // 60:d}:{seconds % 60:02d}"
