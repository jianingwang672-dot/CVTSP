"""OS-released file locks cap simultaneous high-memory phases across jobs."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time


@contextmanager
def gpu_phase_slot(directory, slots=2):
    if slots < 1:
        raise ValueError("GPU slots must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handles = [(directory / f"slot-{i}.lock").open("a+") for i in range(slots)]
    acquired = None
    started = time.perf_counter()
    waiting = directory / f"waiting-{os.getpid()}"
    waiting.touch()
    try:
        while acquired is None:
            for handle in handles:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                acquired = handle
                waiting.unlink(missing_ok=True)
                handle.seek(0)
                handle.truncate()
                handle.write(str(os.getpid()))
                handle.flush()
                break
            if acquired is None:
                time.sleep(.1)
        yield time.perf_counter() - started
    finally:
        waiting.unlink(missing_ok=True)
        if acquired is not None:
            fcntl.flock(acquired.fileno(), fcntl.LOCK_UN)
        for handle in handles:
            handle.close()
        if acquired is not None and any(directory.glob("waiting-*")):
            # Give already-waiting jobs a chance before this job's next batch.
            time.sleep(.15)
