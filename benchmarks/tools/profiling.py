from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


def nvtx_range(name: str, enabled: bool):
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


@contextmanager
def capture_cuda(enabled: bool, device):
    """Capture one diagnostic replay; never synchronize individual steps."""
    if not enabled:
        yield
        return

    torch.cuda.synchronize(device)
    torch.cuda.profiler.start()
    error = None

    try:
        with nvtx_range("tinyvllm/measurement", True):
            yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        # Attempt stop even if synchronization fails. Preserve the first error.
        for cleanup in (
            lambda: torch.cuda.synchronize(device),
            torch.cuda.profiler.stop,
        ):
            try:
                cleanup()
            except BaseException as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"profiler cleanup failed: {exc!r}")

        if error is not None:
            raise error