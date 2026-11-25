"""Delay gradient reduction until the final accumulation microbatch."""

from contextlib import nullcontext


def gradient_sync_context(module, enabled, is_last_microbatch):
    # This context must enclose both forward and backward. The final backward
    # reduces the accumulated gradients using the normal FSDP/DDP path.
    return module.no_sync() if enabled and not is_last_microbatch else nullcontext()
