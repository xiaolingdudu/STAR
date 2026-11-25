"""Load the native fused Qwen3.5 MoE safetensors one FSDP unit at a time."""
from collections import Counter
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


def release_file_pages(path, *, written=False):
    # Flushing freshly downloaded files makes their clean pages discardable.
    # This affects Linux page cache only, never the file contents.
    with path.open("rb") as file:
        fd = file.fileno()
        if written:
            os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def release_downloaded_weights(model_dir):
    files = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {model_dir / name for name in files.values()}
    for path in sorted(shards):
        release_file_pages(path, written=True)
    print(f"[397b-host-memory] released downloaded weight pages for {len(shards)} local shards", flush=True)


def checkpoint_initializer(model, model_dir, device, *, source_rank, release_pages=False):
    model_dir = Path(model_dir)
    files = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    missing = set(dict(model.named_parameters())) - files.keys()
    if missing:
        raise ValueError(f"Checkpoint has no directly mapped parameter {min(missing)}")
    prefixes = {module: name for name, module in model.named_modules()}
    remaining = Counter(files[name] for name, _ in model.named_parameters()) if source_rank and release_pages else None

    @torch.no_grad()
    def materialize(module):
        # FSDP visits each unit before flattening/sharding it, then broadcasts
        # rank zero's weights via sync_module_states. Nonpersistent RoPE
        # buffers were already initialized by the model constructor on CPU.
        for name, param in module.named_parameters(recurse=False):
            if not param.is_meta:
                continue
            key = ".".join(filter(None, (prefixes[module], name)))
            if source_rank:
                with safe_open(str(model_dir / files[key]), framework="pt", device="cpu") as shard:
                    value = shard.get_tensor(key)
                    if value.shape != param.shape:
                        raise ValueError(f"{key}: checkpoint {value.shape}, model {param.shape}")
                    value = value.to(device=device, dtype=param.dtype)
                if remaining is not None:
                    shard_name = files[key]
                    remaining[shard_name] -= 1
                    if remaining[shard_name] == 0:
                        release_file_pages(model_dir / shard_name)
            else:
                value = torch.empty_like(param, device=device)
            setattr(module, name, torch.nn.Parameter(value, requires_grad=param.requires_grad))
        for name, buffer in module.named_buffers(recurse=False):
            module._buffers[name] = buffer.to(device=device)
    return materialize
