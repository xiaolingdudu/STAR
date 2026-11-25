"""Share repeated encoded image inputs across actor dispatch chunks."""

import time

import numpy as np
import ray
import torch


def share_multimodal_inputs(data):
    """Replace image dictionaries with Ray refs without changing the source batch.

    Deduplication uses tensor identity, not uid: different images for the same
    question remain different. Agent-loop caches return shallow dictionary
    copies; these still share the actual tensor objects.
    """
    from verl import DataProto

    started = time.perf_counter()
    columns = [key for key in data.non_tensor_batch if key.endswith("multi_modal_inputs")]
    shared = DataProto(batch=data.batch, non_tensor_batch=dict(data.non_tensor_batch),
                       meta_info=dict(data.meta_info))
    refs_by_identity = {}
    unique_bytes = logical_bytes = 0
    for column in columns:
        refs = np.empty(len(data), dtype=object)
        for i, inputs in enumerate(data.non_tensor_batch[column]):
            identity = None if inputs is None else tuple((key, id(inputs[key])) for key in sorted(inputs))
            if identity not in refs_by_identity:
                # NumPy buffers use Ray's object-store transport. A byte view
                # preserves all tensor dtypes, including bfloat16, exactly.
                payload = None if inputs is None else {
                    key: (value.dtype, tuple(value.shape),
                          value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
                    for key, value in inputs.items()
                }
                size = 0 if payload is None else sum(value[2].nbytes for value in payload.values())
                refs_by_identity[identity] = (ray.put(payload), size)
                unique_bytes += size
            refs[i], size = refs_by_identity[identity]
            logical_bytes += size
        shared.non_tensor_batch[column] = refs
    shared.meta_info["shared_multimodal_columns"] = columns
    metrics = {
        "timing_s/update_actor/share_multimodal": time.perf_counter() - started,
        "perf/shared_mm_objects": len(refs_by_identity),
        "perf/shared_mm_logical_gb": logical_bytes / 1024**3,
        "perf/shared_mm_unique_gb": unique_bytes / 1024**3,
    }
    return shared, metrics


def restore_multimodal_inputs(data):
    """Resolve each unique image once per worker, before microbatch slicing."""
    columns = data.meta_info.pop("shared_multimodal_columns", [])
    if not columns:
        return
    refs = list(dict.fromkeys(ref for key in columns for ref in data.non_tensor_batch[key]))
    decoded = {}
    for ref, payload in zip(refs, ray.get(refs)):
        # Copy once per unique image so downstream tensors remain writable.
        decoded[ref] = None if payload is None else {
            key: torch.from_numpy(raw.copy()).view(dtype).reshape(shape)
            for key, (dtype, shape, raw) in payload.items()
        }
    for key in columns:
        values = np.empty(len(data), dtype=object)
        for i, ref in enumerate(data.non_tensor_batch[key]):
            values[i] = decoded[ref]
        data.non_tensor_batch[key] = values
