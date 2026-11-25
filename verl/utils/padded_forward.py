"""Trim shared padding without joining independent recurrent sequences."""

import torch


def trim_padded_micro_batch(batch):
    """Keep prompt context and the longest live response, preserving row positions."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    batch_size, sequence_length = ids.shape
    old_response_length = batch["responses"].shape[1]
    starts = batch.get("response_start_idx")
    if starts is None:
        starts = torch.full((batch_size,), sequence_length - old_response_length,
                            dtype=torch.long, device=ids.device)
    offsets = torch.arange(old_response_length, device=ids.device)
    response_mask = mask.gather(1, starts[:, None] + offsets[None])
    response_length = int(((offsets + 1) * response_mask.any(0)).max().clamp_min(1).item())
    first_live = int(mask.any(0).to(torch.int32).argmax().item())
    stop = int((starts + response_length).max().item())
    trimmed = dict(batch)
    for key in ("input_ids", "attention_mask", "position_ids"):
        trimmed[key] = batch[key][..., first_live:stop]
    trimmed["responses"] = batch["responses"][:, :response_length]
    trimmed["response_start_idx"] = starts - first_live
    if "response_mask" in batch:
        trimmed["response_mask"] = batch["response_mask"][:, :response_length]
    fraction_removed = 1.0 - (stop - first_live) / sequence_length
    return trimmed, old_response_length, fraction_removed


def restore_response_width(outputs, response_length):
    """Masked suffixes retain the original trainer tensor contract."""
    for key, value in outputs.items():
        if not torch.is_tensor(value) or value.ndim < 2:
            continue
        missing = response_length - value.shape[1]
        if missing:
            outputs[key] = torch.cat((value, value.new_zeros(
                (value.shape[0], missing, *value.shape[2:]))), dim=1)
    return outputs
