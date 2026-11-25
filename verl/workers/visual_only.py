"""Train the Qwen3.5 visual encoder and merger through a frozen language model."""

import torch
import torch.distributed as dist


def freeze_language_for_visual_opd(model):
    visual = model.model.visual
    model.requires_grad_(False)
    visual.requires_grad_(True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in visual.parameters())
    merger = sum(p.numel() for p in visual.merger.parameters())
    return {"total": total, "trainable_visual": trainable,
            "trainable_merger": merger, "frozen_language": total - trainable}


def visual_gradient_stats(model):
    device = next(model.parameters()).device
    values = torch.zeros(3, device=device)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not param.requires_grad:
            values[2] += 1
        else:
            index = 1 if "merger" in name else 0
            values[index] += param.grad.detach().float().square().sum()
    if dist.is_initialized():
        dist.all_reduce(values)
    return {"visual_only/encoder_grad_norm": values[0].sqrt().item(),
            "visual_only/merger_grad_norm": values[1].sqrt().item(),
            "visual_only/frozen_grad_tensors": values[2].item()}
