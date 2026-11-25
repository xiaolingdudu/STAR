"""Memory-bounded reductions over large vocabulary logits."""

import torch


def select_response_logits(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Reuse response-only model logits when their token positions already match."""
    if indices.shape[1] == logits.shape[1]:
        identity = torch.arange(logits.shape[1], device=indices.device).expand_as(indices)
        if torch.equal(indices, identity):
            return logits
    return logits.gather(1, indices[..., None].expand(-1, -1, logits.shape[-1]))


def chunked_logsumexp(logits: torch.Tensor, chunk_rows: int = 256) -> torch.Tensor:
    """Compute logsumexp over the vocabulary without a full-row temporary."""
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    reduced = [
        torch.logsumexp(chunk, dim=-1, keepdim=True)
        for chunk in flat_logits.split(chunk_rows, dim=0)
    ]
    return torch.cat(reduced, dim=0).reshape(*logits.shape[:-1], 1)


def chunked_topk(
    logits: torch.Tensor,
    k: int,
    chunk_rows: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k over the vocabulary with bounded workspace."""
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    chunks = [torch.topk(chunk, k, dim=-1) for chunk in flat_logits.split(chunk_rows, dim=0)]
    values = torch.cat([item.values for item in chunks], dim=0)
    indices = torch.cat([item.indices for item in chunks], dim=0)
    output_shape = (*logits.shape[:-1], k)
    return values.reshape(output_shape), indices.reshape(output_shape)
