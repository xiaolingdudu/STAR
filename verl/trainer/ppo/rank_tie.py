"""Detached minimal probability pooling toward the Teacher's top-1 token."""

import torch


def rank_tie_token_loss(student_logits, teacher_top1, response_mask, initial_topk=100):
    """Return KL(sg[r] || p) per token, with its exact first-order gradient.

    Pool the Teacher choice with the smallest prefix of Student-ranked tokens
    needed to make it a joint winner. Candidates outside that pool keep their
    probabilities. Top-k expands when its boundary still exceeds the pooling
    threshold, so neither the Teacher choice nor additional blockers are lost.

    Only pooled coordinates have a nonzero logit gradient (p-r). We use that
    sparse expression plus a detached value correction to avoid constructing
    another [response_tokens, vocabulary] target and autograd graph. This is
    first-order training, not a higher-order differentiable projection.
    """
    shape = student_logits.shape[:-1]
    vocab = student_logits.shape[-1]
    flat = student_logits.reshape(-1, vocab)
    teacher = teacher_top1.reshape(-1).detach()
    valid = response_mask.reshape(-1).bool()
    with torch.no_grad():
        disagree = valid & (flat.argmax(-1) != teacher)
        selected = disagree.nonzero().flatten()
    loss = flat[:, 0] * 0.0
    totals = student_logits.new_zeros(4, dtype=torch.float32)
    for rows in selected.split(64):
        with torch.no_grad():
            values = flat.detach().index_select(0, rows)
            dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
            log_z = values.to(dtype).logsumexp(-1, keepdim=True)
            chosen = teacher[rows, None]
            chosen_logp = values.gather(-1, chosen).to(dtype) - log_z
            chosen_p = chosen_logp.exp()
            k = min(initial_topk, vocab)
            while True:
                top_logits, indices = values.topk(k, dim=-1)
                logp = top_logits.to(dtype) - log_z
                p = logp.exp()
                other = indices != chosen
                means = (chosen_p + (p * other).cumsum(-1)) / (1 + other.cumsum(-1))
                threshold = means.max(-1, keepdim=True).values
                if k == vocab or bool((threshold >= p[:, -1:]).all()):
                    break
                k = min(2 * k, vocab)
            pooled = other & (p > threshold)
            delta = torch.where(pooled, p - threshold, 0.0)
            chosen_delta = chosen_p - threshold
            log_threshold = threshold.log()
            kl_value = (pooled * threshold * (log_threshold - logp)).sum(-1)
            kl_value += (threshold * (log_threshold - chosen_logp)).squeeze(-1)
            totals += torch.stack([
                (pooled.sum(-1) + 1).sum(),
                (threshold - chosen_p).sum(),
                p[:, 0].sum(),
                (~(indices[:, :min(initial_topk, vocab)] == chosen).any(-1)).sum(),
            ]).to(totals.dtype)
        selected_logits = flat[rows[:, None], indices].to(dtype)
        chosen_logits = flat[rows, teacher[rows]].to(dtype)
        gradient_term = (delta * selected_logits).sum(-1) + chosen_delta.squeeze(-1) * chosen_logits
        exact_loss = gradient_term + (kl_value - gradient_term).detach()
        loss = loss.index_copy(0, rows, exact_loss.to(loss.dtype))
    active_count = selected.numel()
    denominator = max(active_count, 1)
    metrics = {
        "rank_tie/disagreement_fraction": active_count / max(int(valid.sum()), 1),
        "rank_tie/pool_size_mean": float(totals[0]) / denominator,
        "rank_tie/probability_shift_mean": float(totals[1]) / denominator,
        "rank_tie/student_winner_probability": float(totals[2]) / denominator,
        "rank_tie/teacher_outside_student_top100_fraction": float(totals[3]) / denominator,
    }
    return loss.reshape(shape), metrics
