"""Verifiable question success controls full-vocabulary Teacher entropy."""
import re
import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist


def answer_letter(text):
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S).strip()
    text = text.replace("**", "").replace("__", "")
    tagged = re.findall(r"<(?:answer|final)>(.*?)</(?:answer|final)>", text, re.S | re.I)
    if tagged:
        text = tagged[-1].strip()
    pure = re.fullmatch(r"[\s(\[]*([A-D])[\s)\].!。]*", text, re.I)
    if pure:
        return pure[1].upper()
    explicit = re.findall(
        r"(?:\b(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|option|choice)"
        r"\s*(?:is|would\s+be)?\s*[:：=\-]?\s*|\\boxed\{)"
        r"[\s(\[]*([A-D])\b", text, re.I)
    return explicit[-1].upper() if explicit else None


def annotate_outcomes(batch, tokenizer, rollout_n, step=None):
    texts = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    groups = defaultdict(list)
    correct, parsed = [], []
    extra_metrics = {}
    verified = None
    if os.environ.get("CORRECTNESS_VERIFIER") == "hybrid_gptoss":
        from evaluation.rollout_verifier import question_text, verify_rows

        prompts = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)
        rows = [{"question": question_text(prompt), "answer": text,
                 "gold": str(reward["ground_truth"]).strip().upper()}
                for prompt, text, reward in zip(prompts, texts, batch.non_tensor_batch["reward_model"])]
        verified, stats = verify_rows(rows, os.environ["CORRECTNESS_JUDGE_API_BASE"],
                                      int(os.environ.get("CORRECTNESS_JUDGE_CONCURRENCY", "32")))
        log_dir = Path(os.environ["CORRECTNESS_JUDGE_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"{step}.jsonl").open("w") as handle:
            for uid, row, result in zip(batch.non_tensor_batch["uid"], rows, verified):
                handle.write(json.dumps({"step": step, "question_uid": str(uid), **row, **result}, ensure_ascii=False) + "\n")
        if stats["unresolved"]:
            raise RuntimeError(f"{stats['unresolved']} GPT-OSS judgments unresolved; saved at {log_dir}; no update applied")
        extra_metrics = {f"correctness/judge_{key}": value for key, value in stats.items()}
        extra_metrics["correctness/legacy_rule_mean"] = sum(answer_letter(r["answer"]) == r["gold"] for r in rows) / len(rows)
        batch.non_tensor_batch["correctness_judge_source"] = np.array([r["source"] for r in verified], dtype=object)
    for i, (uid, text, reward) in enumerate(zip(batch.non_tensor_batch["uid"], texts,
                                                batch.non_tensor_batch["reward_model"])):
        gold = str(reward["ground_truth"]).strip().upper()
        if gold not in {"A", "B", "C", "D"}:
            raise ValueError(f"Expected MCQ gold A-D, got {gold!r}")
        prediction = verified[i]["rule_option"] if verified is not None else answer_letter(text)
        correct.append(float(verified[i]["correct"]) if verified is not None else float(prediction == gold))
        parsed.append(prediction is not None)
        groups[str(uid)].append(i)
    success = torch.empty(len(texts), dtype=torch.float32, device=batch.batch["responses"].device)
    group_scores = []
    for indices in groups.values():
        if len(indices) != rollout_n:
            raise ValueError(f"Expected {rollout_n} complete rollouts per UID, got {len(indices)}")
        value = sum(correct[i] for i in indices) / rollout_n
        success[indices] = value
        group_scores.append(value)
    batch.batch["question_success"] = success
    batch.batch["response_correct"] = torch.tensor(correct, device=success.device)
    batch.batch["answer_parsed"] = torch.tensor(parsed, device=success.device)
    return {"correctness/mean": sum(group_scores) / len(group_scores),
            "correctness/parse_failure_fraction": 1 - sum(parsed) / len(parsed),
            **extra_metrics,
            **{f"correctness/groups_{k}_of_{rollout_n}": sum(c == k / rollout_n for c in group_scores)
               for k in range(rollout_n + 1)}}


def collapse_identical(micro_batch):
    """Within one synchronized microbatch, retain exact response multiplicities."""
    rows = micro_batch.batch["responses"].cpu().tolist()
    masks = micro_batch.batch["response_mask"].cpu().tolist()
    representatives, counts, seen = [], [], {}
    for i, (uid, row, mask) in enumerate(zip(micro_batch.non_tensor_batch["uid"], rows, masks)):
        key = (str(uid), tuple(row), tuple(mask))
        if key not in seen:
            seen[key] = len(representatives)
            representatives.append(i)
            counts.append(0)
        counts[seen[key]] += 1
    result = micro_batch[representatives]
    result.batch["response_multiplicity"] = torch.tensor(counts, dtype=torch.float32,
                                                         device=result.batch["responses"].device)
    return result


def token_denominator(response_mask, device):
    denominator = response_mask.sum().float().to(device)
    if dist.is_initialized():
        dist.all_reduce(denominator)
        denominator /= dist.get_world_size()
    return denominator.clamp_min(1).item()


def _entropy(logits):
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


@torch.no_grad()
def reference_entropy(logits, chunk_size=32):
    flat = logits.detach().reshape(-1, logits.shape[-1])
    return torch.cat([_entropy(chunk.float()) for chunk in flat.split(chunk_size)]).reshape(logits.shape[:-1])


def student_topk_logps(logits, indices, chunk_size=32):
    # FP32 normalization, with gradients to the entire Student vocabulary.
    flat = logits.reshape(-1, logits.shape[-1])
    ids = indices.reshape(-1, indices.shape[-1])
    def reduce(z, support):
        z = z.float()
        return z.gather(-1, support) - z.logsumexp(-1, keepdim=True)
    from torch.utils.checkpoint import checkpoint
    values = [checkpoint(reduce, z, support, use_reentrant=False)
              for z, support in zip(flat.split(chunk_size), ids.split(chunk_size))]
    return torch.cat(values).reshape(indices.shape)


@torch.no_grad()
def soften_teacher(logits, indices, student_entropy, success, response_mask, tolerance=1e-3, chunk_size=32):
    shape = logits.shape[:-1]
    flat = logits.reshape(-1, logits.shape[-1])
    support = indices.reshape(-1, indices.shape[-1])
    hs = student_entropy[:, :shape[1]].reshape(-1)
    c = success[:, None].expand(shape).reshape(-1)
    valid = response_mask[:, :shape[1]].reshape(-1).bool()
    logps, stats = [], []
    for start in range(0, len(flat), chunk_size):
        stop = start + chunk_size
        z = flat[start:stop].float()
        ht = _entropy(z)
        target = ht + (1 - c[start:stop]) * (hs[start:stop] - ht).clamp_min(0)
        active = valid[start:stop] & (target > ht + tolerance)
        tau = torch.ones_like(ht)
        if active.any():
            za, goal = z[active], target[active]
            low, high = torch.ones_like(goal), torch.full_like(goal, 2.)
            while True:
                below = _entropy(za / high[:, None]) < goal
                if not below.any():
                    break
                high[below] *= 2
            for iteration in range(40):
                middle = (low + high) / 2
                hm = _entropy(za / middle[:, None])
                below = hm < goal
                low = torch.where(below, middle, low)
                high = torch.where(below, high, middle)
                if iteration >= 11 and (hm - goal).abs().max() <= tolerance:
                    break
            else:
                raise RuntimeError("Full-vocabulary entropy search did not converge")
            tau[active] = middle
        scaled = z / tau[:, None]
        logps.append(scaled.gather(-1, support[start:stop]) - scaled.logsumexp(-1, keepdim=True))
        actual = _entropy(scaled)
        stats.append(torch.stack((ht, target, actual, tau, (actual-target).abs()), -1))
    stat = torch.cat(stats).reshape(*shape, 5)
    names = ("teacher_entropy", "target_entropy", "actual_entropy", "temperature", "entropy_residual")
    return torch.cat(logps).reshape(indices.shape), {name: stat[..., i] for i, name in enumerate(names)}


def multiplicity_jsd(student_logps, target_logps, mask, multiplicity, denominator):
    from verl.trainer.ppo.attention_gap import auxiliary_jsd
    weights = mask * multiplicity[:, None]
    return (auxiliary_jsd(student_logps, target_logps) * weights).sum() / denominator
