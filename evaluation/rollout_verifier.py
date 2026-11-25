"""Online MCQ verification shared by training and saved-rollout checks."""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from evaluation.vision_opd_official.repair_judges import PROMPT_TEMPLATE, canonical_label


def visible_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.replace("**", "").replace("__", "").strip()
    tagged = re.findall(r"<(?:answer|final)>(.*?)</(?:answer|final)>", text, re.S | re.I)
    return tagged[-1].strip() if tagged else text


def clear_option(text: str) -> str | None:
    """Only accept an answer declaration or an option at a line boundary."""
    text = visible_answer(text)
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        explicit = re.match(
            r"^(?:the\s+)?(?:final\s+answer|correct\s+(?:answer|option|choice)|answer)"
            r"\s*(?:is|would\s+be)?\s*[:：=\-]?\s*[\s(\[]*([A-D])\b", line, re.I)
        option = re.match(r"^[\s(\[]*([A-D])(?:\s*[)\].!。:]|\s*$)", line)
        match = explicit or option
        if match:
            candidates.append(match[1].upper())
    boxed = re.findall(r"\\boxed\{\s*([A-D])\s*\}", text)
    candidates.extend(boxed)
    choices = set(candidates)
    # Conflicting declarations and option lists go to the semantic judge.
    return next(iter(choices)) if len(choices) == 1 else None


def question_text(prompt: str) -> str:
    users = re.findall(r"<\|im_start\|>user\s*\n(.*?)(?:<\|im_end\|>|$)", prompt, re.S)
    text = users[-1] if users else prompt
    return re.sub(r"<\|[^>]+\|>|<image>", "", text).strip()


def judge_batch(rows: list[dict], api_base: str, concurrency: int = 32) -> tuple[list[dict], dict]:
    from openai import OpenAI

    clients = threading.local()

    def call(row):
        if not hasattr(clients, "client"):
            clients.client = OpenAI(api_key="EMPTY", base_url=api_base, timeout=180, max_retries=0)
        prompt = PROMPT_TEMPLATE.format(question=row["question"], gt=row["gold"], response=row["answer"])
        history = []
        for attempt in range(3):
            kwargs = {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0, "max_tokens": 2048 if attempt == 0 else 8192, "seed": 42}
            if attempt:
                kwargs.update(reasoning_effort="low", extra_body={"structured_outputs": {"choice": ["Yes", "No"]}})
            try:
                response = clients.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                raw = choice.message.content or ""
                label = canonical_label(raw)
                history.append({"raw": raw, "finish_reason": choice.finish_reason, "max_tokens": kwargs["max_tokens"]})
                if label:
                    return {"correct": int(label == "Yes"), "label": label, "attempts": history}
            except Exception as exc:
                history.append({"error": f"{type(exc).__name__}: {exc}"})
            if attempt < 2:
                time.sleep(1)
        return {"correct": None, "label": "", "attempts": history}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(call, rows))
    return results, {"requests": len(rows), "concurrency": concurrency,
                     "seconds": time.perf_counter() - started,
                     "retries": sum(len(r["attempts"]) - 1 for r in results),
                     "unresolved": sum(r["correct"] is None for r in results)}


def verify_rows(rows: list[dict], api_base: str, concurrency: int = 32) -> tuple[list[dict], dict]:
    results, pending, keys = [], {}, []
    for row in rows:
        option = clear_option(row["answer"])
        empty = not visible_answer(row["answer"])
        result = {"rule_option": option, "source": "empty" if empty else "rule" if option else "gpt_oss",
                  "correct": 0 if empty else int(option == row["gold"]) if option else None}
        results.append(result)
        key = (row["question"], row["gold"], row["answer"])
        keys.append(key)
        if result["correct"] is None:
            pending.setdefault(key, row)
    judgments, stats = judge_batch(list(pending.values()), api_base, concurrency)
    lookup = dict(zip(pending, judgments))
    for key, result in zip(keys, results):
        if result["correct"] is None:
            result.update(lookup[key])
    stats["fallback_rollouts"] = sum(r["source"] == "gpt_oss" for r in results)
    return results, stats

