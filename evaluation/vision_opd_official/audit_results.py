#!/usr/bin/env python3
"""Audit answer and judge completeness for the official evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--answer-json", required=True, type=Path)
    parser.add_argument("--judge-json", type=Path)
    parser.add_argument("--answer-source", choices=["generated", "reused", "resumed"], default="generated")
    parser.add_argument("--answer-source-path", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-pending", action="store_true")
    return parser.parse_args()


def stable_uid(item, benchmark):
    if item.get("sample_uid"):
        return str(item["sample_uid"])
    for key in ("uid", "index", "question_id", "id", "sample_id"):
        if item.get(key) is not None and str(item[key]) != "":
            return f"{benchmark}:{key}:{item[key]}"
    raise ValueError("Missing sample ID; use scripts/evaluate.py to prepare stable row IDs.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def read_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"Expected a JSON list of objects: {path}")
    return value


def answer_problem(row, field="model_answer"):
    status = row.get("inference_status")
    answer = row.get(field)
    if status in {"api_error", "future_error"}:
        return status
    if not isinstance(answer, str) or not answer.strip():
        return "empty"
    if answer.strip().startswith(("[API_ERROR]", "[FUTURE_ERROR]")):
        return "generation_error"
    return None


def pending_answers(benchmark, benchmark_json, answer_json):
    inputs = read_json_list(benchmark_json)
    expected = {stable_uid(row, benchmark): row for row in inputs}
    if len(expected) != len(inputs):
        raise ValueError(f"Duplicate source ID in {benchmark}")
    rows = read_jsonl(answer_json)
    seen = set()
    pending = []
    for row in rows:
        uid = row["sample_uid"]
        if uid in seen or uid not in expected:
            raise ValueError(f"Unexpected or duplicate answer ID in {benchmark}: {uid}")
        if any(row.get(k) != expected[uid].get(k) for k in ("images", "query", "response")):
            raise ValueError(f"Changed input in {benchmark}: {uid}")
        seen.add(uid)
        problem = answer_problem(row)
        if problem:
            pending.append(dict(sample_id=uid, reason=problem, finish_reason=row.get("finish_reason")))
    pending.extend(dict(sample_id=uid, reason="missing") for uid in sorted(expected.keys() - seen))
    return dict(benchmark=benchmark, expected=len(expected), valid=len(expected)-len(pending), pending=pending)


def audit(
    *,
    benchmark: str,
    benchmark_json: Path,
    answer_json: Path,
    judge_json: Path | None,
    answer_source: str,
    answer_source_path: str,
) -> dict[str, Any]:
    benchmark_rows = read_json_list(benchmark_json)
    answers = read_jsonl(answer_json)
    expected = len(benchmark_rows)
    if len(answers) != expected:
        raise ValueError(
            f"{benchmark} answer count mismatch: expected {expected}, found {len(answers)} in {answer_json}"
        )

    uids = [stable_uid(item, benchmark) for item in answers]
    unique_uids = set(uids)
    if len(unique_uids) != expected:
        raise ValueError(f"{benchmark} has {expected - len(unique_uids)} duplicate answer sample IDs.")

    invalid_answers = []
    empty_success_answers = []
    for index, item in enumerate(answers):
        value = item.get("model_answer")
        stripped = value.strip() if isinstance(value, str) else ""
        inference_status = item.get("inference_status")
        if not stripped and inference_status == "ok":
            empty_success_answers.append(index)
        elif (
            not stripped
            or inference_status in {"api_error", "future_error"}
            or stripped.startswith("[API_ERROR]")
            or stripped.startswith("[FUTURE_ERROR]")
        ):
            invalid_answers.append(index)
    if invalid_answers:
        raise ValueError(
            f"{benchmark} has {len(invalid_answers)} empty or failed answers; "
            f"first indices: {invalid_answers[:10]}"
        )

    report: dict[str, Any] = {
        "benchmark": benchmark,
        "benchmark_json": str(benchmark_json.resolve()),
        "answer_json": str(answer_json.resolve()),
        "answer_source": answer_source,
        "answer_source_path": answer_source_path,
        "expected_samples": expected,
        "answer_samples": len(answers),
        "unique_answer_samples": len(unique_uids),
        "invalid_answers": 0,
        "empty_success_answers": len(empty_success_answers),
    }
    if benchmark == "vstar" and any(item.get("evaluation_protocol") == "vstar-val-v1" for item in answers):
        if any(item.get("evaluation_protocol") != "vstar-val-v1" for item in answers):
            raise ValueError("VStar answers mix val and historical official protocols.")
        report["evaluation_protocol"] = "vstar-val-v1"
    if any(item.get("evaluation_protocol") == "full6-val-v1" for item in answers):
        if any(item.get("evaluation_protocol") != "full6-val-v1" for item in answers):
            raise ValueError("Answers mix full6-val-v1 and other protocols.")
        report["evaluation_protocol"] = "full6-val-v1"
        report["length_limited_answers"] = sum(item.get("finish_reason") == "length" for item in answers)

    if judge_json is not None:
        judges = read_json_list(judge_json)
        if len(judges) != expected:
            raise ValueError(
                f"{benchmark} judge count mismatch: expected {expected}, found {len(judges)} in {judge_json}"
            )
        judge_uids = [stable_uid(item, benchmark) for item in judges]
        if len(set(judge_uids)) != expected:
            raise ValueError(f"{benchmark} judge output contains duplicate sample IDs.")
        if judge_uids != uids:
            raise ValueError(f"{benchmark} judge sample order does not match the audited answers.")
        if report.get("evaluation_protocol") in {"vstar-val-v1", "full6-val-v1"}:
            allowed = {"first_letter_rule"}
            if benchmark == "zoombench" and report["evaluation_protocol"] == "full6-val-v1":
                allowed.add("numeric_rule")
            if any(item.get("judge_source") not in allowed for item in judges):
                raise ValueError("Val protocol requires rule-only judgments.")
            report["unparsed_answers"] = sum(not item.get("pred") for item in judges)
            if report["evaluation_protocol"] == "full6-val-v1" and all(item.get("scoring_version") == "mcq-and-numeric-v2" for item in judges):
                report["scoring_version"] = "mcq-and-numeric-v2"
                report["numeric_questions"] = sum(item.get("judge_source") == "numeric_rule" for item in judges)
        invalid_labels = [
            index
            for index, item in enumerate(judges)
            if str(item.get("judge", "")).strip() not in {"Yes", "No"}
        ]
        if invalid_labels:
            raise ValueError(
                f"{benchmark} has {len(invalid_labels)} noncanonical judge labels; "
                f"first indices: {invalid_labels[:10]}"
            )
        correct = sum(item["judge"] == "Yes" for item in judges)
        report.update(
            {
                "judge_json": str(judge_json.resolve()),
                "judge_samples": len(judges),
                "unique_judge_samples": len(set(judge_uids)),
                "noncanonical_judges": 0,
                "correct": correct,
                "accuracy": 100.0 * correct / expected if expected else 0.0,
            }
        )

    return report


def main() -> None:
    args = parse_args()
    if args.record_pending:
        report = pending_answers(args.benchmark, args.benchmark_json, args.answer_json)
    else:
        report = audit(
            benchmark=args.benchmark, benchmark_json=args.benchmark_json, answer_json=args.answer_json,
            judge_json=args.judge_json, answer_source=args.answer_source, answer_source_path=args.answer_source_path,
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
