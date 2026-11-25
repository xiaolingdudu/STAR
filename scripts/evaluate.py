"""Run the original six benchmarks, optionally MMVP, through local servers.

Keep all valid answers and judgments. A sample or benchmark failure does not
prevent later benchmarks from running. An incomplete run never receives Avg6.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.vision_opd_official.audit_results import audit, pending_answers

BENCHMARKS = {
    "vstar": ("vstar.json", 191),
    "zoombench": ("zoombench.json", 845),
    "hrbench-4k": ("hr_bench_4k.json", 800),
    "hrbench-8k": ("hr_bench_8k.json", 800),
    "mme-realworld": ("MME_RealWorld.json", 23609),
    "mme-realworld-cn": ("MME_RealWorld_CN.json", 5917),
    "mmvp": ("mmvp.json", 300),
}


def stage_inputs(source, target, benchmark, expected):
    rows = json.loads(source.read_text())
    if len(rows) != expected:
        raise ValueError(f"{benchmark}: expected {expected} questions, found {len(rows)}")
    for i, row in enumerate(rows):
        row.setdefault("sample_uid", f"{benchmark}:row:{i}")
        row.setdefault("sample_id", row["sample_uid"])
        row["images"] = [str((source.parent / p).resolve()) if not Path(p).is_absolute() else p
                         for p in row["images"]]
    if target.exists() and json.loads(target.read_text()) != rows:
        raise ValueError("Inputs changed in an existing run; choose a new output directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False) + "\n")


def mmvp_pairs(judges):
    if {int(row["index"]) for row in judges} != set(range(1, 301)):
        raise ValueError("MMVP requires the original question indices 1..300.")
    pairs = {}
    for row in judges:
        pair = (int(row["index"]) - 1) // 2
        pairs.setdefault(pair, []).append(row["judge"] == "Yes")
    if len(pairs) != 150 or any(len(v) != 2 for v in pairs.values()):
        raise ValueError("MMVP must contain all 150 original pairs (indices 1..300).")
    return 100 * sum(all(v) for v in pairs.values()) / 150


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared benchmark JSONs and local image paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="STAR", help="Student server's served model name")
    parser.add_argument("--student-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-url", default="http://127.0.0.1:8900/v1")
    parser.add_argument("--phase", choices=["answers", "judge", "all"], default="all")
    parser.add_argument("--mmvp", action="store_true")
    parser.add_argument("--student-concurrency", type=int, default=512)
    parser.add_argument("--judge-concurrency", type=int, default=128)
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = list(BENCHMARKS) if args.mmvp else list(BENCHMARKS)[:6]
    code = ROOT / "evaluation/vision_opd_official"
    env = dict(os.environ, EVAL_DEFER_SAMPLE_ERRORS="1", PYTHONPATH=str(ROOT))
    report = {"phase": args.phase, "benchmarks": {}, "complete": False}
    for benchmark in selected:
        filename, count = BENCHMARKS[benchmark]
        staged = args.output / "inputs" / filename
        answers = args.output / "model_answer" / benchmark / "STAR_seed42_answer.jsonl"
        judges = args.output / "judge" / benchmark / "STAR_seed42_answer.jsonl"
        try:
            stage_inputs(args.data / filename, staged, benchmark, count)
            if args.phase != "judge":
                subprocess.run([sys.executable, str(code / "infer.py"),
                    "--benchmark", benchmark, "--benchmark_json", str(staged),
                    "--out_dir", str(args.output / "model_answer"),
                    "--model_name", "STAR_seed42", "--model_id", args.model,
                    "--api_base", args.student_url, "--seed", "42", "--max_tokens", "8192",
                    "--enable_thinking", "False", "--parallel_workers", str(args.student_concurrency)],
                    cwd=args.output, env=env, check=True)
            pending = pending_answers(benchmark, staged, answers)
            if args.phase != "answers":
                subprocess.run([sys.executable, str(code / "judge_qwenlm.py"),
                    "--benchmark", benchmark, "--model", "STAR_seed42",
                    "--api_base", args.judge_url, "--judge_model", "openai/gpt-oss-120b",
                    "--judge_max_tokens", "2048", "--parallel_workers", str(args.judge_concurrency)],
                    cwd=args.output, env=env, check=True)
            if pending["pending"]:
                report["benchmarks"][benchmark] = pending
            else:
                result = audit(benchmark=benchmark, benchmark_json=staged, answer_json=answers,
                    judge_json=judges if args.phase != "answers" else None,
                    answer_source="generated", answer_source_path=str(answers))
                if benchmark == "mmvp" and args.phase != "answers":
                    result["pair_accuracy"] = mmvp_pairs(json.loads(judges.read_text()))
                report["benchmarks"][benchmark] = result
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            report["benchmarks"][benchmark] = {"error": str(exc)}
            print(f"{benchmark}: {exc}; continuing with the remaining benchmarks.", flush=True)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = report["benchmarks"]
    if all("accuracy" in rows[b] for b in selected):
        report["complete"] = True
        report["Avg6"] = sum(rows[b]["accuracy"] for b in list(BENCHMARKS)[:6]) / 6
        if args.mmvp:
            report["Avg7"] = (6 * report["Avg6"] + rows["mmvp"]["pair_accuracy"]) / 7
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.phase == "answers":
        ok = all(rows[b].get("answer_samples") == BENCHMARKS[b][1] for b in selected)
    else:
        ok = report["complete"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
