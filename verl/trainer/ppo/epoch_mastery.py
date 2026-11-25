"""Screen the current Student between two epochs; retain everything except K/K correct."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time

from evaluation.base_rollout_style import normalize, options_from_question
from evaluation.rollout_verifier import clear_option, visible_answer


def grade_answer(response, question, gold, truncated=False):
    """Conservative final-choice matching. Unknown is retained, never counted as correct."""
    if truncated:
        return None, "truncated"
    text = visible_answer(response).strip()
    if not text:
        return False, "empty_answer"
    declarations = re.findall(r"(?m)^\s*[\[(]?([A-D])(?:[\])\.：:]|\s*$)", text)
    if len(set(declarations)) > 1:
        return None, "conflicting_choices"
    options = options_from_question(question)
    if gold not in options:
        raise ValueError(f"Reference option {gold!r} is absent from the training question")
    # Do not treat an option mentioned in an explanation as a selected answer.
    # Whole answer, final line, final explicit answer, or final boxed span only.
    candidate = text.splitlines()[-1].strip()
    boxed = re.search(r"\\boxed\{([^{}]*)\}\s*[.$]*\s*$", text)
    if boxed:
        candidate = boxed[1]
    candidate = re.sub(
        r"^(?:the\s+)?(?:(?:correct|final)\s+)?(?:answer|option|choice)\s*(?:is\s*)?[:：]?\s*",
        "", candidate, flags=re.I,
    )
    candidate = normalize(candidate).casefold()
    matches = []
    for letter, option in options.items():
        forms = [letter, f"({letter})", f"[{letter}]", option]
        forms += [f"{letter}{sep}{option}" for sep in (". ", ") ", ": ", " ", " - ")]
        if candidate in {normalize(form).casefold() for form in forms}:
            matches.append(letter)
    if len(matches) != 1:
        return None, "unresolved_final_answer"
    # Conflicting explicit choices or an option list are not proof of mastery.
    declared = clear_option(text)
    if declared is not None and declared != matches[0]:
        return None, "conflicting_choices"
    return matches[0] == gold, "exact_final_choice"


def prompt_text(messages):
    parts = []
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(part["text"] for part in content if part["type"] == "text")
    return "\n".join(parts)


def classify_group(verdicts):
    correct = sum(value is True for value in verdicts)
    unresolved = sum(value is None for value in verdicts)
    category = ("unresolved" if unresolved else "all_correct" if correct == len(verdicts)
                else "all_wrong" if correct == 0 else "mixed")
    return dict(correct=correct, unresolved=unresolved, category=category,
                keep=category != "all_correct")


def screen_and_filter(trainer):
    """Use live rollout weights, then swap only the data loader. No optimizer/EMA update."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset, SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    started = time.monotonic()
    completed_step = trainer.global_steps - 1
    cfg = trainer.config.trainer.epoch_mastery
    k = int(cfg.rollout_n)
    source = trainer.train_dataset
    old_loader = trainer.train_dataloader
    batch_size = old_loader.batch_size
    directory = Path(trainer.config.trainer.default_local_dir) / "epoch_mastery" / f"step{completed_step}"
    directory.mkdir(parents=True, exist_ok=True)
    screen_batch_size = batch_size * trainer.config.actor_rollout_ref.rollout.n // k
    loader = DataLoader(source, batch_size=screen_batch_size, shuffle=False, drop_last=False,
                        num_workers=0, collate_fn=old_loader.collate_fn)
    divisor = (trainer.config.actor_rollout_ref.rollout.agent.num_workers if trainer.async_rollout_mode
               else trainer.actor_rollout_wg.world_size)
    max_tokens = trainer.config.actor_rollout_ref.rollout.response_length
    eos = trainer.tokenizer.eos_token_id
    kept, summaries = [], []
    offset = 0
    print(f"[epoch-mastery] start current_student_step={completed_step} questions={len(source)} K={k}", flush=True)
    with (directory / "rollouts.jsonl").open("w") as journal:
        for data in loader:
            batch = DataProto.from_single_dict(data)
            questions = [prompt_text(x) for x in batch.non_tensor_batch["raw_prompt"]]
            golds = [str(x["ground_truth"]).strip().upper() for x in batch.non_tensor_batch["reward_model"]]
            count = len(batch)
            batch.non_tensor_batch["uid"] = np.array(
                [f"mastery-step{completed_step}-q{offset+i}" for i in range(count)], dtype=object)
            gen = trainer._get_gen_batch(batch).repeat(repeat_times=k, interleave=True)
            # validate=False deliberately keeps the training prompt and stochastic T=1,
            # rather than val's greedy decoding. No degraded-view metadata is added.
            gen.meta_info.update(global_steps=completed_step, validate=False, do_sample=True,
                                 temperature=trainer.config.actor_rollout_ref.rollout.temperature,
                                 recompute_log_prob=False, eos_token_id=eos,
                                 pad_token_id=trainer.tokenizer.pad_token_id)
            gen, padding = pad_dataproto_to_divisor(gen, divisor)
            engine = trainer.async_rollout_manager if trainer.async_rollout_mode else trainer.actor_rollout_wg
            output = unpad_dataproto(engine.generate_sequences(gen), pad_size=padding)
            if len(output) != count * k:
                raise ValueError("Epoch screening did not return all requested trajectories")
            ids, masks = output.batch["responses"], output.batch["response_mask"]
            for i in range(count):
                verdicts = []
                for j in range(k):
                    tokens = ids[i*k+j][masks[i*k+j].bool()].tolist()
                    response = trainer.tokenizer.decode(tokens, skip_special_tokens=True)
                    truncated = len(tokens) >= max_tokens and tokens[-1] != eos
                    correct, reason = grade_answer(response, questions[i], golds[i], truncated)
                    verdicts.append(correct)
                    journal.write(json.dumps(dict(question_id=offset+i, rollout_index=j,
                        model_step=completed_step, question=questions[i], ground_truth=golds[i],
                        response=response, response_tokens=len(tokens), truncated=truncated,
                        correct=correct, judge_source=reason), ensure_ascii=False)+"\n")
                summary = dict(question_id=offset+i, **classify_group(verdicts))
                summaries.append(summary)
                if summary["keep"]:
                    kept.append(offset+i)
            offset += count
            journal.flush()
            print(f"[epoch-mastery] screened={offset}/{len(source)} responses={offset*k} kept={len(kept)}", flush=True)

    (directory / "question_summary.jsonl").write_text(
        "".join(json.dumps(row)+"\n" for row in summaries))
    (directory / "kept_question_ids.json").write_text(json.dumps(kept)+"\n")
    # Keep the source parquet intact, including crop/caption/bbox columns.
    source.dataframe.select(kept).to_parquet(str(directory / "train.parquet"))
    steps = len(kept) // batch_size
    manifest = dict(model_step=completed_step, questions=len(source), k=k, responses=offset*k,
        kept=len(kept), excluded_all_correct=len(source)-len(kept),
        categories=dict(Counter(row["category"] for row in summaries)),
        unresolved_responses=sum(row["unresolved"] for row in summaries),
        correct_histogram=dict(sorted(Counter(row["correct"] for row in summaries).items())),
        second_epoch_steps=steps, total_training_steps=completed_step+steps,
        dropped_last_rows=len(kept)%batch_size, train_file=str(directory / "train.parquet"),
        seconds=time.monotonic()-started, screening_prompt_batch=screen_batch_size,
        selection="current_student_after_epoch1", view="original",
        temperature=trainer.config.actor_rollout_ref.rollout.temperature,
        criterion="exclude only K/K confirmed correct; retain all-wrong and unresolved",
        state_continuity="same live Student, EMA Teacher, optimizer and LR scheduler")
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    if steps:
        subset = Subset(source, kept)
        sampler = (RandomSampler(subset, generator=old_loader.sampler.generator)
                   if trainer.config.data.shuffle else SequentialSampler(subset))
        trainer.train_dataset = subset
        trainer.train_dataloader = StatefulDataLoader(subset, batch_size=batch_size, sampler=sampler,
            drop_last=True, num_workers=0, collate_fn=old_loader.collate_fn)
    trainer.total_training_steps = completed_step + steps
    print("[epoch-mastery-complete] " + json.dumps(manifest), flush=True)
    return manifest
