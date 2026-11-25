"""Extract explicit MCQ answers without treating the English article 'a' as A."""

import re

SCORER_VERSION = "routing_explicit_v1"


def normalize_answer(text):
    value = str(text).rsplit("</think>", 1)[-1].strip()
    return re.sub(r"\*\*|__|`", "", value).strip()


def extract_routing_option(text):
    value = normalize_answer(text)
    patterns = (
        ("answer_tag", r"<answer>\s*[\(\[]?([A-F])[\)\]]?\s*</answer>"),
        ("boxed", r"\\boxed\s*\{\s*([A-F])\s*\}"),
        ("answer_marker", r"(?i:final\s+answer|correct\s+answer|answer|答案)\s*(?::|：|(?i:is)\b)\s*(?:(?i:option)\s+)?[\(\[]?([A-F])\b"),
        ("letter_only", r"^\s*[\(\[]?([A-Fa-f])[\)\].:：]?\s*$"),
        ("leading_label", r"^\s*[\(\[]?([A-F])[\)\].:：]\s*"),
        ("option_assertion", r"^\s*(?i:option)\s+([A-F])\b"),
    )
    for reason, pattern in patterns:
        matches = re.findall(pattern, value, flags=re.MULTILINE)
        if matches:
            return matches[-1].upper(), reason
    return "", "unresolved"
