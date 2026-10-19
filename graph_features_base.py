#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""提取原始问题、候选关系序列和图执行统计特征。"""

import math
import re
from collections import Counter
from typing import Any, Dict, List


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def get_relations(candidate: Dict[str, Any]) -> List[str]:
    value = candidate.get(
        "relations",
        candidate.get("relation_sequence", candidate.get("sequence", [])),
    )
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        if "->" in value:
            return [item.strip() for item in value.split("->") if item.strip()]
        if "|" in value:
            return [item.strip() for item in value.split("|") if item.strip()]
        return [value.strip()] if value.strip() else []
    return []


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"__rev__", " reverse ", text)
    text = re.sub(r"[_\-:/\\\.]+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def simple_stem_token(token: str) -> str:
    token = token.lower()
    for suffix in ("ies", "ing", "ers", "ors", "es", "s", "ed"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            if suffix == "ies":
                return token[:-3] + "y"
            return token[:-len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [simple_stem_token(token) for token in normalized.split() if token]


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    overlap = sum(
        min(count, right_counts.get(token, 0))
        for token, count in left_counts.items()
    )
    precision = overlap / max(1, sum(right_counts.values()))
    recall = overlap / max(1, sum(left_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def textualize_relation(relation: str) -> str:
    relation = str(relation)
    if relation.startswith("__rev__"):
        return "reverse " + normalize_text(relation[len("__rev__"):])
    return normalize_text(relation)


def build_path_text(candidate: Dict[str, Any]) -> str:
    value = candidate.get("path_text")
    if value:
        return str(value)
    return " followed by ".join(
        textualize_relation(relation) for relation in get_relations(candidate)
    )


def get_answer_count(candidate: Dict[str, Any]) -> float:
    if "answer_count" in candidate:
        return max(0.0, safe_float(candidate.get("answer_count")))
    value = candidate.get("answer_ids", candidate.get("answers", []))
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    return 0.0 if value is None else 1.0


def get_path_count(candidate: Dict[str, Any]) -> float:
    for key in ("path_count", "sample_path_count", "support_count", "instance_count"):
        if key in candidate:
            return max(0.0, safe_float(candidate.get(key)))
    return get_answer_count(candidate)


def derive_graph_features(question: str, candidate: Dict[str, Any]) -> Dict[str, float]:
    """返回十二项路径统计特征。"""
    relations = get_relations(candidate)
    answer_count = get_answer_count(candidate)
    path_count = get_path_count(candidate)
    hop = max(0.0, safe_float(candidate.get("hop", len(relations))))
    semantic = lexical_similarity(question, build_path_text(candidate))
    support_log = math.log1p(path_count)
    answer_log = math.log1p(answer_count)
    density = support_log / math.log1p(answer_count + 1.0)
    compact = -100.0 if answer_count <= 0 else -abs(
        math.log1p(answer_count) - math.log(6.0)
    )
    balanced = semantic + 0.20 * support_log + 0.25 * density + 0.10 * compact
    reverse_count = sum(
        1 for relation in relations if str(relation).startswith("__rev__")
    )
    unique_count = len(set(relations))
    repeat_count = max(0, len(relations) - unique_count)
    return {
        "answer_count_log": answer_log,
        "path_count_log": support_log,
        "hop": hop,
        "reverse_relation_count": float(reverse_count),
        "unique_relation_count": float(unique_count),
        "repeated_relation_count": float(repeat_count),
        "semantic_overlap": semantic,
        "support_log": support_log,
        "answer_scale_log": answer_log,
        "support_density": density,
        "answer_scale_compactness": compact,
        "multiview_balanced_score": balanced,
    }


NUMERIC_FEATURE_NAMES = [
    "answer_count_log",
    "path_count_log",
    "hop",
    "reverse_relation_count",
    "unique_relation_count",
    "repeated_relation_count",
    "semantic_overlap",
    "support_log",
    "answer_scale_log",
    "support_density",
    "answer_scale_compactness",
    "multiview_balanced_score",
]


def numeric_feature_vector(question: str, candidate: Dict[str, Any]) -> List[float]:
    values = derive_graph_features(question, candidate)
    return [float(values[name]) for name in NUMERIC_FEATURE_NAMES]


def candidate_text(question: str, candidate: Dict[str, Any]) -> str:
    relations = " ".join(get_relations(candidate))
    path_text = build_path_text(candidate)
    return f"question: {question} path: {path_text} relations: {relations}"


_GENERIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "in", "is", "it",
    "of", "on", "or", "that", "the", "these", "this", "those", "to",
    "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "with",
}


def coarse_pair_text(question: str, candidate: Dict[str, Any]) -> str:
    """构造问题词与带位置、方向关系词的稀疏交叉表示。"""
    question_tokens = []
    seen_question = set()
    for token in tokenize(question):
        if token in _GENERIC_STOPWORDS or len(token) < 2 or token in seen_question:
            continue
        seen_question.add(token)
        question_tokens.append(token)

    relation_atoms = []
    for index, relation in enumerate(get_relations(candidate), 1):
        reverse = str(relation).startswith("__rev__")
        base = str(relation)[len("__rev__"):] if reverse else str(relation)
        base = normalize_text(base).replace(" ", "_") or "empty"
        direction = "rev" if reverse else "fwd"
        relation_atoms.append(f"p{index}_{direction}_{base}")

    tokens = [f"qw_{token}" for token in question_tokens]
    tokens.extend(f"ra_{atom}" for atom in relation_atoms)
    tokens.extend(
        f"qr_{question_token}__{relation_atom}"
        for question_token in question_tokens
        for relation_atom in relation_atoms
    )
    return " ".join(tokens)
