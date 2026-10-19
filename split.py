#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""使用粗排模型为候选关系序列打分并保留 Top-K 候选。"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack

from graph_features_base import (
    build_path_text,
    candidate_text,
    coarse_pair_text,
    derive_graph_features,
    get_relations,
    numeric_feature_vector,
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}")
    return rows


def get_candidates(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = example.get("candidates", example.get("candidate_relation_sequences", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def relation_key(candidate: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(get_relations(candidate))


def build_matrix(bundle: Dict[str, Any], examples: List[Dict[str, Any]]):
    texts: List[str] = []
    numeric: List[List[float]] = []
    groups: List[int] = []
    use_pair_features = str(bundle.get("feature_version", "")).startswith(
        ("word_relation_cross_graph12", "raw_word_relation_cross_graph_notemplate")
    )
    for example in examples:
        question = str(example.get("question", ""))
        candidates = get_candidates(example)
        groups.append(len(candidates))
        for candidate in candidates:
            texts.append(
                coarse_pair_text(question, candidate)
                if use_pair_features
                else candidate_text(question, candidate)
            )
            numeric.append(numeric_feature_vector(question, candidate))
    text_matrix = bundle["vectorizer"].transform(texts)
    numeric_matrix = csr_matrix(np.asarray(numeric, dtype=np.float32))
    return hstack([text_matrix, numeric_matrix]).tocsr(), groups


def output_candidate(
    candidate: Dict[str, Any],
    cid: str,
    question: str,
    score: float,
) -> Dict[str, Any]:
    # Preserve labels only as metadata for training/evaluation.  Selection and
    # features above have already been computed without reading these fields.
    output = dict(candidate)
    output["cid"] = cid
    output["relations"] = get_relations(candidate)
    output["path_text"] = build_path_text(candidate)
    output["coarse_features"] = derive_graph_features(question, candidate)
    output["coarse_model_score"] = float(score)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    with open(args.model_path, "rb") as handle:
        bundle = pickle.load(handle)
    examples = read_jsonl(args.input)
    matrix, groups = build_matrix(bundle, examples)
    if hasattr(bundle["model"], "decision_function"):
        scores = bundle["model"].decision_function(matrix)
        selector = "logistic_topk_split"
    else:
        scores = bundle["model"].predict(
            matrix,
            num_iteration=bundle["model"].best_iteration_,
        )
        selector = "lambdamart_topk_split"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total_raw = 0
    total_kept = 0
    offset = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for index, (example, group_size) in enumerate(zip(examples, groups)):
            question = str(example.get("question", ""))
            candidates = get_candidates(example)
            group_scores = scores[offset:offset + group_size]
            offset += group_size
            ranked = sorted(
                zip(candidates, group_scores),
                key=lambda item: (-float(item[1]), relation_key(item[0])),
            )
            selected = ranked[:args.top_k]
            output = {
                "qid": str(example.get("qid", example.get("id", f"example-{index}"))),
                "question": question,
                "gold_answers": example.get("gold_answers", example.get("answers", [])),
                "candidates": [
                    output_candidate(candidate, f"C{rank + 1:03d}", question, float(score))
                    for rank, (candidate, score) in enumerate(selected)
                ],
                "meta": {
                    "raw_candidate_count": len(candidates),
                    "kept_candidate_count": len(selected),
                    "top_k": args.top_k,
                    "selector": selector,
                },
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            total_raw += len(candidates)
            total_kept += len(selected)
            if (index + 1) % 1000 == 0:
                print(f"[{index + 1}/{len(examples)}]", flush=True)

    summary = {
        "input": args.input,
        "output": args.output,
        "model_path": args.model_path,
        "selector": selector,
        "examples": len(examples),
        "top_k": args.top_k,
        "avg_raw_candidate_count": total_raw / max(1, len(examples)),
        "avg_kept_candidate_count": total_kept / max(1, len(examples)),
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
