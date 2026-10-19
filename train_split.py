#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""训练用于候选池 Top-K 截断的逻辑回归粗排器。"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from graph_features_base import (
    coarse_pair_text,
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


def candidates(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = example.get("candidates", example.get("candidate_relation_sequences", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def flatten(examples: List[Dict[str, Any]], with_labels: bool):
    texts: List[str] = []
    numeric: List[List[float]] = []
    labels: List[int] = []
    f1_values: List[float] = []
    groups: List[int] = []
    for example in examples:
        question = str(example.get("question", ""))
        group = candidates(example)
        groups.append(len(group))
        for candidate in group:
            texts.append(coarse_pair_text(question, candidate))
            numeric.append(numeric_feature_vector(question, candidate))
            if with_labels:
                f1 = float(candidate.get("answer_f1", 0.0) or 0.0)
                labels.append(int(f1 > 0.0))
                f1_values.append(f1)
    return texts, numeric, np.asarray(labels), np.asarray(f1_values), groups


def oracle_at_k(examples, scores, groups, k: int):
    offset = 0
    total_f1 = 0.0
    exact = 0
    top1_f1 = 0.0
    for example, size in zip(examples, groups):
        group = candidates(example)
        group_scores = scores[offset:offset + size]
        offset += size
        ranked = sorted(
            zip(group, group_scores),
            key=lambda item: (-float(item[1]), tuple(get_relations(item[0]))),
        )
        top = ranked[:k]
        best = max((float(item[0].get("answer_f1", 0.0) or 0.0) for item in top), default=0.0)
        total_f1 += best
        exact += int(abs(best - 1.0) < 1e-12)
        top1_f1 += float(ranked[0][0].get("answer_f1", 0.0) or 0.0) if ranked else 0.0
    count = max(1, len(examples))
    return {
        "examples": len(examples),
        "top_k": k,
        "oracle_set_f1": total_f1 / count,
        "oracle_exact": exact / count,
        "top1_candidate_f1": top1_f1 / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_examples = read_jsonl(args.train)
    dev_examples = read_jsonl(args.dev)
    train_text, train_num, y_train, train_f1, train_groups = flatten(train_examples, True)
    dev_text, dev_num, y_dev, dev_f1, dev_groups = flatten(dev_examples, True)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
        ngram_range=(1, 1),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    train_text_matrix = vectorizer.fit_transform(train_text)
    dev_text_matrix = vectorizer.transform(dev_text)
    x_train = hstack([
        train_text_matrix,
        csr_matrix(np.asarray(train_num, dtype=np.float32)),
    ]).tocsr()
    x_dev = hstack([
        dev_text_matrix,
        csr_matrix(np.asarray(dev_num, dtype=np.float32)),
    ]).tocsr()
    sample_weight = np.where(y_train > 0, 1.0 + 4.0 * train_f1, 1.0)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=args.alpha,
        l1_ratio=0.05,
        max_iter=args.max_iter,
        tol=1e-4,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
        average=True,
    )
    classifier.fit(x_train, y_train, sample_weight=sample_weight)
    train_scores = classifier.decision_function(x_train)
    dev_scores = classifier.decision_function(x_dev)
    train_summary = oracle_at_k(train_examples, train_scores, train_groups, args.top_k)
    dev_summary = oracle_at_k(dev_examples, dev_scores, dev_groups, args.top_k)

    bundle = {
        "model": classifier,
        "vectorizer": vectorizer,
        "feature_version": "word_relation_cross_graph12",
        "args": vars(args),
    }
    model_path = os.path.join(args.output_dir, "coarse_logreg.pkl")
    with open(model_path, "wb") as handle:
        pickle.dump(bundle, handle)
    summary = {
        "method": "sgd_logistic_coarse_selector",
        "model_path": model_path,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "train_candidates": int(len(y_train)),
        "dev_candidates": int(len(y_dev)),
        "text_feature_count": int(len(vectorizer.vocabulary_)),
        "positive_train_candidates": int(y_train.sum()),
        "positive_dev_candidates": int(y_dev.sum()),
        "train": train_summary,
        "dev": dev_summary,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
