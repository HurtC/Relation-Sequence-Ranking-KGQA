#!/usr/bin/env python3
"""在开发集上运行 Full、Text-only 和 Numeric-only 特征消融。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import compare_models as strict  # noqa: E402


SETTINGS = ("full", "text_only", "numeric_only")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hop", required=True, choices=("1hop", "2hop", "3hop"))
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    started_all = time.time()
    train_examples = strict.common.read_jsonl(args.train)
    dev_examples = strict.common.read_jsonl(args.dev)
    train_flat = strict.flatten(args.hop, train_examples, {}, "full")
    dev_flat = strict.flatten(args.hop, dev_examples, {}, "full")
    (
        train_texts,
        train_numbers,
        y_train_f1,
        y_train_rel,
        train_metas,
        train_groups,
        _,
    ) = train_flat
    (
        dev_texts,
        dev_numbers,
        y_dev_f1,
        y_dev_rel,
        dev_metas,
        dev_groups,
        _,
    ) = dev_flat

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=1,
    )
    train_text = vectorizer.fit_transform(train_texts).tocsr()
    dev_text = vectorizer.transform(dev_texts).tocsr()
    train_numeric = csr_matrix(np.asarray(train_numbers, dtype=np.float32))
    dev_numeric = csr_matrix(np.asarray(dev_numbers, dtype=np.float32))
    matrices = {
        "full": (
            hstack([train_text, train_numeric]).tocsr(),
            hstack([dev_text, dev_numeric]).tocsr(),
        ),
        "text_only": (train_text, dev_text),
        "numeric_only": (train_numeric, dev_numeric),
    }

    report: Dict[str, Any] = {
        "protocol": "metaqa_feature_ablation_dev",
        "hop": args.hop,
        "test_loaded": False,
        "train_path": args.train,
        "dev_path": args.dev,
        "train_sha256": sha256(args.train),
        "dev_sha256": sha256(args.dev),
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "train_candidates": len(train_metas),
        "dev_candidates": len(dev_metas),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "numeric_feature_count": train_numeric.shape[1],
        "max_features": args.max_features,
        "random_state": args.random_state,
        "settings": {},
        "notes": [
            "Full combines the text representation with 12 graph features.",
            "Text-only keeps the text and local interaction representation.",
            "Numeric-only removes all TF-IDF and local word-relation interaction features.",
        ],
    }

    for setting in SETTINGS:
        train_matrix, dev_matrix = matrices[setting]
        print(
            f"=== {args.hop} {setting}: train={train_matrix.shape} dev={dev_matrix.shape} ===",
            flush=True,
        )
        started = time.time()
        model, scores, best_iteration = strict.fit_model(
            "lambdamart",
            train_matrix,
            dev_matrix,
            y_train_f1,
            y_train_rel,
            y_dev_f1,
            y_dev_rel,
            train_groups,
            dev_groups,
            1000,
            args.jobs,
            args.random_state,
            1.0,
        )
        summary, question_f1, selections = strict.evaluate(dev_examples, dev_metas, scores)
        result = {
            "feature_setting": setting,
            "train_matrix_shape": list(train_matrix.shape),
            "dev_matrix_shape": list(dev_matrix.shape),
            "best_iteration": best_iteration,
            "elapsed_seconds": time.time() - started,
            "summary": summary,
        }
        report["settings"][setting] = result
        write_json(os.path.join(args.output_dir, args.hop, f"{setting}_summary.json"), result)
        write_json(os.path.join(args.output_dir, args.hop, f"{setting}_selections.json"), selections)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        del model, scores, question_f1
        gc.collect()

    report["elapsed_seconds"] = time.time() - started_all
    write_json(os.path.join(args.output_dir, args.hop, "report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
