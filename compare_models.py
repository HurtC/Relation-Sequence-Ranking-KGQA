#!/usr/bin/env python3
"""在同一开发集特征矩阵上比较 LR、随机森林、ExtraTrees 和 LambdaMART。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import graph_features_base as baseline  # noqa: E402
import graph_features as paironly  # noqa: E402
import train_ranker as common  # noqa: E402


FEATURE_MODES = (
    "full",
    "no_interaction",
    "interaction_anchor_removed",
    "all_question_anchor_removed",
)
MODEL_TYPES = (
    "logistic_regression",
    "extratrees",
    "randomforest",
    "lightgbm_regression",
    "lambdamart",
)


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


def candidate_rows(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return common.get_candidates(example)


def load_anchor_lookup(paths: Sequence[str]) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for path in paths:
        if not path:
            continue
        for index, row in enumerate(common.read_jsonl(path)):
            qid = common.get_qid(row, index)
            anchors = row.get("anchors", [])
            if isinstance(anchors, str):
                anchors = [anchors]
            lookup[qid] = [str(item) for item in anchors if str(item).strip()]
    return lookup


def anchors_for(example: Dict[str, Any], qid: str, lookup: Dict[str, List[str]]) -> List[str]:
    anchors = example.get("anchors", lookup.get(qid, []))
    if isinstance(anchors, str):
        anchors = [anchors]
    return [str(item) for item in anchors if str(item).strip()]


def remove_anchor_spans(question: str, anchors: Iterable[str]) -> str:
    result = str(question)
    for anchor in sorted(set(anchors), key=len, reverse=True):
        if not anchor:
            continue
        result = re.sub(re.escape(anchor), " ", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def text_and_numeric(
    hop: str,
    question: str,
    stripped_question: str,
    candidate: Dict[str, Any],
    mode: str,
) -> Tuple[str, List[float]]:
    if mode == "full":
        if hop == "1hop":
            return (
                baseline.candidate_text(question, candidate),
                paironly.numeric_feature_vector(question, candidate),
            )
        return (
            paironly.ranking_pair_text(question, candidate),
            paironly.numeric_feature_vector(question, candidate),
        )
    if mode == "no_interaction":
        return (
            baseline.candidate_text(question, candidate),
            paironly.numeric_feature_vector(question, candidate),
        )
    if mode == "interaction_anchor_removed":
        raw = baseline.candidate_text(question, candidate)
        local = paironly.coarse_pair_text(stripped_question, candidate)
        return (
            f"{raw} interactions: {local}",
            paironly.numeric_feature_vector(question, candidate),
        )
    if mode == "all_question_anchor_removed":
        if hop == "1hop":
            return (
                baseline.candidate_text(stripped_question, candidate),
                paironly.numeric_feature_vector(stripped_question, candidate),
            )
        return (
            paironly.ranking_pair_text(stripped_question, candidate),
            paironly.numeric_feature_vector(stripped_question, candidate),
        )
    raise ValueError(mode)


def flatten(
    hop: str,
    examples: Sequence[Dict[str, Any]],
    anchor_lookup: Dict[str, List[str]],
    mode: str,
) -> Tuple[List[str], List[List[float]], np.ndarray, np.ndarray, List[Dict[str, Any]], List[int], Dict[str, Any]]:
    texts: List[str] = []
    numbers: List[List[float]] = []
    y_f1: List[float] = []
    y_rel: List[int] = []
    metas: List[Dict[str, Any]] = []
    groups: List[int] = []
    missing_anchor_qids: List[str] = []
    changed_questions = 0

    for example_index, example in enumerate(examples):
        qid = common.get_qid(example, example_index)
        question = common.get_question(example)
        anchors = anchors_for(example, qid, anchor_lookup)
        stripped = remove_anchor_spans(question, anchors)
        if stripped != question:
            changed_questions += 1
        if mode in {"interaction_anchor_removed", "all_question_anchor_removed"} and not anchors:
            missing_anchor_qids.append(qid)
        candidates = candidate_rows(example)
        groups.append(len(candidates))
        for candidate_index, candidate in enumerate(candidates):
            text, numeric = text_and_numeric(hop, question, stripped, candidate, mode)
            f1 = common.get_answer_f1(candidate)
            texts.append(text)
            numbers.append(numeric)
            y_f1.append(f1)
            y_rel.append(common.relevance_from_f1(f1, "linear20"))
            metas.append(
                {
                    "qid": qid,
                    "cid": common.get_cid(candidate, candidate_index),
                    "candidate_index": candidate_index,
                }
            )

    anchor_audit = {
        "question_count": len(examples),
        "changed_question_count": changed_questions,
        "missing_anchor_count": len(missing_anchor_qids),
        "missing_anchor_qids_first20": missing_anchor_qids[:20],
    }
    return (
        texts,
        numbers,
        np.asarray(y_f1, dtype=np.float32),
        np.asarray(y_rel, dtype=np.int32),
        metas,
        groups,
        anchor_audit,
    )


def build_matrices(
    train_texts: List[str],
    train_numbers: List[List[float]],
    dev_texts: List[str],
    dev_numbers: List[List[float]],
    max_features: int,
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
    )
    train_text = vectorizer.fit_transform(train_texts)
    dev_text = vectorizer.transform(dev_texts)
    train_numeric = csr_matrix(np.asarray(train_numbers, dtype=np.float32))
    dev_numeric = csr_matrix(np.asarray(dev_numbers, dtype=np.float32))
    train_matrix = hstack([train_text, train_numeric]).tocsr()
    dev_matrix = hstack([dev_text, dev_numeric]).tocsr()
    return vectorizer, train_matrix, dev_matrix


def fit_model(
    model_type: str,
    train_matrix,
    dev_matrix,
    y_train_f1: np.ndarray,
    y_train_rel: np.ndarray,
    y_dev_f1: np.ndarray,
    y_dev_rel: np.ndarray,
    train_groups: List[int],
    dev_groups: List[int],
    trees: int,
    jobs: int,
    random_state: int,
    randomforest_max_features: Any,
):
    if model_type == "logistic_regression":
        model = LogisticRegression(
            solver="saga",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            n_jobs=jobs,
        )
        model.fit(
            train_matrix,
            (y_train_f1 > 0.0).astype(np.int32),
            sample_weight=1.0 + 4.0 * y_train_f1,
        )
        scores = model.predict_proba(dev_matrix)[:, 1]
        return model, np.asarray(scores, dtype=np.float64), None
    if model_type == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=jobs,
            random_state=random_state,
        )
        model.fit(train_matrix.tocsc(), y_train_f1)
        return model, np.asarray(model.predict(dev_matrix), dtype=np.float64), None
    if model_type == "randomforest":
        model = RandomForestRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=randomforest_max_features,
            bootstrap=True,
            n_jobs=jobs,
            random_state=random_state,
        )
        model.fit(train_matrix.tocsc(), y_train_f1)
        return model, np.asarray(model.predict(dev_matrix), dtype=np.float64), None
    if model_type == "lightgbm_regression":
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            metric="l2",
            boosting_type="gbdt",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=jobs,
            verbose=-1,
        )
        model.fit(
            train_matrix,
            y_train_f1,
            eval_set=[(dev_matrix, y_dev_f1)],
            callbacks=[lgb.log_evaluation(50), lgb.early_stopping(80, verbose=True)],
        )
        best = int(model.best_iteration_ or trees)
        return model, np.asarray(model.predict(dev_matrix, num_iteration=best), dtype=np.float64), best
    if model_type == "lambdamart":
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=jobs,
            verbose=-1,
            label_gain=list(range(21)),
        )
        model.fit(
            train_matrix,
            y_train_rel,
            group=train_groups,
            eval_set=[(dev_matrix, y_dev_rel)],
            eval_group=[dev_groups],
            eval_at=[1, 3, 5, 10],
            callbacks=[lgb.log_evaluation(50), lgb.early_stopping(80, verbose=True)],
        )
        best = int(model.best_iteration_ or trees)
        return model, np.asarray(model.predict(dev_matrix, num_iteration=best), dtype=np.float64), best
    raise ValueError(model_type)


def evaluate(
    examples: Sequence[Dict[str, Any]],
    metas: Sequence[Dict[str, Any]],
    scores: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, List[Dict[str, Any]]]:
    score_index = common.score_index_from_predictions(list(metas), scores)
    precision = recall = f1_sum = 0.0
    exact = hits = 0
    per_question_f1: List[float] = []
    selections: List[Dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        qid = common.get_qid(example, example_index)
        ranked = common.ranked_candidates(example, score_index.get(qid, {}))
        if ranked:
            cid, score, candidate = ranked[0]
            prediction = common.get_answer_ids(candidate)
            relations = common.get_relations(candidate)
        else:
            cid, score, prediction, relations = None, None, set(), []
        gold = common.get_gold_answers(example)
        p, r, item_f1, is_exact = common.compute_prf(prediction, gold)
        precision += p
        recall += r
        f1_sum += item_f1
        exact += int(is_exact)
        hits += int(bool(prediction & gold))
        per_question_f1.append(item_f1)
        selections.append(
            {
                "qid": qid,
                "cid": cid,
                "score": None if score is None else float(score),
                "relations": relations,
                "f1": item_f1,
                "exact": bool(is_exact),
                "hit": bool(prediction & gold),
            }
        )
    count = max(1, len(examples))
    summary = {
        "examples": len(examples),
        "avg_precision": precision / count,
        "avg_recall": recall / count,
        "avg_f1": f1_sum / count,
        "exact_rate": exact / count,
        "set_hits_at_1": hits / count,
    }
    return summary, np.asarray(per_question_f1, dtype=np.float64), selections


def paired_bootstrap(
    reference: np.ndarray,
    comparison: np.ndarray,
    random_state: int,
    replicates: int = 10000,
) -> Dict[str, float]:
    delta = reference - comparison
    rng = np.random.default_rng(random_state)
    values: List[np.ndarray] = []
    batch = 250
    for start in range(0, replicates, batch):
        current = min(batch, replicates - start)
        indices = rng.integers(0, len(delta), size=(current, len(delta)))
        values.append(delta[indices].mean(axis=1))
    samples = np.concatenate(values)
    return {
        "reference_minus_comparison": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_replicates": replicates,
    }


def run_mode(
    args,
    train_examples,
    dev_examples,
    anchor_lookup,
    mode: str,
    model_types: Sequence[str],
) -> Dict[str, Any]:
    print(f"\n=== feature mode: {mode} ===", flush=True)
    train_flat = flatten(args.hop, train_examples, anchor_lookup, mode)
    dev_flat = flatten(args.hop, dev_examples, anchor_lookup, mode)
    train_texts, train_nums, y_train_f1, y_train_rel, train_metas, train_groups, train_anchor_audit = train_flat
    dev_texts, dev_nums, y_dev_f1, y_dev_rel, dev_metas, dev_groups, dev_anchor_audit = dev_flat
    vectorizer, train_matrix, dev_matrix = build_matrices(
        train_texts, train_nums, dev_texts, dev_nums, args.max_features
    )
    report: Dict[str, Any] = {
        "feature_mode": mode,
        "train_matrix_shape": list(train_matrix.shape),
        "dev_matrix_shape": list(dev_matrix.shape),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_anchor_audit": train_anchor_audit,
        "dev_anchor_audit": dev_anchor_audit,
        "models": {},
    }
    f1_vectors: Dict[str, np.ndarray] = {}
    for model_type in model_types:
        print(f"--- train {model_type} ---", flush=True)
        started = time.time()
        model, scores, best_iteration = fit_model(
            model_type,
            train_matrix,
            dev_matrix,
            y_train_f1,
            y_train_rel,
            y_dev_f1,
            y_dev_rel,
            train_groups,
            dev_groups,
            args.trees,
            args.jobs,
            args.random_state,
            args.randomforest_max_features,
        )
        summary, question_f1, selections = evaluate(dev_examples, dev_metas, scores)
        result = {
            "summary": summary,
            "best_iteration": best_iteration,
            "elapsed_seconds": time.time() - started,
        }
        report["models"][model_type] = result
        f1_vectors[model_type] = question_f1
        write_json(
            os.path.join(args.output_dir, args.hop, mode, f"{model_type}_selections.json"),
            selections,
        )
        write_json(
            os.path.join(args.output_dir, args.hop, mode, f"{model_type}_summary.json"),
            result,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        del model, scores
        gc.collect()

    if "lambdamart" in f1_vectors:
        report["paired_bootstrap"] = {}
        for model_type, vector in f1_vectors.items():
            if model_type == "lambdamart":
                continue
            report["paired_bootstrap"][f"lambdamart_vs_{model_type}"] = paired_bootstrap(
                f1_vectors["lambdamart"], vector, args.random_state
            )
    del vectorizer, train_matrix, dev_matrix
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hop", required=True, choices=("1hop", "2hop", "3hop"))
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--anchor-source", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--randomforest-max-features",
        default="1.0",
        help="1.0 for all features or sqrt for the conventional random-subspace baseline.",
    )
    parser.add_argument(
        "--full-models",
        nargs="+",
        choices=MODEL_TYPES,
        default=list(MODEL_TYPES),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=FEATURE_MODES,
        default=list(FEATURE_MODES),
    )
    args = parser.parse_args()
    if args.randomforest_max_features != "sqrt":
        args.randomforest_max_features = float(args.randomforest_max_features)

    os.makedirs(os.path.join(args.output_dir, args.hop), exist_ok=True)
    train_examples = common.read_jsonl(args.train)
    dev_examples = common.read_jsonl(args.dev)
    anchor_lookup = load_anchor_lookup(args.anchor_source)
    model_family_modes = {
        "full": tuple(args.full_models),
        "no_interaction": ("lambdamart",),
        "interaction_anchor_removed": ("lambdamart",),
        "all_question_anchor_removed": ("lambdamart",),
    }
    report: Dict[str, Any] = {
        "protocol": "metaqa_model_comparison_dev",
        "hop": args.hop,
        "test_loaded": False,
        "train_path": args.train,
        "dev_path": args.dev,
        "train_sha256": sha256(args.train),
        "dev_sha256": sha256(args.dev),
        "anchor_sources": [
            {"path": path, "sha256": sha256(path)} for path in args.anchor_source
        ],
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "max_features": args.max_features,
        "tree_estimators": args.trees,
        "randomforest_max_features": args.randomforest_max_features,
        "random_state": args.random_state,
        "notes": [
            "All model families in full consume the identical sparse matrix.",
        ],
        "feature_modes": {},
    }
    for mode in args.modes:
        report["feature_modes"][mode] = run_mode(
            args,
            train_examples,
            dev_examples,
            anchor_lookup,
            mode,
            model_family_modes[mode],
        )
        write_json(os.path.join(args.output_dir, args.hop, "report.partial.json"), report)
    write_json(os.path.join(args.output_dir, args.hop, "report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
