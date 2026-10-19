#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""使用原始问题—路径文本和图统计特征训练 LambdaMART 组内排序器。"""

import argparse
import json
import math
import os
import pickle
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import lightgbm as lgb

from graph_features_base import (
    NUMERIC_FEATURE_NAMES,
    candidate_text as build_candidate_text,
    numeric_feature_vector,
)

RELEVANCE_LEVELS = {"linear20": 20}


# ============================================================
# IO
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx + 1} in {path}: {e}")
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_pickle(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ============================================================
# Basic helpers
# ============================================================

def to_set(xs: Any) -> Set[str]:
    if xs is None:
        return set()
    if isinstance(xs, set):
        return set(str(x) for x in xs)
    if isinstance(xs, list):
        return set(str(x) for x in xs)
    return {str(xs)}


def get_qid(ex: Dict[str, Any], idx: int = 0) -> str:
    return str(ex.get("qid", ex.get("id", f"example-{idx}")))


def get_question(ex: Dict[str, Any]) -> str:
    return str(ex.get("question", ""))


def get_gold_answers(ex: Dict[str, Any]) -> Set[str]:
    for key in ["gold_answers", "answers", "gold_answer", "gold_answer_ids"]:
        if key in ex:
            return to_set(ex[key])
    return set()


def get_candidates(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    cands = ex.get("candidates", ex.get("candidate_relation_sequences", []))
    if not isinstance(cands, list):
        return []
    return [c for c in cands if isinstance(c, dict)]


def get_cid(cand: Dict[str, Any], idx: int) -> str:
    return str(cand.get("cid", f"C{idx + 1:03d}"))


def get_relations(cand: Dict[str, Any]) -> List[str]:
    rels = cand.get("relations", cand.get("relation_sequence", cand.get("sequence", [])))
    if rels is None:
        return []
    if isinstance(rels, list):
        return [str(x) for x in rels]
    if isinstance(rels, str):
        if "->" in rels:
            return [x.strip() for x in rels.split("->") if x.strip()]
        if "|" in rels:
            return [x.strip() for x in rels.split("|") if x.strip()]
        return [rels.strip()]
    return []


def get_answer_ids(cand: Dict[str, Any]) -> Set[str]:
    for key in ["answer_ids", "answers", "pred_answers", "answer_entities"]:
        if key in cand:
            return to_set(cand[key])
    return set()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def get_answer_f1(cand: Dict[str, Any]) -> float:
    return safe_float(cand.get("answer_f1", 0.0))


def relevance_from_f1(f1: float, scheme: str) -> int:
    if scheme == "linear20":
        if f1 <= 0.0:
            return 0
        bins = RELEVANCE_LEVELS[scheme]
        return max(1, min(bins, int(math.floor(f1 * bins + 0.5))))
    raise ValueError(f"Unknown relevance scheme: {scheme}")


def candidate_text(ex: Dict[str, Any], cand: Dict[str, Any]) -> str:
    return build_candidate_text(str(ex.get("question", "")), cand)


def get_num_features(
    ex: Dict[str, Any],
    cand: Dict[str, Any],
) -> List[float]:
    return numeric_feature_vector(str(ex.get("question", "")), cand)


def compute_prf(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float, bool]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0, True
    if not pred:
        return 0.0, 0.0, 0.0, False
    if not gold:
        return 0.0, 0.0, 0.0, False

    inter = len(pred & gold)
    p = inter / max(1, len(pred))
    r = inter / max(1, len(gold))
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    exact = pred == gold
    return p, r, f1, exact


# ============================================================
# Flatten
# ============================================================

def flatten_examples(
    examples: List[Dict[str, Any]],
    relevance_scheme: str,
):
    texts = []
    nums = []
    rel_labels = []
    f1_labels = []
    metas = []
    group_sizes = []

    for ex_idx, ex in enumerate(examples):
        qid = get_qid(ex, ex_idx)
        candidates = get_candidates(ex)
        group_sizes.append(len(candidates))

        for cand_idx, cand in enumerate(candidates):
            cid = get_cid(cand, cand_idx)
            f1 = get_answer_f1(cand)

            texts.append(candidate_text(ex, cand))
            nums.append(get_num_features(ex, cand))
            rel_labels.append(relevance_from_f1(f1, relevance_scheme))
            f1_labels.append(f1)

            metas.append({
                "qid": qid,
                "cid": cid,
                "candidate_index": cand_idx,
                "answer_f1": f1,
                "relevance": relevance_from_f1(f1, relevance_scheme),
            })

    return (
        texts,
        nums,
        np.asarray(rel_labels, dtype=np.int32),
        np.asarray(f1_labels, dtype=np.float32),
        metas,
        group_sizes,
    )


# ============================================================
# Features
# ============================================================

def build_train_features(texts: List[str], nums: List[List[float]], max_features: int):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
    )

    X_text = vectorizer.fit_transform(texts)
    X_num = csr_matrix(np.asarray(nums, dtype=np.float32))
    X = hstack([X_text, X_num]).tocsr()
    return vectorizer, X


def build_features(vectorizer, texts: List[str], nums: List[List[float]]):
    X_text = vectorizer.transform(texts)
    X_num = csr_matrix(np.asarray(nums, dtype=np.float32))
    return hstack([X_text, X_num]).tocsr()


# ============================================================
# Prediction / Evaluation
# ============================================================

def score_index_from_predictions(metas: List[Dict[str, Any]], scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for meta, score in zip(metas, scores):
        qid = meta["qid"]
        cid = meta["cid"]
        out.setdefault(qid, {})[cid] = float(score)
    return out


def ranked_candidates(ex: Dict[str, Any], q_scores: Dict[str, float]):
    ranked = []
    for cand_idx, cand in enumerate(get_candidates(ex)):
        cid = get_cid(cand, cand_idx)
        score = q_scores.get(cid, -1e9)
        ranked.append((cid, score, cand))
    # 使用关系序列和候选编号处理分数并列。
    ranked.sort(
        key=lambda x: (
            -float(x[1]),
            tuple(get_relations(x[2])),
            x[0],
        )
    )
    return ranked


def select_from_ranked(
    ranked: List[Tuple[str, float, Dict[str, Any]]],
    mode: str,
    max_select: int,
    second_score_threshold: float,
    score_margin_threshold: float,
):
    if not ranked:
        return []

    if mode == "top1":
        return ranked[:1]

    if mode == "top2":
        return ranked[:max_select]

    if mode == "adaptive":
        selected = [ranked[0]]
        if max_select >= 2 and len(ranked) >= 2:
            top_score = ranked[0][1]
            second_score = ranked[1][1]
            margin = top_score - second_score
            if second_score >= second_score_threshold and margin <= score_margin_threshold:
                selected.append(ranked[1])
        return selected[:max_select]

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_selection(
    examples: List[Dict[str, Any]],
    score_index: Dict[str, Dict[str, float]],
    mode: str,
    max_select: int,
    second_score_threshold: float,
    score_margin_threshold: float,
    output_path: str,
    summary_path: str,
):
    rows = []

    total = 0
    total_p = 0.0
    total_r = 0.0
    total_f1 = 0.0
    exact_count = 0
    total_selected = 0
    empty_count = 0

    for ex_idx, ex in enumerate(examples):
        qid = get_qid(ex, ex_idx)
        gold = get_gold_answers(ex)

        ranked = ranked_candidates(ex, score_index.get(qid, {}))
        selected = select_from_ranked(
            ranked=ranked,
            mode=mode,
            max_select=max_select,
            second_score_threshold=second_score_threshold,
            score_margin_threshold=score_margin_threshold,
        )

        pred = set()
        selected_ids = []
        selected_scores = []

        for cid, score, cand in selected:
            selected_ids.append(cid)
            selected_scores.append(float(score))
            pred |= get_answer_ids(cand)

        if not selected_ids:
            empty_count += 1

        p, r, f1, exact = compute_prf(pred, gold)

        total += 1
        total_p += p
        total_r += r
        total_f1 += f1
        exact_count += int(exact)
        total_selected += len(selected_ids)

        rows.append({
            "qid": qid,
            "selection_mode": mode,
            "selected_ids": selected_ids,
            "selected_scores": selected_scores,
            "gold_answers": sorted(gold),
            "pred_answers": sorted(pred),
            "precision": p,
            "recall": r,
            "f1": f1,
            "exact": exact,
            "top_candidates": [
                {
                    "cid": cid,
                    "score": float(score),
                    "relations": get_relations(cand),
                    "answer_count": len(get_answer_ids(cand)),
                    "answer_f1_for_analysis_only": get_answer_f1(cand),
                }
                for cid, score, cand in ranked[:10]
            ],
        })

    summary = {
        "total_examples": total,
        "selection_mode": mode,
        "max_select": max_select,
        "second_score_threshold": second_score_threshold if mode == "adaptive" else None,
        "score_margin_threshold": score_margin_threshold if mode == "adaptive" else None,
        "avg_precision": total_p / max(1, total),
        "avg_recall": total_r / max(1, total),
        "avg_f1": total_f1 / max(1, total),
        "exact_count": exact_count,
        "exact_rate": exact_count / max(1, total),
        "avg_selected_count": total_selected / max(1, total),
        "empty_selection_count": empty_count,
        "empty_selection_rate": empty_count / max(1, total),
    }

    write_jsonl(output_path, rows)
    write_json(summary_path, summary)
    return summary


def tune_adaptive_thresholds(
    dev_examples: List[Dict[str, Any]],
    dev_score_index: Dict[str, Dict[str, float]],
    output_dir: str,
):
    top_scores = []
    second_scores = []
    margins = []

    for ex_idx, ex in enumerate(dev_examples):
        qid = get_qid(ex, ex_idx)
        ranked = ranked_candidates(ex, dev_score_index.get(qid, {}))
        if len(ranked) >= 2:
            top_scores.append(ranked[0][1])
            second_scores.append(ranked[1][1])
            margins.append(ranked[0][1] - ranked[1][1])

    if not second_scores:
        return 0.0, 0.0, None

    second_scores_np = np.asarray(second_scores, dtype=np.float32)
    margins_np = np.asarray(margins, dtype=np.float32)

    second_thresholds = sorted(set(float(x) for x in np.quantile(
        second_scores_np,
        [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    )))

    margin_thresholds = sorted(set(float(x) for x in np.quantile(
        margins_np,
        [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    )))

    best = None
    best_second = None
    best_margin = None

    tmp_dir = os.path.join(output_dir, "_tmp_adaptive_threshold_search")
    os.makedirs(tmp_dir, exist_ok=True)

    for second_t in second_thresholds:
        for margin_t in margin_thresholds:
            out_path = os.path.join(tmp_dir, f"dev_adaptive_s{second_t:.6f}_m{margin_t:.6f}.jsonl")
            sum_path = os.path.join(tmp_dir, f"dev_adaptive_s{second_t:.6f}_m{margin_t:.6f}.json")

            summary = evaluate_selection(
                examples=dev_examples,
                score_index=dev_score_index,
                mode="adaptive",
                max_select=2,
                second_score_threshold=second_t,
                score_margin_threshold=margin_t,
                output_path=out_path,
                summary_path=sum_path,
            )

            if best is None:
                best = summary
                best_second = second_t
                best_margin = margin_t
            else:
                if (
                    summary["avg_f1"] > best["avg_f1"]
                    or (
                        summary["avg_f1"] == best["avg_f1"]
                        and summary["exact_rate"] > best["exact_rate"]
                    )
                    or (
                        summary["avg_f1"] == best["avg_f1"]
                        and summary["exact_rate"] == best["exact_rate"]
                        and summary["avg_precision"] > best["avg_precision"]
                    )
                ):
                    best = summary
                    best_second = second_t
                    best_margin = margin_t

    return best_second, best_margin, best


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument(
        "--test",
        default=None,
        help="Optional held-out evaluation file.",
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument(
        "--relevance-scheme",
        choices=sorted(RELEVANCE_LEVELS),
        default="linear20",
        help="linear20 maps candidate answer-set F1 to 0-20 relevance levels.",
    )

    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=10)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--max-select", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Loading data...")
    print("=" * 80)

    train_examples = read_jsonl(args.train)
    dev_examples = read_jsonl(args.dev)
    test_examples = read_jsonl(args.test) if args.test else []

    print(f"train/dev/test examples: {len(train_examples)}/{len(dev_examples)}/{len(test_examples)}")

    print("=" * 80)
    print("Flatten examples...")
    print("=" * 80)

    train_texts, train_nums, y_train_rel, y_train_f1, train_metas, train_group = flatten_examples(
        train_examples,
        args.relevance_scheme,
    )
    dev_texts, dev_nums, y_dev_rel, y_dev_f1, dev_metas, dev_group = flatten_examples(
        dev_examples,
        args.relevance_scheme,
    )
    if test_examples:
        test_texts, test_nums, y_test_rel, y_test_f1, test_metas, test_group = flatten_examples(
            test_examples,
            args.relevance_scheme,
        )
    else:
        test_texts, test_nums, test_metas, test_group = [], [], [], []
        y_test_rel = np.asarray([], dtype=np.int32)
        y_test_f1 = np.asarray([], dtype=np.float32)

    print(f"train candidates: {len(y_train_rel)}")
    print(f"dev candidates: {len(y_dev_rel)}")
    print(f"test candidates: {len(y_test_rel)}")
    print(f"train group count: {len(train_group)}")
    print(f"dev group count: {len(dev_group)}")
    print(f"test group count: {len(test_group)}")
    print(f"train relevance counts: {dict(zip(*np.unique(y_train_rel, return_counts=True)))}")
    print(f"dev relevance counts: {dict(zip(*np.unique(y_dev_rel, return_counts=True)))}")

    print("=" * 80)
    print("Build features...")
    print("=" * 80)

    vectorizer, X_train = build_train_features(train_texts, train_nums, args.max_features)
    X_dev = build_features(vectorizer, dev_texts, dev_nums)
    X_test = build_features(vectorizer, test_texts, test_nums) if test_examples else None

    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"X_test shape: {X_test.shape if X_test is not None else None}")

    print("=" * 80)
    print("Train LGBMRanker / LambdaMART...")
    print("=" * 80)

    ranker_kwargs = dict(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
        n_jobs=-1,
        verbose=-1,
    )
    if args.relevance_scheme == "linear20":
        ranker_kwargs["label_gain"] = list(range(RELEVANCE_LEVELS["linear20"] + 1))
    ranker = lgb.LGBMRanker(**ranker_kwargs)

    ranker.fit(
        X_train,
        y_train_rel,
        group=train_group,
        eval_set=[(X_dev, y_dev_rel)],
        eval_group=[dev_group],
        eval_at=[1, 3, 5, 10],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=80, verbose=True),
        ],
    )

    print(f"best_iteration: {ranker.best_iteration_}")

    print("=" * 80)
    print("Predict dev/test scores...")
    print("=" * 80)

    dev_scores = ranker.predict(X_dev, num_iteration=ranker.best_iteration_)
    test_scores = (
        ranker.predict(X_test, num_iteration=ranker.best_iteration_)
        if X_test is not None
        else np.asarray([], dtype=np.float32)
    )

    dev_score_index = score_index_from_predictions(dev_metas, dev_scores)
    test_score_index = score_index_from_predictions(test_metas, test_scores)

    print("=" * 80)
    print("Evaluate dev...")
    print("=" * 80)

    dev_top1 = evaluate_selection(
        examples=dev_examples,
        score_index=dev_score_index,
        mode="top1",
        max_select=1,
        second_score_threshold=0.0,
        score_margin_threshold=0.0,
        output_path=os.path.join(args.output_dir, "dev_top1.eval.jsonl"),
        summary_path=os.path.join(args.output_dir, "dev_top1.summary.json"),
    )

    dev_top2 = evaluate_selection(
        examples=dev_examples,
        score_index=dev_score_index,
        mode="top2",
        max_select=2,
        second_score_threshold=0.0,
        score_margin_threshold=0.0,
        output_path=os.path.join(args.output_dir, "dev_top2.eval.jsonl"),
        summary_path=os.path.join(args.output_dir, "dev_top2.summary.json"),
    )

    best_second, best_margin, dev_adaptive = tune_adaptive_thresholds(
        dev_examples=dev_examples,
        dev_score_index=dev_score_index,
        output_dir=args.output_dir,
    )

    print("dev_top1:")
    print(json.dumps(dev_top1, ensure_ascii=False, indent=2))
    print("dev_top2:")
    print(json.dumps(dev_top2, ensure_ascii=False, indent=2))
    print(f"dev_adaptive best_second={best_second}, best_margin={best_margin}:")
    print(json.dumps(dev_adaptive, ensure_ascii=False, indent=2))

    test_top1 = None
    test_top2 = None
    test_adaptive = None
    if test_examples:
        print("=" * 80)
        print("Evaluate optional held-out test...")
        print("=" * 80)

        test_top1 = evaluate_selection(
            examples=test_examples,
            score_index=test_score_index,
            mode="top1",
            max_select=1,
            second_score_threshold=0.0,
            score_margin_threshold=0.0,
            output_path=os.path.join(args.output_dir, "test_top1.eval.jsonl"),
            summary_path=os.path.join(args.output_dir, "test_top1.summary.json"),
        )

        test_top2 = evaluate_selection(
            examples=test_examples,
            score_index=test_score_index,
            mode="top2",
            max_select=2,
            second_score_threshold=0.0,
            score_margin_threshold=0.0,
            output_path=os.path.join(args.output_dir, "test_top2.eval.jsonl"),
            summary_path=os.path.join(args.output_dir, "test_top2.summary.json"),
        )

        test_adaptive = evaluate_selection(
            examples=test_examples,
            score_index=test_score_index,
            mode="adaptive",
            max_select=2,
            second_score_threshold=float(best_second),
            score_margin_threshold=float(best_margin),
            output_path=os.path.join(args.output_dir, "test_adaptive.eval.jsonl"),
            summary_path=os.path.join(args.output_dir, "test_adaptive.summary.json"),
        )

    bundle = {
        "model": ranker,
        "vectorizer": vectorizer,
        "feature_version": "raw_question_path_graph12",
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "relevance_scheme": args.relevance_scheme,
        "args": vars(args),
        "best_second_score_threshold": float(best_second),
        "best_score_margin_threshold": float(best_margin),
    }

    model_path = os.path.join(args.output_dir, "lambdamart_path_ranker.pkl")
    save_pickle(model_path, bundle)

    final_summary = {
        "method": "lightgbm_lambdamart_path_ranker",
        "feature_version": "raw_question_path_graph12",
        "relevance_scheme": args.relevance_scheme,
        "model_path": model_path,
        "best_iteration": int(ranker.best_iteration_) if ranker.best_iteration_ is not None else None,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "test_examples": len(test_examples),
        "train_candidate_count": int(len(y_train_rel)),
        "dev_candidate_count": int(len(y_dev_rel)),
        "test_candidate_count": int(len(y_test_rel)),
        "train_relevance_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train_rel, return_counts=True))},
        "dev_relevance_counts": {str(k): int(v) for k, v in zip(*np.unique(y_dev_rel, return_counts=True))},
        "numeric_feature_count": 12,
        "adaptive_second_score_threshold": float(best_second),
        "adaptive_score_margin_threshold": float(best_margin),
        "dev_top1_summary": dev_top1,
        "dev_top2_summary": dev_top2,
        "dev_adaptive_summary": dev_adaptive,
        "test_top1_summary": test_top1,
        "test_top2_summary": test_top2,
        "test_adaptive_summary": test_adaptive,
    }

    write_json(os.path.join(args.output_dir, "final_summary.json"), final_summary)

    print("=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
