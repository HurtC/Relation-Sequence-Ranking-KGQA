#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""载入交叉特征 LambdaMART 模型，完成候选排序和答案集合评测。"""

import argparse
import json
import math
import os
import pickle
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix

from graph_features import (
    numeric_feature_vector,
    ranking_pair_text,
)

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


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Helpers
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


def candidate_text(ex: Dict[str, Any], cand: Dict[str, Any]) -> str:
    return ranking_pair_text(str(ex.get("question", "")), cand)


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
# Feature building
# ============================================================

def flatten_examples(examples: List[Dict[str, Any]]):
    texts = []
    nums = []
    metas = []

    for ex_idx, ex in enumerate(examples):
        qid = get_qid(ex, ex_idx)
        for cand_idx, cand in enumerate(get_candidates(ex)):
            cid = get_cid(cand, cand_idx)

            texts.append(candidate_text(ex, cand))
            nums.append(get_num_features(ex, cand))

            metas.append({
                "qid": qid,
                "cid": cid,
                "candidate_index": cand_idx,
                "answer_f1": get_answer_f1(cand),
            })

    return texts, nums, metas


def build_features(vectorizer, texts: List[str], nums: List[List[float]]):
    X_text = vectorizer.transform(texts)
    X_num = csr_matrix(np.asarray(nums, dtype=np.float32))
    return hstack([X_text, X_num]).tocsr()


def score_index_from_predictions(metas: List[Dict[str, Any]], scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for meta, score in zip(metas, scores):
        qid = meta["qid"]
        cid = meta["cid"]
        out.setdefault(qid, {})[cid] = float(score)
    return out


# ============================================================
# Selection / Evaluation
# ============================================================

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
    hit_count = 0
    total_selected = 0
    empty_count = 0

    total_gold_count = 0
    total_pred_count = 0
    total_overlap_count = 0

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
        overlap = pred & gold
        hit = len(overlap) > 0

        total += 1
        total_p += p
        total_r += r
        total_f1 += f1
        exact_count += int(exact)
        hit_count += int(hit)
        total_selected += len(selected_ids)

        total_gold_count += len(gold)
        total_pred_count += len(pred)
        total_overlap_count += len(overlap)

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
            "answer_hit": hit,
            "gold_count": len(gold),
            "pred_count": len(pred),
            "overlap_count": len(overlap),
            "top_candidates": [
                {
                    "cid": cid,
                    "score": float(score),
                    "relations": get_relations(cand),
                    "answer_count": len(get_answer_ids(cand)),
                    "path_count": cand.get("path_count", None),
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

        "answer_hit_count": hit_count,
        "answer_hit_rate": hit_count / max(1, total),

        "avg_selected_count": total_selected / max(1, total),
        "empty_selection_count": empty_count,
        "empty_selection_rate": empty_count / max(1, total),

        "avg_gold_count": total_gold_count / max(1, total),
        "avg_pred_count": total_pred_count / max(1, total),
        "avg_overlap_count": total_overlap_count / max(1, total),
    }

    write_jsonl(output_path, rows)
    write_json(summary_path, summary)
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-select", type=int, default=2)
    parser.add_argument("--second-score-threshold", type=float, default=None)
    parser.add_argument("--score-margin-threshold", type=float, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Loading model bundle...")
    print("=" * 80)
    print(f"model_path: {args.model_path}")

    bundle = load_pickle(args.model_path)
    ranker = bundle["model"]
    vectorizer = bundle["vectorizer"]
    if bundle.get("feature_version") not in {
        "raw_question_relation_interaction_graph12",
        "raw_question_pair_graph12_notemplate_v3_ablation",
    }:
        raise ValueError("Incompatible model bundle for the interaction feature predictor")

    default_second = float(bundle.get("best_second_score_threshold", 0.0))
    default_margin = float(bundle.get("best_score_margin_threshold", 0.0))

    second_score_threshold = (
        float(args.second_score_threshold)
        if args.second_score_threshold is not None
        else default_second
    )
    score_margin_threshold = (
        float(args.score_margin_threshold)
        if args.score_margin_threshold is not None
        else default_margin
    )

    best_iteration = getattr(ranker, "best_iteration_", None)

    print(f"best_iteration: {best_iteration}")
    print(f"adaptive second_score_threshold: {second_score_threshold}")
    print(f"adaptive score_margin_threshold: {score_margin_threshold}")

    print("=" * 80)
    print("Loading test pool...")
    print("=" * 80)
    print(f"test: {args.test}")

    test_examples = read_jsonl(args.test)
    print(f"test examples: {len(test_examples)}")

    print("=" * 80)
    print("Flatten and build features...")
    print("=" * 80)

    test_texts, test_nums, test_metas = flatten_examples(test_examples)
    X_test = build_features(vectorizer, test_texts, test_nums)

    print(f"test candidates: {len(test_metas)}")
    print(f"X_test shape: {X_test.shape}")

    print("=" * 80)
    print("Predict scores...")
    print("=" * 80)

    if best_iteration is not None and best_iteration > 0:
        scores = ranker.predict(X_test, num_iteration=best_iteration)
    else:
        scores = ranker.predict(X_test)

    score_index = score_index_from_predictions(test_metas, scores)

    print("=" * 80)
    print("Evaluate Top-1 / Top-2 / Adaptive...")
    print("=" * 80)

    top1 = evaluate_selection(
        examples=test_examples,
        score_index=score_index,
        mode="top1",
        max_select=1,
        second_score_threshold=0.0,
        score_margin_threshold=0.0,
        output_path=os.path.join(args.output_dir, "test_top1.eval.jsonl"),
        summary_path=os.path.join(args.output_dir, "test_top1.summary.json"),
    )

    top2 = evaluate_selection(
        examples=test_examples,
        score_index=score_index,
        mode="top2",
        max_select=2,
        second_score_threshold=0.0,
        score_margin_threshold=0.0,
        output_path=os.path.join(args.output_dir, "test_top2.eval.jsonl"),
        summary_path=os.path.join(args.output_dir, "test_top2.summary.json"),
    )

    adaptive = evaluate_selection(
        examples=test_examples,
        score_index=score_index,
        mode="adaptive",
        max_select=args.max_select,
        second_score_threshold=second_score_threshold,
        score_margin_threshold=score_margin_threshold,
        output_path=os.path.join(args.output_dir, "test_adaptive.eval.jsonl"),
        summary_path=os.path.join(args.output_dir, "test_adaptive.summary.json"),
    )

    final_summary = {
        "method": "apply_trained_lambdamart_path_ranker",
        "feature_version": bundle.get("feature_version", "unknown"),
        "model_path": args.model_path,
        "test_path": args.test,
        "test_examples": len(test_examples),
        "test_candidate_count": len(test_metas),
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "adaptive_second_score_threshold": second_score_threshold,
        "adaptive_score_margin_threshold": score_margin_threshold,
        "test_top1_summary": top1,
        "test_top2_summary": top2,
        "test_adaptive_summary": adaptive,
    }

    write_json(os.path.join(args.output_dir, "final_summary.json"), final_summary)

    print("=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
