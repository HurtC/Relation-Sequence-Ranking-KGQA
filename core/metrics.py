# -*- coding: utf-8 -*-

"""汇总候选空间规模、覆盖率和候选池 Oracle 指标。"""

from collections import Counter


def avg(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def percentile(values, p: float):
    if not values:
        return 0.0

    values = sorted(values)
    idx = int(round((len(values) - 1) * p))

    return values[idx]


def summarize_candidate_space(rows):
    total = len(rows)

    if total == 0:
        return {
            "total_examples": 0
        }

    anchor_available = [
        1 if row["stats"]["valid_anchor_count"] > 0 else 0
        for row in rows
    ]

    covered = [
        1 if row["stats"]["answer_covered"] else 0
        for row in rows
    ]

    sequence_counts = [
        row["stats"]["num_sequences"]
        for row in rows
    ]

    candidate_answer_counts = [
        row["stats"]["num_candidate_answers"]
        for row in rows
    ]

    elapsed_seconds = [
        row["stats"]["elapsed_sec"]
        for row in rows
    ]

    expanded_state_counts = [
        row["stats"]["expanded_state_count"]
        for row in rows
    ]

    best_answer_f1_values = [
        row["stats"].get("best_answer_f1", 0.0)
        for row in rows
    ]

    best_answer_precision_values = [
        row["stats"].get("best_answer_precision", 0.0)
        for row in rows
    ]

    best_answer_recall_values = [
        row["stats"].get("best_answer_recall", 0.0)
        for row in rows
    ]

    best_sequence_counts = [
        row["stats"].get("best_sequence_count", 0)
        for row in rows
    ]

    min_gold_hops = [
        row["stats"]["min_gold_hop"]
        for row in rows
        if row["stats"]["min_gold_hop"] is not None
    ]

    hop_counter = Counter(min_gold_hops)

    summary = {
        "total_examples": total,

        "anchor_available_count": sum(anchor_available),
        "anchor_available_rate": avg(anchor_available),

        "answer_covered_count": sum(covered),
        "answer_coverage_rate": avg(covered),

        # oracle 上限指标
        "avg_best_answer_f1": avg(best_answer_f1_values),
        "median_best_answer_f1": percentile(best_answer_f1_values, 0.5),
        "p90_best_answer_f1": percentile(best_answer_f1_values, 0.9),
        "min_best_answer_f1": min(best_answer_f1_values),

        "avg_best_answer_precision": avg(best_answer_precision_values),
        "avg_best_answer_recall": avg(best_answer_recall_values),
        "avg_best_sequence_count": avg(best_sequence_counts),
        "max_best_sequence_count": max(best_sequence_counts),

        # 候选关系序列数量
        "avg_sequence_count": avg(sequence_counts),
        "median_sequence_count": percentile(sequence_counts, 0.5),
        "p90_sequence_count": percentile(sequence_counts, 0.9),
        "p95_sequence_count": percentile(sequence_counts, 0.95),
        "max_sequence_count": max(sequence_counts),

        # 候选答案数量
        "avg_candidate_answer_count": avg(candidate_answer_counts),
        "median_candidate_answer_count": percentile(candidate_answer_counts, 0.5),
        "p90_candidate_answer_count": percentile(candidate_answer_counts, 0.9),
        "p95_candidate_answer_count": percentile(candidate_answer_counts, 0.95),
        "max_candidate_answer_count": max(candidate_answer_counts),

        # 图搜索规模
        "avg_expanded_state_count": avg(expanded_state_counts),
        "median_expanded_state_count": percentile(expanded_state_counts, 0.5),
        "p95_expanded_state_count": percentile(expanded_state_counts, 0.95),
        "max_expanded_state_count": max(expanded_state_counts),

        # 运行时间
        "avg_elapsed_sec": avg(elapsed_seconds),
        "median_elapsed_sec": percentile(elapsed_seconds, 0.5),
        "p95_elapsed_sec": percentile(elapsed_seconds, 0.95),
        "max_elapsed_sec": max(elapsed_seconds),

        "min_gold_hop_counter": dict(sorted(hop_counter.items())),
    }

    return summary


def print_summary(summary):
    print("=" * 100)
    print("Candidate Reasoning Space Summary")
    print("=" * 100)

    for key, value in summary.items():
        print(f"{key:35s}: {value}")

    print("=" * 100)
