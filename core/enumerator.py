# -*- coding: utf-8 -*-

"""枚举可执行关系序列，并计算候选答案集合及训练指标。"""

import time
from collections import defaultdict, Counter


def normalize_text(x):
    """
    最小规范化：
    当前只做字符串化和去除首尾空格。
    后续跨数据集时，如果 answer id/name 对齐复杂，可以单独扩展。
    """
    return str(x).strip()


def compute_answer_metrics(candidate_answers, gold_answers):
    """
    计算一条关系序列执行答案与 gold answer 的重合情况。

    返回：
        contains_gold:
            是否至少命中一个 gold answer

        hit_count:
            命中的 gold answer 数量

        precision:
            命中答案数 / 候选答案数

        recall:
            命中答案数 / gold answer 数

        f1:
            precision 和 recall 的调和平均
    """
    candidate_set = {
        normalize_text(x)
        for x in candidate_answers
        if normalize_text(x)
    }

    gold_set = {
        normalize_text(x)
        for x in gold_answers
        if normalize_text(x)
    }

    if not candidate_set or not gold_set:
        return {
            "contains_gold": False,
            "hit_count": 0,
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
        }

    hit_set = candidate_set & gold_set
    hit_count = len(hit_set)

    precision = hit_count / len(candidate_set) if candidate_set else 0.0
    recall = hit_count / len(gold_set) if gold_set else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "contains_gold": hit_count > 0,
        "hit_count": hit_count,
        "answer_precision": precision,
        "answer_recall": recall,
        "answer_f1": f1,
    }


def enumerate_relation_sequences(
    kg,
    anchors,
    gold_answers,
    max_hop: int = 3,
    max_frontier_size: int = 50000,
    max_sequences: int = 20000,
    max_answer_per_sequence: int = 500,
    avoid_repeat_entity: bool = True,
):
    """
    从 anchor 实体出发枚举可执行关系序列。

    核心过程：
        1. 从 anchor 出发做多跳图遍历
        2. 把实体路径压缩成关系序列
        3. 聚合同一关系序列对应的候选答案
        4. 计算每条关系序列的 answer precision / recall / f1
    """
    start_time = time.time()

    # 只保留 KG 中真实存在的 anchor
    valid_anchors = [
        normalize_text(anchor)
        for anchor in anchors
        if kg.has_entity(normalize_text(anchor))
    ]

    # frontier 中每个元素表示一条当前路径状态
    frontier = []

    for anchor in valid_anchors:
        frontier.append({
            "entity": anchor,
            "relations": tuple(),
            "visited": (anchor,),
        })

    # 关系序列 -> 该序列执行后到达的候选答案集合
    seq_to_answers = defaultdict(set)

    # 关系序列 -> 支持该序列的实体路径数量
    seq_to_path_count = Counter()

    frontier_sizes = []
    expanded_state_count = 0
    truncated_frontier_count = 0
    truncated_sequence_count = False
    skipped_repeat_count = 0

    # 逐跳扩展
    for depth in range(1, max_hop + 1):
        new_frontier = []

        for state in frontier:
            current_entity = state["entity"]
            previous_relations = state["relations"]
            visited_entities = state["visited"]

            for relation, next_entity in kg.neighbors(current_entity):

                # 通用防环机制：避免同一路径中重复访问实体
                if avoid_repeat_entity and next_entity in visited_entities:
                    skipped_repeat_count += 1
                    continue

                new_relations = previous_relations + (relation,)

                # 防止关系序列数量爆炸
                if len(seq_to_answers) >= max_sequences and new_relations not in seq_to_answers:
                    truncated_sequence_count = True
                    continue

                # 记录该关系序列能到达的候选答案
                if len(seq_to_answers[new_relations]) < max_answer_per_sequence:
                    seq_to_answers[new_relations].add(next_entity)

                # 记录实体路径支持数
                seq_to_path_count[new_relations] += 1

                new_frontier.append({
                    "entity": next_entity,
                    "relations": new_relations,
                    "visited": visited_entities + (next_entity,),
                })

                expanded_state_count += 1

        # 防止 frontier 爆炸
        if len(new_frontier) > max_frontier_size:
            new_frontier = new_frontier[:max_frontier_size]
            truncated_frontier_count += 1

        frontier_sizes.append(len(new_frontier))
        frontier = new_frontier

        if not frontier:
            break

    candidates = []
    answer_covered = False
    min_gold_hop = None
    total_candidate_answers = set()

    best_answer_f1 = 0.0
    best_answer_recall = 0.0
    best_answer_precision = 0.0
    best_sequence_count = 0

    for relation_sequence, answer_set in seq_to_answers.items():
        answer_metrics = compute_answer_metrics(answer_set, gold_answers)

        contains_gold = answer_metrics["contains_gold"]

        if contains_gold:
            answer_covered = True
            hop = len(relation_sequence)

            if min_gold_hop is None or hop < min_gold_hop:
                min_gold_hop = hop

        total_candidate_answers.update(answer_set)

        answer_f1 = answer_metrics["answer_f1"]

        if answer_f1 > best_answer_f1:
            best_answer_f1 = answer_f1
            best_answer_precision = answer_metrics["answer_precision"]
            best_answer_recall = answer_metrics["answer_recall"]
            best_sequence_count = 1
        elif answer_f1 == best_answer_f1 and answer_f1 > 0:
            best_sequence_count += 1

        candidate = {
            "relations": list(relation_sequence),
            "hop": len(relation_sequence),
            "answer_ids": sorted(answer_set),
            "answer_count": len(answer_set),
            "path_count": int(seq_to_path_count[relation_sequence]),

            # 粗粒度覆盖指标
            "contains_gold": contains_gold,
            "hit_count": answer_metrics["hit_count"],

            # 更可靠的答案质量指标
            "answer_precision": answer_metrics["answer_precision"],
            "answer_recall": answer_metrics["answer_recall"],
            "answer_f1": answer_metrics["answer_f1"],
        }

        candidates.append(candidate)

    # 按关系序列和结构字段确定候选的稳定输出顺序。
    candidates.sort(
        key=lambda item: (
            tuple(str(x) for x in item["relations"]),
            item["hop"],
            item["answer_count"],
            -item["path_count"],
        )
    )

    stats = {
        "valid_anchor_count": len(valid_anchors),

        # 候选空间规模
        "num_sequences": len(candidates),
        "num_candidate_answers": len(total_candidate_answers),

        # 是否至少有一条序列覆盖 gold answer
        "answer_covered": answer_covered,
        "min_gold_hop": min_gold_hop,

        # oracle 质量：候选空间中最好关系序列能做到什么程度
        "best_answer_f1": best_answer_f1,
        "best_answer_precision": best_answer_precision,
        "best_answer_recall": best_answer_recall,
        "best_sequence_count": best_sequence_count,

        # 搜索过程诊断
        "frontier_sizes": frontier_sizes,
        "expanded_state_count": expanded_state_count,
        "truncated_frontier_count": truncated_frontier_count,
        "truncated_sequence_count": truncated_sequence_count,
        "skipped_repeat_count": skipped_repeat_count,
        "elapsed_sec": round(time.time() - start_time, 6),
    }

    return candidates, stats
