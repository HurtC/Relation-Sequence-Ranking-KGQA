# -*- coding: utf-8 -*-

"""从主题实体出发枚举可执行关系序列，并输出候选答案与统计信息。"""

import argparse

from core.io_utils import read_jsonl, write_jsonl, write_json
from core.kg import KnowledgeGraph
from core.dataset import (
    extract_qid,
    extract_question,
    extract_answers,
    extract_topic_entities,
)
from core.enumerator import enumerate_relation_sequences
from core.metrics import summarize_candidate_space, print_summary


def build_output_row(
    idx,
    sample,
    dataset_name,
    max_hop,
    candidates,
    stats,
    save_all_candidates,
    max_candidates_to_save,
    prediction_input,
):
    """
    构造单条样本的输出结果。

    输出中保存：
        qid
        question
        gold_answers
        anchors
        candidate_relation_sequences
        stats

    为了避免输出文件过大，默认只保存：
        1. 最多 50 条 positive candidates
        2. 最多 50 条 negative candidates

    如果指定 --save-all-candidates，
    则保存排序后的前 max_candidates_to_save 条候选。
    """
    qid = extract_qid(sample, idx)
    question = extract_question(sample)
    gold_answers = extract_answers(sample)
    anchors = extract_topic_entities(sample)

    if save_all_candidates:
        saved_candidates = candidates[:max_candidates_to_save]
    else:
        positive_candidates = [
            candidate for candidate in candidates
            if candidate["contains_gold"]
        ]

        negative_candidates = [
            candidate for candidate in candidates
            if not candidate["contains_gold"]
        ]

        saved_candidates = positive_candidates[:50] + negative_candidates[:50]

    if prediction_input:
        label_fields = {
            "contains_gold",
            "hit_count",
            "answer_precision",
            "answer_recall",
            "answer_f1",
        }
        saved_candidates = [
            {key: value for key, value in candidate.items() if key not in label_fields}
            for candidate in saved_candidates
        ]

    return {
        "qid": qid,
        "dataset": dataset_name,
        "question": question,
        "gold_answers": gold_answers,
        "anchors": anchors,
        "max_hop": max_hop,
        "candidate_relation_sequences": saved_candidates,
        "stats": stats,
    }


def run(args):
    """
    主运行流程。
    """
    print("=" * 100)
    print("Arguments")
    print("=" * 100)

    for key, value in vars(args).items():
        print(f"{key:35s}: {value}")

    print("=" * 100)

    # 1. 加载知识图谱
    kg = KnowledgeGraph.load(
        args.kg,
        add_reverse_edges=args.add_reverse_edges,
    )

    # 2. 加载 clean jsonl 数据
    samples = read_jsonl(args.input, args.sample_size)

    print("=" * 100)
    print("Dataset loaded")
    print("=" * 100)
    print(f"dataset    : {args.dataset}")
    print(f"input      : {args.input}")
    print(f"num_samples: {len(samples)}")
    print("=" * 100)

    results = []

    # 3. 对每个问题构建候选关系序列空间
    for idx, sample in enumerate(samples):
        if (idx + 1) % args.log_every == 0:
            print(f"[Progress] {idx + 1}/{len(samples)}")

        # 提取 gold answer
        gold_answers = extract_answers(sample)

        # 提取 topic entity
        # 当前 MetaQA 先使用 clean 文件中已有的主题实体
        anchors = extract_topic_entities(sample)

        # 4. 从 topic entity 出发枚举关系序列
        candidates, stats = enumerate_relation_sequences(
            kg=kg,
            anchors=anchors,
            gold_answers=[] if args.prediction_input else gold_answers,
            max_hop=args.max_hop,
            max_frontier_size=args.max_frontier_size,
            max_sequences=args.max_sequences,
            max_answer_per_sequence=args.max_answer_per_sequence,
            avoid_repeat_entity=not args.allow_repeat_entity,
        )

        # 5. 构造输出行
        row = build_output_row(
            idx=idx,
            sample=sample,
            dataset_name=args.dataset,
            max_hop=args.max_hop,
            candidates=candidates,
            stats=stats,
            save_all_candidates=args.save_all_candidates,
            max_candidates_to_save=args.max_candidates_to_save,
            prediction_input=args.prediction_input,
        )

        results.append(row)

    # 6. 汇总指标
    summary = summarize_candidate_space(results)

    # 7. 写出逐样本结果和 summary
    write_jsonl(args.output, results)
    write_json(args.summary, summary)

    print_summary(summary)

    print(f"Saved details: {args.output}")
    print(f"Saved summary: {args.summary}")


def parse_args():
    """
    命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Evaluate candidate reasoning space based on executable relation sequences."
    )

    # 数据集名称，仅用于输出记录
    parser.add_argument("--dataset", type=str, default="metaqa")

    # KG 路径
    parser.add_argument("--kg", type=str, default="data/metaqa/kb.txt")
    parser.add_argument(
    "--add-reverse-edges",
    action="store_true",
    help="是否为每条三元组加入通用反向边",
    )

    # clean jsonl 输入路径
    parser.add_argument("--input", type=str, required=True)

    # 逐样本输出路径
    parser.add_argument("--output", type=str, required=True)

    # 汇总指标输出路径
    parser.add_argument("--summary", type=str, required=True)

    # 只读取前 N 条；-1 表示全量
    parser.add_argument("--sample-size", type=int, default=-1)

    # 最大扩展跳数
    parser.add_argument("--max-hop", type=int, default=3)

    # 防止图搜索爆炸：每层 frontier 最大状态数
    parser.add_argument("--max-frontier-size", type=int, default=50000)

    # 防止候选序列爆炸：最多保留多少种关系序列
    parser.add_argument("--max-sequences", type=int, default=20000)

    # 防止单条序列答案过多：每条关系序列最多保存多少个答案
    parser.add_argument("--max-answer-per-sequence", type=int, default=500)

    # 默认禁止路径中重复实体；加上该参数后允许重复
    parser.add_argument("--allow-repeat-entity", action="store_true")

    # 是否保存所有候选。默认只保存部分候选，避免文件太大
    parser.add_argument("--save-all-candidates", action="store_true")

    # 如果保存所有候选，最多保存前多少条
    parser.add_argument("--max-candidates-to-save", type=int, default=200)

    # 生成预测输入时移除候选级训练标签。
    parser.add_argument("--prediction-input", action="store_true")

    # 每多少条打印一次进度
    parser.add_argument("--log-every", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
