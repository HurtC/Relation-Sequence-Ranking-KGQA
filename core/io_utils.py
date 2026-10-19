# -*- coding: utf-8 -*-

"""提供 JSON、JSONL 读写和输出目录创建函数。"""

import json
import os


def ensure_dir(path: str) -> None:
    """
    确保输出文件所在目录存在。

    例如：
    path = candidate_reasoning_space/outputs/result.jsonl
    如果 outputs 目录不存在，就自动创建。
    """
    dirname = os.path.dirname(path)

    if dirname:
        os.makedirs(dirname, exist_ok=True)


def read_jsonl(path: str, sample_size: int = -1):
    """
    读取 jsonl 文件。

    参数：
        path: 输入文件路径
        sample_size: 只读取前多少条；-1 表示全部读取

    返回：
        rows: List[dict]
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

            if sample_size > 0 and len(rows) >= sample_size:
                break

    return rows


def write_jsonl(path: str, rows) -> None:
    """
    写出 jsonl 文件。

    每一行是一个 JSON 对象。
    """
    ensure_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj) -> None:
    """
    写出普通 json 文件。

    主要用于保存 summary 指标。
    """
    ensure_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
