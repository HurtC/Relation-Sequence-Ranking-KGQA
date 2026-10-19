#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""按固定随机种子生成可复现的训练集或开发集子集。"""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
    return rows


def stable_key(row: Dict[str, Any], index: int, seed: int) -> str:
    identity = str(row.get("qid", row.get("id", index)))
    question = str(row.get("question", ""))
    payload = f"{seed}\0{identity}\0{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (stable_key(item[1], item[0], args.seed), item[0]),
    )
    sample_size = min(max(0, args.sample_size), len(ranked))
    selected = [row for _, row in ranked[:sample_size]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": args.output,
        "source_examples": len(rows),
        "selected_examples": len(selected),
        "selection": "deterministic_uniform_hash_sample",
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
