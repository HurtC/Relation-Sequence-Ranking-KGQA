#!/usr/bin/env bash
# 依次完成数据抽样、候选生成、3-hop 粗筛、LambdaMART 训练和测试集评测。

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/metaqa}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TRAIN_SIZE="${TRAIN_SIZE:-10000}"
DEV_SIZE="${DEV_SIZE:-3000}"

mkdir -p "${OUTPUT_ROOT}"

for HOP in 1 2 3; do
  TASK_ROOT="${OUTPUT_ROOT}/metaqa_${HOP}hop"
  mkdir -p "${TASK_ROOT}/data" "${TASK_ROOT}/candidates" "${TASK_ROOT}/split" \
    "${TASK_ROOT}/model" "${TASK_ROOT}/evaluation"

  "${PYTHON_BIN}" prepare_data.py \
    --input "${DATA_ROOT}/processed/${HOP}hop.train.clean.jsonl" \
    --output "${TASK_ROOT}/data/train.jsonl" \
    --summary "${TASK_ROOT}/data/train.summary.json" \
    --sample-size "${TRAIN_SIZE}" --seed 42

  "${PYTHON_BIN}" prepare_data.py \
    --input "${DATA_ROOT}/processed/${HOP}hop.dev.clean.jsonl" \
    --output "${TASK_ROOT}/data/dev.jsonl" \
    --summary "${TASK_ROOT}/data/dev.summary.json" \
    --sample-size "${DEV_SIZE}" --seed 4242

  for PART in train dev; do
    "${PYTHON_BIN}" build_candidates.py \
      --dataset "metaqa_${HOP}hop" --kg "${DATA_ROOT}/kb.txt" --add-reverse-edges \
      --input "${TASK_ROOT}/data/${PART}.jsonl" \
      --output "${TASK_ROOT}/candidates/${PART}.jsonl" \
      --summary "${TASK_ROOT}/candidates/${PART}.summary.json" \
      --max-hop "${HOP}" --save-all-candidates --log-every 100
  done

  "${PYTHON_BIN}" build_candidates.py \
    --dataset "metaqa_${HOP}hop" --kg "${DATA_ROOT}/kb.txt" --add-reverse-edges \
    --input "${DATA_ROOT}/processed/${HOP}hop.test.clean.jsonl" \
    --output "${TASK_ROOT}/candidates/test.jsonl" \
    --summary "${TASK_ROOT}/candidates/test.summary.json" \
    --max-hop "${HOP}" --save-all-candidates --prediction-input --log-every 500

  TRAIN_POOL="${TASK_ROOT}/candidates/train.jsonl"
  DEV_POOL="${TASK_ROOT}/candidates/dev.jsonl"
  TEST_POOL="${TASK_ROOT}/candidates/test.jsonl"

  if [[ "${HOP}" == "3" ]]; then
    "${PYTHON_BIN}" train_split.py \
      --train "${TRAIN_POOL}" --dev "${DEV_POOL}" \
      --output-dir "${TASK_ROOT}/split/model" --top-k 30
    for PART in train dev test; do
      "${PYTHON_BIN}" split.py \
        --model-path "${TASK_ROOT}/split/model/coarse_logreg.pkl" \
        --input "${TASK_ROOT}/candidates/${PART}.jsonl" \
        --output "${TASK_ROOT}/split/${PART}.jsonl" \
        --summary "${TASK_ROOT}/split/${PART}.summary.json" --top-k 30
    done
    TRAIN_POOL="${TASK_ROOT}/split/train.jsonl"
    DEV_POOL="${TASK_ROOT}/split/dev.jsonl"
    TEST_POOL="${TASK_ROOT}/split/test.jsonl"
  fi

  if [[ "${HOP}" == "1" ]]; then
    TRAIN_PROGRAM="train_ranker_base.py"
    PREDICT_PROGRAM="predict_base.py"
  else
    TRAIN_PROGRAM="train_ranker.py"
    PREDICT_PROGRAM="predict.py"
  fi

  "${PYTHON_BIN}" "${TRAIN_PROGRAM}" \
    --train "${TRAIN_POOL}" --dev "${DEV_POOL}" \
    --output-dir "${TASK_ROOT}/model" --relevance-scheme linear20 \
    --max-features 50000 --n-estimators 1000 --learning-rate 0.03 \
    --num-leaves 31 --min-child-samples 10 --subsample 0.9 \
    --colsample-bytree 0.9 --reg-alpha 0.0 --reg-lambda 1.0 \
    --random-state 42 --max-select 2

  "${PYTHON_BIN}" "${PREDICT_PROGRAM}" \
    --model-path "${TASK_ROOT}/model/lambdamart_path_ranker.pkl" \
    --test "${TEST_POOL}" --output-dir "${TASK_ROOT}/evaluation" --max-select 2
done
