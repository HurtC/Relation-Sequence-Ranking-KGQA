# MetaQA 图关系序列排序主线

本目录保留论文当前结果对应的主线代码、冻结模型和结果摘要。代码流程为：

1. 从主题实体出发，在带反向边的知识图谱中枚举可执行关系序列。
2. 执行关系序列，得到候选答案集合、路径实例数和结构统计量。
3. 1-hop、2-hop直接使用完整候选池；3-hop用训练集拟合的逻辑回归粗排器保留Top-30。
4. 将同一问题的候选路径组成一个排序组，以候选答案集合F1映射得到的0–20级标签训练LambdaMART。
5. 1-hop使用原始问题—路径TF-IDF和12项统计特征；2-hop、3-hop额外加入问题词与带位置、方向关系词的局部交叉项。
6. 测试时选择得分最高的一条关系序列，执行后得到答案集合，并计算Precision、Recall、Set F1、Exact和Set-Hits@1。

## 文件说明

- `prepare_data.py`：按固定种子生成训练集和开发集子集。
- `build_candidates.py`：生成可执行关系序列及候选答案集合。
- `core/`：数据字段、知识图谱、路径枚举、读写和候选池指标。
- `train_split.py`：训练3-hop候选粗排器。
- `split.py`：用粗排器保留Top-K候选。
- `graph_features_base.py`：原始问题—路径文本和12项统计特征。
- `graph_features.py`：在基础表示上加入局部词—关系交叉项，数值部分仍为12项。
- `train_ranker_base.py`：训练1-hop LambdaMART。
- `train_ranker.py`：训练2-hop、3-hop LambdaMART。
- `predict_base.py`：执行1-hop模型并评测。
- `predict.py`：执行2-hop、3-hop模型并评测。
- `compare_models.py`：在同一开发集矩阵上比较LR、随机森林、ExtraTrees和LambdaMART。
- `feature_ablation.py`：运行Full、Text-only和Numeric-only消融。
- `run_pipeline.sh`：依次运行完整训练与测试流程。
- `models/`：当前冻结的3个LambdaMART模型和3-hop粗排模型。
- `results/`：完整测试、粗筛、模型消融和特征消融结果。
- `PAPER_CHECK.md`：当前文稿中需要人工修正的明显问题。

## 环境

服务器实验环境为Python、NumPy、SciPy、scikit-learn和LightGBM，精确版本见`requirements.txt`。

```bash
python -m pip install -r requirements.txt
```

## 运行

将`DATA_ROOT`指向包含`kb.txt`和`processed/`的MetaQA目录：

```bash
export DATA_ROOT=/path/to/data/metaqa
export OUTPUT_ROOT=/path/to/outputs
bash run_pipeline.sh
```

默认使用每个hop的10,000条训练样本和3,000条开发样本，随机种子分别为42和4242。三个排序器均使用50,000维文本词表、学习率0.03、31个叶节点和最多1,000轮迭代，并在开发集NDCG上早停。

## 当前完整测试集结果

| 任务 | 样本数 | Set F1 | Exact | Set-Hits@1 |
|---|---:|---:|---:|---:|
| 1-hop | 9,947 | 98.015% | 94.501% | 99.940% |
| 2-hop | 14,872 | 99.962% | 99.939% | 99.973% |
| 3-hop | 14,274 | 98.190% | 87.060% | 99.559% |
| 按题数加权 | 39,093 | 98.820% | 93.853% | 99.813% |

精确小数保存在`results/main_results.json`及各hop的原始摘要中。2-hop开发集F1为99.995238%，表格建议保留三位小数，避免显示成100.00%。
