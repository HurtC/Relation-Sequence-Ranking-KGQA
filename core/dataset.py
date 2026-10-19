# -*- coding: utf-8 -*-

"""统一读取问题、答案、主题实体和样本编号字段。"""


def to_list(x):
    """
    把输入统一转成 list。

    例如：
        None -> []
        "a" -> ["a"]
        ["a", "b"] -> ["a", "b"]
    """
    if x is None:
        return []

    if isinstance(x, list):
        return x

    return [x]


def extract_qid(sample, idx: int) -> str:
    """
    提取样本 ID。

    如果样本中没有 qid，就用 sample_编号 代替。
    """
    for key in ["qid", "id", "question_id", "uid"]:
        if key in sample and sample[key] is not None:
            return str(sample[key])

    return f"sample_{idx}"


def extract_question(sample) -> str:
    """
    提取问题文本。
    """
    for key in ["question", "Question", "query", "utterance"]:
        if key in sample and sample[key] is not None:
            return str(sample[key])

    return ""


def extract_answers(sample):
    """
    提取 gold answer。

    支持以下字段：
        answers
        answer
        gold_answers
        gold_answer
        answer_ids
        gold_answer_ids

    支持以下格式：
        "Tom Hanks"
        ["Tom Hanks", "A"]
        {"id": "...", "name": "..."}
        [{"id": "...", "name": "..."}]
    """
    fields = [
        "answers",
        "answer",
        "gold_answers",
        "gold_answer",
        "answer_ids",
        "gold_answer_ids",
    ]

    raw = None

    for key in fields:
        if key in sample:
            raw = sample[key]
            break

    answers = []

    for item in to_list(raw):
        if item is None:
            continue

        if isinstance(item, str):
            answers.append(item)

        elif isinstance(item, dict):
            # 优先取 id；如果没有 id，再取 name/text 等字段
            for k in ["id", "name", "text", "answer", "value", "label"]:
                if k in item and item[k] is not None:
                    answers.append(str(item[k]))
                    break

        else:
            answers.append(str(item))

    return sorted(set(str(x).strip() for x in answers if str(x).strip()))


def extract_topic_entities(sample):
    """
    提取主题实体。

    当前 MetaQA 如果已经有 topic entity，就直接用。
    后续没有 topic entity 的数据集，会额外写 anchor discovery 模块。

    支持字段：
        topic_entities
        topic_entity
        topic_entity_id
        topic_entity_ids
        entities
        head_entity
        subject
    """
    fields = [
        "topic_entities",
        "topic_entity",
        "topic_entity_id",
        "topic_entity_ids",
        "entities",
        "head_entity",
        "subject",
    ]

    raw = None

    for key in fields:
        if key in sample:
            raw = sample[key]
            break

    topics = []

    for item in to_list(raw):
        if item is None:
            continue

        if isinstance(item, str):
            topics.append(item)

        elif isinstance(item, dict):
            # 优先取 id；如果没有 id，再取 name/text 等字段
            for k in ["id", "name", "text", "entity", "value", "label"]:
                if k in item and item[k] is not None:
                    topics.append(str(item[k]))
                    break

        else:
            topics.append(str(item))

    return sorted(set(str(x).strip() for x in topics if str(x).strip()))
