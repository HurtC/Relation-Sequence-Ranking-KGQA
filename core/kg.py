# -*- coding: utf-8 -*-

"""加载知识图谱，并建立正向边和可选反向边索引。"""

from collections import defaultdict
from typing import Optional, Tuple


class KnowledgeGraph:
    def __init__(self):
        # 出边索引：实体 -> [(关系, 尾实体), ...]
        self.out_edges = defaultdict(list)

        # 实体集合
        self.entities = set()

        # 关系集合
        self.relations = set()

        # 三元组数量
        self.triple_count = 0

        # 原始三元组数量，不包含反向边
        self.raw_triple_count = 0

    @staticmethod
    def parse_line(line: str) -> Optional[Tuple[str, str, str]]:
        """
        解析一行三元组。

        兼容：
            head|relation|tail
            head \t relation \t tail
        """
        line = line.strip()

        if not line:
            return None

        if "\t" in line:
            parts = line.split("\t")
        elif "|" in line:
            parts = line.split("|")
        else:
            return None

        parts = [p.strip() for p in parts]

        if len(parts) < 3:
            return None

        head, relation, tail = parts[0], parts[1], parts[2]

        if not head or not relation or not tail:
            return None

        return head, relation, tail

    def add_triple(self, head: str, relation: str, tail: str) -> None:
        """
        添加一条有向边：

            head --relation--> tail
        """
        self.out_edges[head].append((relation, tail))
        self.entities.add(head)
        self.entities.add(tail)
        self.relations.add(relation)
        self.triple_count += 1

    @classmethod
    def load(cls, kg_path: str, add_reverse_edges: bool = False):
        """
        加载知识图谱。

        参数：
            kg_path:
                KG 文件路径

            add_reverse_edges:
                是否加入反向边。
                如果为 True，则每条原始三元组：

                    h --r--> t

                会额外加入：

                    t --__rev__r--> h
        """
        kg = cls()
        bad_lines = 0
        raw_triple_count = 0

        with open(kg_path, "r", encoding="utf-8") as f:
            for line in f:
                item = cls.parse_line(line)

                if item is None:
                    bad_lines += 1
                    continue

                head, relation, tail = item

                # 原始正向边
                kg.add_triple(head, relation, tail)
                raw_triple_count += 1

                # 通用反向边
                if add_reverse_edges:
                    reverse_relation = "__rev__" + relation
                    kg.add_triple(tail, reverse_relation, head)

        kg.raw_triple_count = raw_triple_count

        print("=" * 100)
        print("KG loaded")
        print("=" * 100)
        print(f"kg_path          : {kg_path}")
        print(f"raw_triple_count : {kg.raw_triple_count}")
        print(f"triple_count     : {kg.triple_count}")
        print(f"entity_count     : {len(kg.entities)}")
        print(f"relation_count   : {len(kg.relations)}")
        print(f"bad_lines        : {bad_lines}")
        print(f"add_reverse_edges: {add_reverse_edges}")
        print("=" * 100)

        return kg

    def has_entity(self, entity: str) -> bool:
        """
        判断实体是否存在于 KG 中。
        """
        return entity in self.entities

    def neighbors(self, entity: str):
        """
        获取实体的一跳邻居。

        返回：
            [(relation, tail), ...]
        """
        return self.out_edges.get(entity, [])
