"""
研知通 - 知识图谱模块
基于 NetworkX 的轻量知识图谱：实体/关系抽取 → 子图检索增强 RAG
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

import config

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """轻量知识图谱，存储实体节点和关系边"""

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_to_node: Dict[str, str] = {}  # entity_name -> node_id
        self.node_entities: Dict[str, Set[str]] = {}  # node_id -> {entity_names}
        self.node_sources: Dict[str, Set[str]] = {}  # node_id -> {source_files}

    def add_node(self, entity_name: str, entity_type: str = "unknown",
                 source_file: str = ""):
        """添加实体节点（同名实体合并）"""
        if not entity_name or not entity_name.strip():
            return
        name = entity_name.strip()

        if name in self.entity_to_node:
            node_id = self.entity_to_node[name]
            self.node_entities[node_id].add(name)
            if source_file:
                self.node_sources.setdefault(node_id, set()).add(source_file)
            return node_id

        node_id = f"e{len(self.entity_to_node)}"
        self.entity_to_node[name] = node_id
        self.graph.add_node(node_id, label=name, type=entity_type)
        self.node_entities[node_id] = {name}
        if source_file:
            self.node_sources[node_id] = {source_file}
        return node_id

    def add_edge(self, source: str, target: str, relation: str,
                 source_file: str = ""):
        """添加关系边"""
        src_id = self.add_node(source, source_file=source_file)
        tgt_id = self.add_node(target, source_file=source_file)
        if src_id and tgt_id and src_id != tgt_id:
            relations = self.graph.get_edge_data(src_id, tgt_id, default={})
            existing = relations.get("relations", set())
            if isinstance(existing, list):
                existing = set(existing)
            existing.add(relation)
            self.graph.add_edge(src_id, tgt_id, relations=list(existing))
            if source_file:
                srcs = self.graph.get_edge_data(src_id, tgt_id, default={}).get("sources", set())
                if isinstance(srcs, list):
                    srcs = set(srcs)
                srcs.add(source_file)
                self.graph.add_edge(src_id, tgt_id, sources=list(srcs))

    def subgraph_around(self, entities: List[str], max_depth: int = 2) -> nx.Graph:
        """获取指定实体周围的子图（BFS 展开 max_depth 跳）"""
        if not self.graph or not entities:
            return nx.Graph()

        seed_nodes = []
        for name in entities:
            node_id = self.entity_to_node.get(name.strip())
            if node_id:
                seed_nodes.append(node_id)

        if not seed_nodes:
            # 模糊匹配：在 node labels 中搜索
            for name in entities:
                name_lower = name.strip().lower()
                for nid, data in self.graph.nodes(data=True):
                    label = data.get("label", "").lower()
                    if name_lower in label or label in name_lower:
                        seed_nodes.append(nid)
                        break

        if not seed_nodes:
            return nx.Graph()

        # BFS 展开子图
        visited = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return self.graph.subgraph(visited).copy()

    def to_context(self, subgraph: nx.Graph, max_triples: int = 20) -> str:
        """将子图转换为文本上下文"""
        parts = []

        # 实体节点
        nodes_data = []
        for nid in subgraph.nodes():
            label = subgraph.nodes[nid].get("label", nid)
            ntype = subgraph.nodes[nid].get("type", "unknown")
            nodes_data.append(f"- {label}（{ntype}）")
        if nodes_data:
            parts.append("【知识图谱实体】\n" + "\n".join(nodes_data[:20]))

        # 关系边
        edges_data = []
        for src, tgt, data in subgraph.edges(data=True):
            s_label = subgraph.nodes[src].get("label", src)
            t_label = subgraph.nodes[tgt].get("label", tgt)
            relations = data.get("relations", [])
            rel_str = "、".join(relations) if relations else "关联"
            edges_data.append(f"- {s_label} --[{rel_str}]--> {t_label}")
        if edges_data:
            parts.append("【知识图谱关系】\n" + "\n".join(edges_data[:max_triples]))

        return "\n\n".join(parts) if parts else ""

    def stats(self) -> dict:
        """返回图谱统计"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_entities": len(self.entity_to_node),
        }

    def to_dict(self) -> dict:
        """序列化为 JSON"""
        return {
            "entity_to_node": self.entity_to_node,
            "node_entities": {k: list(v) for k, v in self.node_entities.items()},
            "node_sources": {k: list(v) for k, v in self.node_sources.items()},
            "nodes": [
                {"id": nid, **data}
                for nid, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """从 JSON 反序列化"""
        kg = cls()
        kg.entity_to_node = data.get("entity_to_node", {})
        kg.node_entities = {k: set(v) for k, v in data.get("node_entities", {}).items()}
        kg.node_sources = {k: set(v) for k, v in data.get("node_sources", {}).items()}

        for node_data in data.get("nodes", []):
            nid = node_data.pop("id")
            kg.graph.add_node(nid, **node_data)

        for edge_data in data.get("edges", []):
            src = edge_data.pop("source")
            tgt = edge_data.pop("target")
            kg.graph.add_edge(src, tgt, **edge_data)

        return kg


# ──────────────────────────────────────────
# 实体/关系提取
# ──────────────────────────────────────────

def extract_triples_batch(
    texts: List[Tuple[str, str]],  # [(chunk_text, source_file), ...]
    llm,
    max_entities_per_chunk: int = None,
) -> List[Tuple[str, str, str, str]]:
    """
    批量提取实体-关系三元组
    返回: [(entity1, relation, entity2, source_file), ...]
    """
    if max_entities_per_chunk is None:
        max_entities_per_chunk = config.KG_MAX_ENTITIES_PER_CHUNK

    all_triples = []
    processed = 0

    # 每次最多处理 3 个 chunk，避免 prompt 过长
    batch_size = 3
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        chunks_prompt = ""
        for j, (text, src) in enumerate(batch):
            snippet = text[:500].replace("\n", " ")
            chunks_prompt += f"\n[Chunk {j+1} 来源：{src}]\n{snippet}\n"

        prompt = (
            f"从以下文本片段中提取知识三元组（实体1, 关系, 实体2）。\n"
            f"每个片段最多提取 {max_entities_per_chunk} 组。\n"
            f"文本：{chunks_prompt}\n"
            f"只返回 JSON 数组："
            f'[{{"entity1": "...", "relation": "...", "entity2": "...", "chunk": 1}}, ...]'
        )

        try:
            response = llm.invoke(prompt)
            triples = _parse_triples(response.content)
            for t in triples:
                chunk_idx = t.get("chunk", 1)
                if 1 <= chunk_idx <= len(batch):
                    src = batch[chunk_idx - 1][1]
                else:
                    src = ""
                all_triples.append((
                    t.get("entity1", ""),
                    t.get("relation", ""),
                    t.get("entity2", ""),
                    src,
                ))
            processed += len(batch)
            logger.info(f"[KG] 已提取 {processed}/{len(texts)} 个块的实体关系")
        except Exception as e:
            logger.warning(f"[KG] 批量提取失败：{e}")
            continue

    logger.info(f"[KG] 共提取 {len(all_triples)} 个三元组")
    return all_triples


def _parse_triples(text: str) -> List[dict]:
    """解析 LLM 返回的三元组 JSON"""
    text = text.strip()
    # 尝试直接解析
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # 尝试从 markdown 代码块中提取
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


# ──────────────────────────────────────────
# 图谱构建（供 ingest / build_index 调用）
# ──────────────────────────────────────────

def build_graph_from_chunks(
    chunks: list,
    llm,
    persist_dir: str = None,
) -> KnowledgeGraph:
    """从分块后的文档构建知识图谱"""
    if persist_dir is None:
        persist_dir = config.KG_PERSIST_DIR

    kg = KnowledgeGraph()

    texts = [(c.page_content, c.metadata.get("source_file", "")) for c in chunks]
    triples = extract_triples_batch(texts, llm)

    for entity1, relation, entity2, src in triples:
        if entity1 and entity2 and relation:
            kg.add_edge(entity1, entity2, relation, source_file=src)

    os.makedirs(persist_dir, exist_ok=True)
    graph_path = os.path.join(persist_dir, "graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(kg.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"[KG] 图谱已保存：{graph_path}（{kg.stats()}）")

    return kg


# ──────────────────────────────────────────
# 图谱加载与检索（供 rag_engine.py 调用）
# ──────────────────────────────────────────

def load_graph(persist_dir: str = None) -> Optional[KnowledgeGraph]:
    """加载已持久化的知识图谱"""
    if persist_dir is None:
        persist_dir = config.KG_PERSIST_DIR

    graph_path = os.path.join(persist_dir, "graph.json")
    if not os.path.exists(graph_path):
        return None

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return KnowledgeGraph.from_dict(data)
    except Exception as e:
        logger.warning(f"[KG] 图谱加载失败：{e}")
        return None


def retrieve_graph_context(
    kg: KnowledgeGraph,
    entities: List[str],
    max_depth: int = None,
    max_triples: int = 20,
) -> str:
    """检索图谱上下文：实体子图 → 文本描述"""
    if max_depth is None:
        max_depth = config.KG_RETRIEVAL_DEPTH
    if not entities:
        return ""
    subgraph = kg.subgraph_around(entities, max_depth=max_depth)
    return kg.to_context(subgraph, max_triples=max_triples)
