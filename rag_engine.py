"""
研知通 - RAG 引擎核心
负责检索增强生成的完整流程：检索 → 重排 → 生成
"""

import os

# 禁用 ChromaDB 遥测，避免 posthog/tenacity 引发 RuntimeError
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# 检索器
# ──────────────────────────────────────────

class ResearchRetriever:
    """知识库检索器，支持向量检索 + 元数据过滤"""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search_type: str = config.RETRIEVAL_SEARCH_TYPE,
    ) -> List[Document]:
        """
        语义检索
        - query: 查询文本
        - top_k: 返回数量
        - note_type: 按笔记类型过滤（knowledge/experiment/daily/...）
        - tags: 按标签过滤
        """
        # 构建过滤条件
        filter_dict = {}
        if note_type:
            filter_dict["type"] = note_type
        if tags:
            # 多标签使用 $and 组合 $contains
            if len(tags) == 1:
                filter_dict["tags"] = {"$contains": tags[0]}
            else:
                filter_dict["$and"] = [{"tags": {"$contains": tag}} for tag in tags]

        # 执行检索
        if search_type == "mmr":
            results = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=top_k,
                lambda_param=config.MMR_LAMBDA,
                filter=filter_dict if filter_dict else None,
            )
        else:
            results = self.vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter=filter_dict if filter_dict else None,
            )

        return results

    def search_by_type(self, query: str, note_type: str, top_k: int = 3) -> List[Document]:
        """按类型检索的快捷方法"""
        return self.search(query, top_k=top_k, note_type=note_type)

    def get_knowledge(self, query: str, top_k: int = 3) -> List[Document]:
        """检索知识卡"""
        return self.search_by_type(query, "knowledge", top_k)

    def get_experiments(self, query: str, top_k: int = 3) -> List[Document]:
        """检索实验"""
        return self.search_by_type(query, "experiment", top_k)

    def get_daily(self, query: str, top_k: int = 3) -> List[Document]:
        """检索日报"""
        return self.search_by_type(query, "daily", top_k)

    def get_skills(self, query: str, top_k: int = 3) -> List[Document]:
        """检索工作流"""
        return self.search_by_type(query, "skill", top_k)

    def get_sources(self, query: str, top_k: int = 3) -> List[Document]:
        """检索原始资料"""
        return self.search_by_type(query, "source", top_k)


# ──────────────────────────────────────────
# 意图识别
# ──────────────────────────────────────────

def classify_intent(query: str) -> str:
    """
    简单的意图分类，决定检索策略
    - learn: 学习某主题 → 优先 Knowledge + Sources
    - experiment: 做实验 → 优先 Experiments + Skills
    - decision: 决策推理 → 优先 Daily + Knowledge
    - workflow: 工作流/怎么做 → 优先 Skills + Templates
    - explore: 探索/创意 → 优先 Knowledge + Maps
    - general: 通用 → 全量检索
    """
    learn_keywords = ["怎么学", "学习路线", "学习地图", "怎么理解", "什么是", "解释", "概念"]
    experiment_keywords = ["实验", "复现", "训练", "微调", "跑通", "smoke test"]
    decision_keywords = ["为什么", "决策", "选择", "入库", "评分", "候选"]
    workflow_keywords = ["怎么做", "流程", "工作流", "Skill", "模板", "步骤"]
    explore_keywords = ["有什么", "创意", "idea", "想法", "灵感", "推荐"]

    query_lower = query.lower()

    for kw in experiment_keywords:
        if kw in query_lower:
            return "experiment"
    for kw in workflow_keywords:
        if kw in query_lower:
            return "workflow"
    for kw in decision_keywords:
        if kw in query_lower:
            return "decision"
    for kw in learn_keywords:
        if kw in query_lower:
            return "learn"
    for kw in explore_keywords:
        if kw in query_lower:
            return "explore"

    return "general"


INTENT_RETRIEVAL_STRATEGY = {
    "learn": [("knowledge", 3), ("source", 2), ("daily", 1)],
    "experiment": [("experiment", 3), ("skill", 2), ("knowledge", 1)],
    "decision": [("daily", 3), ("knowledge", 2)],
    "workflow": [("skill", 3), ("template", 2), ("knowledge", 1)],
    "explore": [("knowledge", 3), ("map", 1), ("source", 1)],
    "general": None,  # 全量检索
}


# ──────────────────────────────────────────
# RAG 引擎
# ──────────────────────────────────────────

class ResearchRAGEngine:
    """RAG 主引擎"""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.retriever = ResearchRetriever(vectorstore)
        self.llm = None
        self.memory = None
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM（按需调用，避免未配置 API Key 时失败）"""
        if not config.LLM_API_KEY:
            return
        try:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_API_BASE,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )
            self.memory = ConversationBufferWindowMemory(
                k=config.MAX_CHAT_HISTORY,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
        except Exception as e:
            logger.warning(f"LLM 初始化失败：{e}")

    def _retrieve(self, question: str, use_intent: bool = True):
        """检索 + 构建上下文，返回 (source_docs, intent, context, prompt_text)"""
        intent = classify_intent(question) if use_intent else "general"

        if use_intent and INTENT_RETRIEVAL_STRATEGY[intent] is not None:
            strategy = INTENT_RETRIEVAL_STRATEGY[intent]
            all_docs = []
            for note_type, k in strategy:
                docs = self.retriever.search_by_type(question, note_type, top_k=k)
                all_docs.extend(docs)
            seen = set()
            unique_docs = []
            for doc in all_docs:
                key = f"{doc.metadata.get('source_file', '')}_{doc.metadata.get('chunk_index', '')}"
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)
            source_docs = unique_docs[:config.DEFAULT_TOP_K * 2]
        else:
            source_docs = self.retriever.search(question)

        context_parts = []
        for doc in source_docs:
            src = doc.metadata.get("source_file", "未知")
            note_type = doc.metadata.get("type", "unknown")
            type_label = config.NOTE_TYPE_LABELS.get(note_type, note_type)
            context_parts.append(f"[{type_label}：{src}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        prompt_text = f"""{config.SYSTEM_PROMPT}

以下是从知识库中检索到的相关内容：

{context}

---

根据以上知识库内容回答用户问题。如果知识库中没有相关信息，请如实说明并建议搜索方向。

问题：{question}

回答（引用来源时标注 [来源：文件名]）："""

        return source_docs, intent, prompt_text

    def _build_sources(self, source_docs: List[Document]) -> List[dict]:
        sources = []
        for doc in source_docs:
            sources.append({
                "file": doc.metadata.get("source_file", ""),
                "type": doc.metadata.get("type", ""),
                "type_label": config.NOTE_TYPE_LABELS.get(
                    doc.metadata.get("type", ""), ""
                ),
                "filename": doc.metadata.get("filename", ""),
                "tags": doc.metadata.get("tags", []),
                "status": doc.metadata.get("status", ""),
                "content_preview": doc.page_content[:150] + "...",
            })
        return sources

    def query(self, question: str, use_intent: bool = True) -> dict:
        """
        执行 RAG 查询
        返回: {"answer": str, "sources": List[dict], "intent": str}
        """
        if not self.llm:
            return {
                "answer": "⚠️ LLM 未配置，请先配置 API Key",
                "sources": [],
                "intent": "error",
            }

        source_docs, intent, prompt_text = self._retrieve(question, use_intent)

        response = self.llm.invoke(prompt_text)
        answer = response.content

        sources = self._build_sources(source_docs)

        self.memory.save_context(
            {"question": question},
            {"answer": answer},
        )

        return {
            "answer": answer,
            "sources": sources,
            "intent": intent,
        }

    def query_stream(self, question: str, use_intent: bool = True):
        """
        流式 RAG 查询，yield 每个 token chunk
        最后一个 yield 包含 sources 和 intent
        """
        if not self.llm:
            yield "⚠️ LLM 未配置，请先配置 API Key"
            return

        source_docs, intent, prompt_text = self._retrieve(question, use_intent)
        sources = self._build_sources(source_docs)

        full_answer = []
        for chunk in self.llm.stream(prompt_text):
            token = chunk.content
            if token:
                full_answer.append(token)
                yield token

        answer = "".join(full_answer)
        self.memory.save_context(
            {"question": question},
            {"answer": answer},
        )

        yield {"__meta__": {"sources": sources, "intent": intent}}

    def search_only(self, query: str, top_k: int = 10, note_type: Optional[str] = None) -> List[dict]:
        """纯检索模式，不调用 LLM"""
        docs = self.retriever.search(query, top_k=top_k, note_type=note_type)
        results = []
        for doc in docs:
            results.append({
                "file": doc.metadata.get("source_file", ""),
                "type": doc.metadata.get("type", ""),
                "type_label": config.NOTE_TYPE_LABELS.get(
                    doc.metadata.get("type", ""), ""
                ),
                "filename": doc.metadata.get("filename", ""),
                "tags": doc.metadata.get("tags", []),
                "status": doc.metadata.get("status", ""),
                "content": doc.page_content,
            })
        return results

    def clear_memory(self):
        """清空对话历史"""
        if self.memory:
            self.memory.clear()

    def get_kb_stats(self) -> dict:
        """获取知识库统计信息"""
        collection = self.vectorstore._collection
        count = collection.count()

        # 统计各类型数量
        type_counts = {}
        all_docs = self.vectorstore.get(include=["metadatas"])
        for meta in all_docs["metadatas"]:
            t = meta.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_chunks": count,
            "type_counts": type_counts,
        }


# ──────────────────────────────────────────
# 初始化引擎
# ──────────────────────────────────────────

def init_engine(force_rebuild: bool = False) -> ResearchRAGEngine:
    """初始化 RAG 引擎（含向量库构建）"""
    from ingest import ingest

    vectorstore = ingest(force_rebuild=force_rebuild)
    if vectorstore is None:
        raise RuntimeError("向量库构建失败，请检查知识库路径和 API Key 配置")

    return ResearchRAGEngine(vectorstore)


def update_engine_incremental() -> ResearchRAGEngine:
    """增量更新引擎，只处理变更文件"""
    from ingest import incremental_update

    vectorstore = incremental_update()
    if vectorstore is None:
        raise RuntimeError("增量更新失败，请运行 build_index.py 全量重建")

    return ResearchRAGEngine(vectorstore)


def load_engine() -> Optional[ResearchRAGEngine]:
    """加载已有向量库（不重新构建）"""
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        return None

    try:
        from ingest import get_embeddings
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )
        return ResearchRAGEngine(vectorstore)
    except Exception as e:
        logger.error(f"加载引擎失败：{e}")
        return None
