"""
研知通 - 索引构建工具
用法：
  python build_index.py              增量更新（默认）
  python build_index.py --force       强制全量重建
  python build_index.py --graph       构建/重建知识图谱
  python build_index.py --force --graph  全量重建索引 + 图谱
"""
import os
import sys
import logging

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import config


def build_graph():
    """构建知识图谱"""
    from langchain_openai import ChatOpenAI
    from ingest import load_all_documents, split_documents, expand_links
    from graph import build_graph_from_chunks

    if not config.LLM_API_KEY:
        logger.warning("LLM API Key 未配置，跳过高风险构建")
        return

    logger.info("正在构建知识图谱...")
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0.1,
        max_tokens=1024,
    )

    docs = load_all_documents(config.KB_ROOT)
    expanded = expand_links(docs, config.KB_ROOT)
    chunks = split_documents(expanded)
    logger.info(f"共 {len(docs)} 个文档，{len(chunks)} 个块")

    kg = build_graph_from_chunks(chunks, llm)
    logger.info(f"知识图谱构建完成：{kg.stats()}")
    return kg


def main():
    force = "--force" in sys.argv or "-f" in sys.argv
    build_kg = "--graph" in sys.argv or "-g" in sys.argv

    logger.info("=" * 60)
    logger.info("研知通 - 向量索引构建")
    logger.info("=" * 60)
    logger.info(f"知识库: {config.KB_ROOT}")
    logger.info(f"索引: {config.CHROMA_PERSIST_DIR}")
    logger.info(f"模式: {'全量重建' if force else '增量更新'}")
    logger.info(f"Embedding: {config.EMBEDDING_MODE}")
    if build_kg:
        logger.info("知识图谱: 构建")

    try:
        if force or not os.path.exists(config.CHROMA_PERSIST_DIR):
            from ingest import ingest
            vectorstore = ingest(force_rebuild=force)
        else:
            from ingest import incremental_update
            vectorstore = incremental_update()

        if vectorstore is None:
            logger.error("构建失败")
            sys.exit(1)

        count = vectorstore._collection.count()
        logger.info(f"完成！共 {count} 个向量块")

        if build_kg:
            build_graph()
    except Exception as e:
        logger.error(f"构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
