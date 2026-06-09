"""
研知通 - 索引构建工具
在命令行运行以构建或重建向量索引，避免阻塞 Streamlit
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
from ingest import ingest


def main():
    force = "--force" in sys.argv or "-f" in sys.argv

    logger.info("=" * 60)
    logger.info("研知通 - 向量索引构建")
    logger.info("=" * 60)
    logger.info(f"知识库路径: {config.KB_ROOT}")
    logger.info(f"索引存储: {config.CHROMA_PERSIST_DIR}")
    logger.info(f"Embedding 模式: {config.EMBEDDING_MODE}")

    try:
        vectorstore = ingest(force_rebuild=force)
        if vectorstore is None:
            logger.error("构建失败：未找到文档，请检查 KB_ROOT 路径")
            sys.exit(1)

        count = vectorstore._collection.count()
        logger.info(f"构建完成！共 {count} 个向量块")
        logger.info(f"索引已保存至: {config.CHROMA_PERSIST_DIR}")
    except Exception as e:
        logger.error(f"构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
