"""
研知通 - 索引构建工具
用法：
  python build_index.py          增量更新（默认）
  python build_index.py --force   强制全量重建
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


def main():
    force = "--force" in sys.argv or "-f" in sys.argv

    logger.info("=" * 60)
    logger.info("研知通 - 向量索引构建")
    logger.info("=" * 60)
    logger.info(f"知识库: {config.KB_ROOT}")
    logger.info(f"索引: {config.CHROMA_PERSIST_DIR}")
    logger.info(f"模式: {'全量重建' if force else '增量更新'}")
    logger.info(f"Embedding: {config.EMBEDDING_MODE}")

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
        logger.info(f"✓ 完成！共 {count} 个向量块")
    except Exception as e:
        logger.error(f"构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
