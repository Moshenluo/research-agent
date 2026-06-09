import os
import time

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

start = time.time()

# 1. 加载文档
print("[1/5] 正在加载文档...")
from ingest import load_all_documents
docs = load_all_documents()
print(f"  -> 加载了 {len(docs)} 个文档")

# 2. 链接扩展
print("[2/5] 链接扩展...")
from ingest import expand_links
import config
expanded = expand_links(docs, config.KB_ROOT)
print(f"  -> 完成")

# 3. 分块
print("[3/5] 文本分块...")
from ingest import split_documents
chunks = split_documents(expanded)
print(f"  -> 分块完成，共 {len(chunks)} 个块")

# 4. 加载 Embedding 模型
print("[4/5] 加载 Embedding 模型...")
from ingest import get_embeddings
embeddings = get_embeddings()
print(f"  -> 模型加载完成")

# 5. 构建向量索引
print("[5/5] 构建向量索引...")
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=config.CHROMA_PERSIST_DIR,
    collection_name=config.CHROMA_COLLECTION_NAME,
)
vectorstore.persist()

elapsed = time.time() - start
print(f"\n✅ 完成！耗时 {elapsed:.1f} 秒")
print(f"索引块数: {vectorstore._collection.count()}")
