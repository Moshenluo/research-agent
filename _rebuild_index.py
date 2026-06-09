import os, time
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

start = time.time()

from ingest import load_all_documents, expand_links, split_documents, get_embeddings
from langchain_community.vectorstores import Chroma
import config

# 1. 加载文档
docs = load_all_documents()
print(f"[1/4] 加载了 {len(docs)} 个文档")

# 2. 链接扩展
expanded = expand_links(docs, config.KB_ROOT)
print(f"[2/4] 链接扩展完成")

# 3. 分块
chunks = split_documents(expanded)
print(f"[3/4] 分块完成，共 {len(chunks)} 个块")

# 4. 构建向量索引
embeddings = get_embeddings()
print(f"[4/4] Embedding 模型已加载，开始构建向量索引...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=config.CHROMA_PERSIST_DIR,
    collection_name=config.CHROMA_COLLECTION_NAME,
)
vectorstore.persist()

elapsed = time.time() - start
print(f"完成！耗时 {elapsed:.1f} 秒")
print(f"索引块数: {vectorstore._collection.count()}")
