import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
from ingest import load_all_documents, expand_links, split_documents, get_embeddings
import config

print("[1/4] 加载文档...")
docs = load_all_documents()
print(f"  -> {len(docs)} 个文档")

print("[2/4] 链接扩展...")
expanded = expand_links(docs, config.KB_ROOT)

print("[3/4] 分块...")
chunks = split_documents(expanded)
print(f"  -> {len(chunks)} 个块")

print("[4/4] 构建向量索引...")
embeddings = get_embeddings()

# 用 chromadb 原生 PersistentClient
client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

try:
    client.delete_collection(config.CHROMA_COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=config.CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "l2"},
)

# 分批添加（避免内存问题）
batch_size = 50
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    texts = [c.page_content for c in batch]
    metas = [c.metadata for c in batch]
    ids = [f"chunk_{i+j}" for j in range(len(batch))]
    
    # 计算 embeddings
    vectors = embeddings.embed_documents(texts)
    
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metas,
    )
    print(f"  -> 已添加 {i+len(batch)}/{len(chunks)}")

print(f"\n✅ 完成！Collection count: {collection.count()}")
