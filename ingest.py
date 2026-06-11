"""
研知通 - 数据摄入模块
负责加载 Markdown 文件、解析 YAML frontmatter、提取 Obsidian 链接、分块并构建向量索引
"""

import os

# 禁用 ChromaDB 遥测，避免 posthog/tenacity 引发 RuntimeError
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import yaml
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# 文件 Hash 与 Manifest
# ──────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """计算文件的 MD5 hash"""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (FileNotFoundError, IOError):
        return ""


def load_manifest(persist_dir: str) -> Dict[str, str]:
    """加载文件 hash manifest"""
    manifest_path = os.path.join(persist_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_manifest(persist_dir: str, manifest: Dict[str, str]):
    """保存文件 hash manifest"""
    manifest_path = os.path.join(persist_dir, "manifest.json")
    os.makedirs(persist_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def get_embeddings():
    """获取 Embedding 模型实例，优先本地，回退 API"""
    mode = config.EMBEDDING_MODE

    if mode == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info(f"[Local] 使用本地 Embedding 模型：{config.EMBEDDING_LOCAL_MODEL}")
            return HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_LOCAL_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 1,
                },
            )
        except ImportError:
            logger.warning("HuggingFaceEmbeddings 未安装，回退到 API 模式")
            mode = "api"

    if mode == "api":
        from langchain_openai import OpenAIEmbeddings
        api_key = config.EMBEDDING_API_KEY or config.LLM_API_KEY
        api_base = config.EMBEDDING_API_BASE or config.LLM_API_BASE
        if not api_key:
            raise ValueError("Embedding API Key 未配置")
        logger.info(f"[API] 使用远程 Embedding API：{config.EMBEDDING_API_MODEL}")
        return OpenAIEmbeddings(
            model=config.EMBEDDING_API_MODEL,
            api_key=api_key,
            base_url=api_base,
        )

    raise ValueError(f"未知的 Embedding 模式：{mode}")


# ──────────────────────────────────────────
# YAML frontmatter 解析
# ──────────────────────────────────────────

FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> Tuple[dict, str]:
    """解析 YAML frontmatter，返回 (metadata_dict, body_text)"""
    match = FM_PATTERN.match(text)
    if not match:
        return {}, text

    fm_str = match.group(1)
    body = text[match.end():]

    try:
        meta = yaml.safe_load(fm_str) or {}
    except yaml.YAMLError:
        meta = {}

    return meta, body


# ──────────────────────────────────────────
# Obsidian 链接提取
# ──────────────────────────────────────────

WIKILINK_PATTERN = re.compile(
    r"\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]"
)


def extract_wikilinks(text: str) -> List[str]:
    """提取 [[target|alias]] 链接，返回 target 列表"""
    links = []
    for match in WIKILINK_PATTERN.finditer(text):
        target = match.group(1).strip()
        links.append(target)
    return links


def resolve_wikilink(target: str, kb_root: str, filename_index: Optional[dict] = None) -> Optional[str]:
    """将 Obsidian wikilink target 解析为实际文件路径"""
    kb = Path(kb_root)

    # 使用预构建的索引快速查找
    if filename_index:
        # 尝试直接匹配 stem
        if target in filename_index:
            return filename_index[target]
        # 尝试匹配带 .md 后缀
        if f"{target}.md" in filename_index:
            return filename_index[f"{target}.md"]
        return None

    # 回退：尝试直接匹配
    direct = kb / f"{target}.md"
    if direct.exists():
        return str(direct)

    return None


def build_filename_index(kb_root: str) -> dict:
    """构建文件名索引 {stem: full_path}，用于快速解析 wikilink"""
    index = {}
    kb = Path(kb_root)

    for md_file in kb.rglob("*.md"):
        # 跳过隐藏目录
        parts = md_file.relative_to(kb).parts
        if any(p.startswith(".") for p in parts):
            continue
        # 以 stem 为 key
        index[md_file.stem] = str(md_file)
        # 也以相对路径（不含后缀）为 key
        try:
            rel = md_file.relative_to(kb)
            index[str(rel.with_suffix(""))] = str(md_file)
        except ValueError:
            pass

    return index


# ──────────────────────────────────────────
# Markdown 文件加载
# ──────────────────────────────────────────

def load_markdown_file(filepath: str, kb_root: str, compute_hash: bool = True) -> Optional[Document]:
    """加载单个 Markdown 文件，解析元数据和链接"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return None

    if not text.strip():
        return None

    meta, body = parse_frontmatter(text)
    wikilinks = extract_wikilinks(text)

    # 构建元数据
    rel_path = str(Path(filepath).relative_to(kb_root))
    meta["source_file"] = rel_path
    meta["source_path"] = filepath
    meta["wikilinks"] = ",".join(wikilinks) if wikilinks else ""
    meta["filename"] = Path(filepath).stem

    # 计算文件 hash（用于增量索引）
    if compute_hash:
        meta["file_hash"] = compute_file_hash(filepath)

    # tags 保持为列表（ChromaDB 支持列表类型）
    if "tags" in meta and isinstance(meta["tags"], list):
        meta["tags"] = [str(t) for t in meta["tags"]]
    elif "tags" in meta and isinstance(meta["tags"], str):
        meta["tags"] = [meta["tags"]]

    # 从路径推断类型
    parts = Path(rel_path).parts
    if len(parts) > 1:
        folder = parts[0].lower()
        type_map = {
            "knowledge": "knowledge",
            "sources": "source",
            "experiments": "experiment",
            "results": "result",
            "writing": "writing",
            "daily": "daily",
            "skills": "skill",
            "maps": "map",
            "templates": "template",
            "_system": "system",
        }
        meta.setdefault("type", type_map.get(folder, "unknown"))

    return Document(page_content=body, metadata=meta)


def load_all_documents(kb_root: str = config.KB_ROOT) -> List[Document]:
    """加载知识库中所有 Markdown 文件"""
    docs = []
    kb = Path(kb_root)

    # 加载顶层文件
    for filename in config.KB_TOP_FILES:
        filepath = kb / filename
        if filepath.exists():
            doc = load_markdown_file(str(filepath), kb_root)
            if doc:
                docs.append(doc)

    # 加载子目录文件
    for subdir in config.KB_SUBDIRS:
        subdir_path = kb / subdir
        if not subdir_path.exists():
            continue
        for md_file in subdir_path.rglob("*.md"):
            # 跳过隐藏目录
            parts = md_file.relative_to(kb).parts
            if any(p.startswith(".") for p in parts):
                continue
            doc = load_markdown_file(str(md_file), kb_root)
            if doc:
                docs.append(doc)

    return docs


# ──────────────────────────────────────────
# 链接扩展（把链接目标的摘要加入上下文）
# ──────────────────────────────────────────

def expand_links(docs: List[Document], kb_root: str) -> List[Document]:
    """对每个文档，将其链接目标的前 N 个字符附加到内容中"""
    if not config.EXPAND_LINKS:
        return docs

    # 构建文件名索引（用于快速解析 wikilink）
    filename_index = build_filename_index(kb_root)

    # 构建文件名到文档的映射
    doc_map = {}
    for doc in docs:
        fname = doc.metadata.get("filename", "")
        rel_path = doc.metadata.get("source_file", "")
        doc_map[fname] = doc
        doc_map[rel_path] = doc

    expanded_docs = []
    for doc in docs:
        links_str = doc.metadata.get("wikilinks", "")
        links = links_str.split(",") if links_str else []
        link_summaries = []

        for link_target in links:
            link_target = link_target.strip()
            if not link_target:
                continue
            # 尝试从已加载文档中查找
            target_doc = doc_map.get(link_target)
            if not target_doc:
                # 使用索引快速解析路径
                resolved = resolve_wikilink(link_target, kb_root, filename_index)
                if resolved:
                    target_name = Path(resolved).stem
                    target_doc = doc_map.get(target_name)

            if target_doc and target_doc != doc:
                # 附加链接目标的摘要（前200字）
                summary = target_doc.page_content[:200].strip()
                if summary:
                    link_summaries.append(
                        f"【关联：{target_doc.metadata.get('filename', link_target)}】{summary}..."
                    )

        if link_summaries:
            expanded_content = doc.page_content + "\n\n--- 关联内容 ---\n" + "\n".join(link_summaries)
            expanded_docs.append(Document(
                page_content=expanded_content,
                metadata=doc.metadata
            ))
        else:
            expanded_docs.append(doc)

    return expanded_docs


# ──────────────────────────────────────────
# 文本分块
# ──────────────────────────────────────────

def split_documents(docs: List[Document]) -> List[Document]:
    """将文档分块，保留元数据"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "；", "，", " "],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    # 为每个块添加块序号
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


# ──────────────────────────────────────────
# 向量索引构建
# ──────────────────────────────────────────

def build_vectorstore(
    docs: List[Document],
    persist_dir: str = config.CHROMA_PERSIST_DIR,
    force_rebuild: bool = False,
) -> Chroma:
    """构建或加载 Chroma 向量数据库"""

    embeddings = get_embeddings()

    # 检查是否需要重建
    if not force_rebuild and os.path.exists(persist_dir):
        # 加载旧的 manifest
        old_manifest = load_manifest(persist_dir)

        # 构建当前文件的 hash manifest
        current_manifest = {}
        for doc in docs:
            source_file = doc.metadata.get("source_file", "")
            file_hash = doc.metadata.get("file_hash", "")
            if source_file and file_hash:
                current_manifest[source_file] = file_hash

        # 比较 manifest，检查是否有变化
        if old_manifest == current_manifest:
            logger.info(f"[Load] 文件无变化，加载已有向量索引：{persist_dir}")
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
        else:
            # 统计变化
            added = set(current_manifest.keys()) - set(old_manifest.keys())
            removed = set(old_manifest.keys()) - set(current_manifest.keys())
            changed = {
                k for k in set(current_manifest.keys()) & set(old_manifest.keys())
                if current_manifest[k] != old_manifest[k]
            }
            logger.info(f"[Change] 检测到文件变化：新增 {len(added)}，删除 {len(removed)}，修改 {len(changed)}")

    # 分块
    logger.info(f"[Load] 加载了 {len(docs)} 个文档")
    expanded = expand_links(docs, config.KB_ROOT)
    logger.info("[Link] 链接扩展完成")
    chunks = split_documents(expanded)
    logger.info(f"[Split] 分块完成，共 {len(chunks)} 个块")

    # 构建向量库（分批处理，避免内存溢出）
    logger.info(f"[Build] 构建向量索引（模型：{config.EMBEDDING_LOCAL_MODEL if config.EMBEDDING_MODE == 'local' else config.EMBEDDING_API_MODEL}）...")

    # 如果存在旧索引，先删除
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir, ignore_errors=True)

    # 分批添加到 ChromaDB
    INDEX_BATCH_SIZE = 10
    vectorstore = None
    for i in range(0, len(chunks), INDEX_BATCH_SIZE):
        batch = chunks[i:i + INDEX_BATCH_SIZE]
        logger.info(f"[Build] 处理批次 {i // INDEX_BATCH_SIZE + 1}/{(len(chunks) - 1) // INDEX_BATCH_SIZE + 1}（{len(batch)} 个块）...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)
        vectorstore.persist()

    # 保存 manifest
    new_manifest = {}
    for doc in docs:
        source_file = doc.metadata.get("source_file", "")
        file_hash = doc.metadata.get("file_hash", "")
        if source_file and file_hash:
            new_manifest[source_file] = file_hash
    save_manifest(persist_dir, new_manifest)

    logger.info(f"[Done] 向量索引已保存至：{persist_dir}")

    return vectorstore


# ──────────────────────────────────────────
# 增量更新
# ──────────────────────────────────────────

def incremental_update(persist_dir: str = config.CHROMA_PERSIST_DIR):
    """增量更新索引：只处理变更的文件"""
    embeddings = get_embeddings()

    # 加载已有向量库
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )

    # 加载 manifest
    old_manifest = load_manifest(persist_dir)
    if not old_manifest:
        logger.warning("Manifest 不存在，回退到全量重建")
        return None

    # 扫描当前文件
    kb = Path(config.KB_ROOT)
    if not kb.exists():
        logger.error(f"知识库路径不存在：{config.KB_ROOT}")
        return vectorstore

    current_manifest = {}
    all_file_paths = set()

    # 顶层文件
    for filename in config.KB_TOP_FILES:
        fp = kb / filename
        if fp.exists():
            current_manifest[filename] = compute_file_hash(str(fp))
            all_file_paths.add(str(fp))

    # 子目录文件
    for subdir in config.KB_SUBDIRS:
        subpath = kb / subdir
        if not subpath.exists():
            continue
        for md_file in subpath.rglob("*.md"):
            parts = md_file.relative_to(kb).parts
            if any(p.startswith(".") for p in parts):
                continue
            rel = str(md_file.relative_to(kb))
            current_manifest[rel] = compute_file_hash(str(md_file))
            all_file_paths.add(str(md_file))

    added = set(current_manifest.keys()) - set(old_manifest.keys())
    removed = set(old_manifest.keys()) - set(current_manifest.keys())
    changed = {
        k for k in set(current_manifest.keys()) & set(old_manifest.keys())
        if current_manifest[k] != old_manifest[k]
    }

    if not added and not removed and not changed:
        logger.info("[Update] 知识库无变更")
        return vectorstore

    logger.info(f"[Update] 变更检测：新增 {len(added)}，删除 {len(removed)}，修改 {len(changed)}")

    # 删除已移除的文件对应的 chunks
    ids_to_delete = []
    if removed:
        collection = vectorstore._collection
        all_items = collection.get(include=["metadatas"])
        for i, meta in enumerate(all_items.get("metadatas", [])):
            if meta and meta.get("source_file", "") in removed:
                ids_to_delete.append(all_items["ids"][i])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(f"[Update] 删除 {len(ids_to_delete)} 个过期块")

    # 处理新增和修改的文件
    to_process = added | changed
    if to_process:
        new_docs = []
        for rel_path in to_process:
            filepath = str(kb / rel_path)
            if not os.path.exists(filepath):
                continue
            doc = load_markdown_file(filepath, config.KB_ROOT, compute_hash=True)
            if doc:
                new_docs.append(doc)

        if new_docs:
            # 先删除旧版本
            if changed:
                collection = vectorstore._collection
                all_items = collection.get(include=["metadatas"])
                ids_to_del = []
                for i, meta in enumerate(all_items.get("metadatas", [])):
                    if meta and meta.get("source_file", "") in changed:
                        ids_to_del.append(all_items["ids"][i])
                if ids_to_del:
                    collection.delete(ids=ids_to_del)

            # 扩展链接、分块
            expanded = expand_links(new_docs, config.KB_ROOT)
            chunks = split_documents(expanded)
            logger.info(f"[Update] 处理 {len(new_docs)} 个文件，生成 {len(chunks)} 个块")

            # 分批添加
            BATCH = 10
            for i in range(0, len(chunks), BATCH):
                batch = chunks[i:i + BATCH]
                vectorstore.add_documents(batch)
                if (i // BATCH + 1) % 5 == 0:
                    logger.info(f"[Update] 批次 {i // BATCH + 1}/{ (len(chunks) - 1) // BATCH + 1}")

    # 保存新 manifest
    save_manifest(persist_dir, current_manifest)
    logger.info(f"[Update] 增量更新完成，共 {vectorstore._collection.count()} 个块")

    return vectorstore


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────

def ingest(force_rebuild: bool = False):
    """完整的数据摄入流程"""
    logger.info("=" * 60)
    logger.info("[研知通] 数据摄入")
    logger.info("=" * 60)
    logger.info(f"[Path] 知识库路径：{config.KB_ROOT}")

    # 1. 加载文档
    docs = load_all_documents()
    logger.info(f"[Load] 加载了 {len(docs)} 个文档")

    if not docs:
        logger.error("未找到任何文档，请检查路径配置")
        return None

    # 2. 构建向量库
    vectorstore = build_vectorstore(docs, force_rebuild=force_rebuild)
    logger.info("[Done] 数据摄入完成！")

    return vectorstore


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args()
    ingest(force_rebuild=args.force_rebuild)
