# 研知通 · Research KB RAG Agent — 技术文档

## 概述

研知通是一个基于 **RAG（检索增强生成）** 技术的科研知识库智能助手，连接 Obsidian 知识库，提供语义搜索和对话问答功能。

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **前端** | Streamlit 1.40+ | Web UI 框架，对话/搜索/浏览界面 |
| **LLM** | LangChain + OpenAI 兼容 API | ChatOpenAI 调用 DeepSeek / Qwen / GPT-4o 等 |
| **向量数据库** | ChromaDB | 存储文档块的向量表示，执行相似度检索 |
| **Embedding** | HuggingFace Sentence-Transformers | 本地 BAAI/bge-small-zh-v1.5 模型（512 维） |
| **文档解析** | PyYAML + 正则表达式 | 解析 Obsidian Markdown 的 YAML Frontmatter |
| **文本分块** | LangChain RecursiveCharacterTextSplitter | 递归分块，保留语义边界 |
| **向量检索** | LangChain Chroma Wrapper | MMR（最大边际相关性）检索 + 元数据过滤 |
| **增量索引** | MD5 Hash + JSON Manifest | 文件变更检测，差异化重建 |
| **配置** | python-dotenv | .env 文件 + 环境变量支持 |

## 架构

```
┌──────────────────────────────────────────────┐
│                  Streamlit UI                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 对话问答 │  │ 语义搜索 │  │ 知识浏览 │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
├───────┼──────────────┼─────────────┼─────────┤
│       ▼              ▼             ▼         │
│  ┌──────────────────────────────────────┐   │
│  │         ResearchRAGEngine            │   │
│  │  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │ 意图分类 │  │  分层检索策略     │ │   │
│  │  └──────────┘  └──────────────────┘ │   │
│  │  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │ LLM 调用 │  │  流式输出 Stream │ │   │
│  │  └──────────┘  └──────────────────┘ │   │
│  └──────────────┬───────────────────────┘   │
│                 │                             │
│  ┌──────────────▼───────────────────────┐   │
│  │        ResearchRetriever             │   │
│  │  MMR Search + Metadata Filter        │   │
│  └──────────────┬───────────────────────┘   │
├─────────────────┼───────────────────────────┤
│                 ▼                             │
│  ┌──────────────────────────────────────┐   │
│  │           ChromaDB                    │   │
│  │   Vector Storage + Similarity Search │   │
│  └──────────────┬───────────────────────┘   │
├─────────────────┼───────────────────────────┤
│                 ▼                             │
│  ┌──────────────────────────────────────┐   │
│  │         ingest.py (数据摄入)          │   │
│  │  ┌─────────┐  ┌──────┐  ┌────────┐  │   │
│  │  │YAML解析 │  │分块  │  │Embedding│  │   │
│  │  └─────────┘  └──────┘  └────────┘  │   │
│  │  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │链接扩展  │  │ 增量更新(Manifest)│  │   │
│  │  └──────────┘  └──────────────────┘  │   │
│  └──────────────┬───────────────────────┘   │
│                 ▼                             │
│  ┌──────────────────────────────────────┐   │
│  │   D:\...\self-growing-research-kb    │   │
│  │   Obsidian Knowledge Base (.md)      │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

## 核心流程

### 1. 数据摄入（ingest.py）

```
Markdown 文件
    │
    ▼
YAML Frontmatter 解析 ──→ 提取 tags, type, status 等元数据
    │
    ▼
Obsidian [[wikilink]] 提取 ──→ 链接扩展（附加上下文摘要）
    │
    ▼
RecursiveCharacterTextSplitter 分块 ──→ 256 字符/块，32 字符重叠
    │
    ▼
HuggingFace Embedding ──→ 512 维向量
    │
    ▼
ChromaDB 持久化存储
```

### 2. 检索（rag_engine.py）

```
用户问题
    │
    ▼
意图分类 ──→ learn / experiment / decision / workflow / explore / general
    │
    ▼
分层检索策略 ──→ 按意图配比不同类型笔记（Knowledge:3 + Sources:2 + ...）
    │
    ▼
MMR 检索 ──→ 最大边际相关性，避免冗余结果
    │
    ▼
上下文拼接 ──→ [类型：文件名] + 内容
    │
    ▼
LLM 生成回答 ──→ 流式输出
```

### 3. 增量更新（ingest.py: incremental_update）

```
启动/定时检查
    │
    ▼
扫描 KB 目录，计算所有文件 MD5 hash
    │
    ▼
对比 manifest.json 中的历史 hash
    │
    ├── 新增文件 → 加载、分块、embedding、添加到 ChromaDB
    ├── 修改文件 → 删除旧 chunks，重新处理
    ├── 删除文件 → 从 ChromaDB 删除对应 chunks
    └── 无变更   → 跳过
    │
    ▼
更新 manifest.json
```

## 意图分类策略

| 意图 | 触发词 | 检索策略 |
|------|--------|----------|
| learn（学习） | 怎么学、什么是、解释、概念 | Knowledge:3 + Source:2 + Daily:1 |
| experiment（实验） | 实验、复现、训练、微调 | Experiment:3 + Skill:2 + Knowledge:1 |
| decision（决策） | 为什么、决策、选择、入库 | Daily:3 + Knowledge:2 |
| workflow（流程） | 怎么做、流程、工作流、步骤 | Skill:3 + Template:2 + Knowledge:1 |
| explore（探索） | 有什么、创意、idea、灵感 | Knowledge:3 + Map:1 + Source:1 |
| general（通用） | 其他 | 全量 MMR 检索 |

## 笔记类型映射

| 目录 | 类型 | 用途 |
|------|------|------|
| Knowledge/ | knowledge | 知识卡 |
| Sources/ | source | 原始资料 |
| Experiments/ | experiment | 实验记录 |
| Skills/ | skill | 工作流 |
| Daily/ | daily | 日报 |
| Maps/ | map | 知识地图 |
| Writing/ | writing | 写作 |
| Results/ | result | 结论 |
| Templates/ | template | 模板 |
| _system/ | system | 系统配置 |

## 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| KB_ROOT | `D:\...\self-growing-research-kb` | 知识库路径 |
| LLM_MODEL | deepseek-chat | LLM 模型 |
| EMBEDDING_MODE | local | embedding 模式（local/api） |
| EMBEDDING_LOCAL_MODEL | BAAI/bge-small-zh-v1.5 | 本地 embedding 模型 |
| CHUNK_SIZE | 256 | 分块大小（字符） |
| DEFAULT_TOP_K | 5 | 默认检索数量 |
| RETRIEVAL_SEARCH_TYPE | mmr | 检索算法（mmr/similarity） |

## 关键文件

| 文件 | 职责 |
|------|------|
| `app.py` | Streamlit UI，自动初始化引擎，检测更新 |
| `config.py` | 全局配置 |
| `ingest.py` | 数据摄入、分块、索引构建、增量更新 |
| `rag_engine.py` | 检索器、意图分类、RAG 引擎 |
| `build_index.py` | CLI 索引构建工具 |
