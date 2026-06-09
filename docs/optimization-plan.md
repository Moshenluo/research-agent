# Research RAG Agent 优化方案

> **执行方式：** 直接执行

**目标：** 修复已知 bug、提升性能、改善代码质量

**架构：** 保持现有模块结构（config/ingest/rag_engine/app），优化内部实现

**技术栈：** Python, Streamlit, LangChain, ChromaDB

---

## 优先级分类

### P0 - 必须修复的 Bug
1. 删除死代码（ConversationalRetrievalChain）
2. 修复快捷提问功能
3. 移除 ingest.py 中无用的 streamlit import
4. 修复 tags 过滤逻辑
5. 修复 search_only 中的 relevance_score 问题

### P1 - 性能优化
6. 优化 resolve_wikilink（构建文件名索引缓存）
7. 添加增量索引（基于文件 hash）

### P2 - 代码质量
8. 替换 print 为 logging
9. 支持 .env 配置文件
10. 添加流式输出

---

## Task 1: 删除死代码

**Files:**
- Modify: `rag_engine.py:188-226`

**问题：** `_create_chain()` 创建了 `ConversationalRetrievalChain`，但 `query()` 方法从未使用它，而是手动拼 prompt。

**修改：**
- 删除 `_create_chain()` 方法
- 删除 `self.chain` 属性
- 保留 `self.memory`（用于对话历史）

---

## Task 2: 修复快捷提问 Bug

**Files:**
- Modify: `app.py:379-387`

**问题：** 快捷问题设置 `st.session_state.quick_question` 后，`st.chat_input` 返回的 `prompt` 是用户输入（None），快捷问题赋值后被删除，实际从未使用。

**修改：**
- 在渲染 chat input 前检查 `quick_question`
- 如果有快捷问题，直接使用它作为 prompt

---

## Task 3: 移除无用 import

**Files:**
- Modify: `ingest.py:17`

**问题：** `import streamlit as st` 完全未使用，导致 ingest 无法脱离 Streamlit 独立运行。

**修改：** 删除该行。

---

## Task 4: 修复 Tags 过滤

**Files:**
- Modify: `rag_engine.py:48-56`

**问题：** Tags 存为逗号分隔字符串，但 `$contains` 期望 list。且多标签只取第一个。

**修改：**
- 在 ingest 时将 tags 存为 list（ChromaDB 支持）
- 在检索时正确处理多标签过滤

---

## Task 5: 修复 relevance_score

**Files:**
- Modify: `rag_engine.py:322-339`

**问题：** `Document` 对象没有 `relevance_score` 属性，永远返回 None。

**修改：** 删除该字段，或从 ChromaDB 的 distance 计算相似度。

---

## Task 6: 优化 resolve_wikilink

**Files:**
- Modify: `ingest.py:104-126`, `ingest.py:179-206`

**问题：** 每次解析链接都全量扫描，O(n) per link。

**修改：**
- 在 `load_all_documents` 时构建文件名索引 `{stem: path}`
- 传递给 `resolve_wikilink` 使用

---

## Task 7: 增量索引

**Files:**
- Modify: `ingest.py:287-323`

**问题：** 每次重建全量索引。

**修改：**
- 计算文件 hash，存入 metadata
- 重建时比对 hash，只处理变更文件

---

## Task 8: 替换 print 为 logging

**Files:**
- Modify: `ingest.py`, `rag_engine.py`, `app.py`

**修改：**
- 添加 `logger = logging.getLogger(__name__)`
- 替换所有 `print()` 为 `logger.info/debug/warning/error()`

---

## Task 9: 支持 .env 配置

**Files:**
- Modify: `config.py`

**修改：**
- 添加 `python-dotenv` 依赖
- 在 config.py 中加载 `.env` 文件

---

## Task 10: 流式输出

**Files:**
- Modify: `rag_engine.py:228-320`, `app.py:395-428`

**修改：**
- 在 engine 中添加 `query_stream()` 方法
- 在 app 中使用 `st.write_stream()` 显示流式输出

---

## 执行顺序

1. Task 1-5 (Bug fixes)
2. Task 6-7 (Performance)
3. Task 8-10 (Quality)
