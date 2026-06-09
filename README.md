# 🔬 研知通 · Research KB RAG Agent

你的 Obsidian 科研知识库智能助手 — 搜索、检索、对话学习。

## 功能

- **💬 对话问答**：基于 RAG 的智能问答，自动识别意图、分层检索、带来源引用
- **🔍 语义搜索**：直接检索知识库内容，按类型过滤
- **📊 知识库浏览**：可视化查看知识库统计和内容

## 特色

- 📂 自动解析 Obsidian Markdown（YAML frontmatter + `[[]]` 双向链接）
- 🧠 意图识别：自动判断你的问题是学习/实验/决策/流程/探索，优化检索策略
- 🔗 链接扩展：检索时自动扩展 Obsidian 链接目标的上下文
- 🏷️ 元数据过滤：按笔记类型、标签过滤检索
- 📎 来源引用：每条回答都标注知识库来源文件

## 快速开始

### 1. 安装依赖

```bash
cd C:\Users\Administrator\Desktop\research-rag-agent
pip install -r requirements.txt
```

### 2. 配置 API Key

方式一：在 Streamlit 侧边栏的「⚙️ API 配置」中填写

方式二：设置环境变量

```bash
# OpenAI
set OPENAI_API_KEY=sk-xxx

# 或使用国产模型（以通义千问为例）
set OPENAI_API_KEY=sk-xxx
set OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
set LLM_MODEL=qwen-plus
set EMBEDDING_MODEL=text-embedding-v3
```

### 3. 启动

```bash
streamlit run app.py --server.port 8503
```

首次启动会自动构建向量索引（约 1-2 分钟），后续启动直接加载缓存。

### 4. 使用

- 在对话框输入问题，如「Transformer 怎么学？」
- 侧边栏有快捷提问按钮
- 切换到「语义搜索」模式可直接检索
- 点击「🔄 重建索引」重新构建向量库

## 项目结构

```
research-rag-agent/
├── app.py              # Streamlit 主界面
├── config.py           # 配置文件
├── ingest.py           # 数据摄入（加载MD/解析YAML/分块/向量索引）
├── rag_engine.py       # RAG 引擎（检索+意图识别+生成）
├── requirements.txt    # 依赖
├── chroma_db/          # 向量数据库缓存（自动生成）
└── README.md
```

## 支持的 LLM

| 提供商 | API Base | 模型示例 |
|--------|----------|---------|
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini, gpt-4o |
| 通义千问 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus, qwen-max |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
| 智谱 AI | https://open.bigmodel.cn/api/paas/v4 | glm-4-flash |
| 本地 Ollama | http://localhost:11434/v1 | qwen2.5:7b |

## 知识库路径

默认指向：`D:\Backup\Documents\11\Research\self-growing-research-kb`

如需修改，编辑 `config.py` 中的 `KB_ROOT`。
