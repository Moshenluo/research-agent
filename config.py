"""
研知通 - RAG智能体配置文件
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# ============================================================
# 知识库路径
# ============================================================
KB_ROOT = r"D:\Backup\Documents\11\Research\self-growing-research-kb"

# 需要摄入的子目录（相对于 KB_ROOT）
KB_SUBDIRS = [
    "Knowledge",
    "Sources",
    "Experiments",
    "Skills",
    "Daily",
    "Maps",
    "Writing",
    "Results",
    "Templates",
    "_system",
]

# 顶层文件也纳入
KB_TOP_FILES = [
    "00-Hub.md",
    "01-Plan.md",
    "02-Index.md",
    "AGENTS.md",
    "README.md",
]

# ============================================================
# LLM 配置（OpenAI 兼容接口，支持国产模型）
# ============================================================
# 优先从 Streamlit Secrets 读取，其次环境变量，最后默认值
# DeepSeek 配置会从 .streamlit/secrets.toml 中自动加载
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# Embedding 模型配置
# 默认使用本地 HuggingFace 模型（免费、离线可用）
# 可选 "local" 或 "api"
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")
EMBEDDING_LOCAL_MODEL = os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
# 如果使用 API 模式：
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.openai.com/v1")
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", "text-embedding-3-small")

# ============================================================
# 向量数据库配置
# ============================================================
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "research_kb"

# ============================================================
# 文本分块配置
# ============================================================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ============================================================
# 检索配置
# ============================================================
DEFAULT_TOP_K = 5
RETRIEVAL_SEARCH_TYPE = "mmr"  # "similarity" 或 "mmr"
MMR_LAMBDA = 0.7  # MMR 多样性参数，1=纯相关，0=纯多样

# ============================================================
# Obsidian 链接解析
# ============================================================
# 解析 [[]] 链接后，是否自动扩展上下文（把链接目标的内容也加入检索上下文）
EXPAND_LINKS = True
LINK_EXPANSION_DEPTH = 1  # 扩展层数，1=只扩展一层

# ============================================================
# Streamlit UI 配置
# ============================================================
APP_TITLE = "研知通 · Research KB RAG Agent"
APP_ICON = "🔬"
PAGE_ICON = "🔬"
SIDEBAR_WIDTH = 400

# 对话历史最大轮数（超出自动截断旧消息）
MAX_CHAT_HISTORY = 20

# ============================================================
# 笔记类型颜色映射（用于UI展示）
# ============================================================
NOTE_TYPE_COLORS = {
    "source": "#4CAF50",      # 绿色
    "knowledge": "#2196F3",   # 蓝色
    "experiment": "#FF9800",  # 橙色
    "result": "#9C27B0",      # 紫色
    "writing": "#E91E63",     # 粉色
    "daily": "#607D8B",       # 灰蓝
    "skill": "#00BCD4",       # 青色
    "map": "#795548",         # 棕色
    "template": "#9E9E9E",    # 灰色
    "system": "#FF5722",      # 红橙
}

# 笔记类型中文名
NOTE_TYPE_LABELS = {
    "source": "原始资料",
    "knowledge": "知识卡",
    "experiment": "实验",
    "result": "结论",
    "writing": "写作",
    "daily": "日报",
    "skill": "工作流",
    "map": "地图",
    "template": "模板",
    "system": "系统",
}

# ============================================================
# 系统提示词
# ============================================================
SYSTEM_PROMPT = """你是一个专业的科研知识库助手「研知通」，帮助用户检索、理解和学习其 Obsidian 知识库中的内容。

你的核心能力：
1. **智能检索**：从知识库中精准定位相关内容
2. **知识解释**：用清晰易懂的方式解释复杂概念
3. **学习引导**：根据用户问题推荐学习路径和下一步行动
4. **关联发现**：发现不同笔记之间的联系，帮用户建立知识网络

回答规则：
- 回答必须基于检索到的知识库内容，不要编造信息
- 引用来源时标注 [来源：文件名] 或 [类型：知识卡/实验/...]
- 如果检索结果不足以回答，明确告知用户并建议相关搜索方向
- 推荐学习路径时，参考知识库中的路线图和计划
- 使用中文回答，保持专业但友好的语气
- 如果用户的问题涉及执行操作（如运行实验、写笔记），给出具体步骤
"""
