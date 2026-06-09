"""
研知通 · Research KB RAG Agent
基于 Streamlit 的交互式 RAG 智能体
"""

import os
import traceback

# 禁用 ChromaDB 遥测，避免 posthog/tenacity 引发 RuntimeError
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st
from pathlib import Path

import config

# ──────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
# 自定义 CSS
# ──────────────────────────────────────────

CUSTOM_CSS = """
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .app-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .app-subtitle { text-align: center; color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .source-tag {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600; margin: 2px; color: white;
    }
    .intent-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 500; background: #e8f4fd;
        color: #1976d2; margin-bottom: 0.5rem;
    }
    .stat-card {
        background: #f8f9fa; border-radius: 8px; padding: 1rem;
        margin: 0.5rem 0; border-left: 4px solid #667eea;
    }
    .stat-card .stat-number { font-size: 1.5rem; font-weight: 700; color: #333; }
    .stat-card .stat-label { font-size: 0.85rem; color: #666; }
    .search-result-card {
        background: white; border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 1rem; margin: 0.5rem 0; transition: box-shadow 0.2s;
    }
    .search-result-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .quick-btn { margin: 4px !important; }
    .chat-message-user {
        background: #e3f2fd; border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
    }
    .chat-message-assistant {
        background: #f5f5f5; border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────
# 初始化 Session State
# ──────────────────────────────────────────

def load_secrets():
    """从 Streamlit Secrets 加载 API 配置"""
    try:
        if hasattr(st, "secrets") and st.secrets:
            for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
                        "EMBEDDING_API_KEY", "EMBEDDING_API_BASE", "EMBEDDING_MODEL"):
                if key in st.secrets:
                    config_key = key
                    if key == "LLM_BASE_URL":
                        config_key = "LLM_API_BASE"
                    elif key == "EMBEDDING_MODEL":
                        config_key = "EMBEDDING_API_MODEL"
                    setattr(config, config_key, st.secrets[key])
                    if key == "LLM_API_KEY":
                        os.environ["LLM_API_KEY"] = st.secrets[key]
    except Exception:
        pass


load_secrets()


def init_session_state():
    """初始化会话状态"""
    defaults = {
        "engine": None,
        "chat_history": [],
        "initialized": False,
        "api_configured": bool(config.LLM_API_KEY),
        "init_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ──────────────────────────────────────────
# API 配置面板
# ──────────────────────────────────────────

def render_api_config():
    """渲染 API 配置界面"""
    st.markdown("### 🔑 API 配置")

    with st.form("api_config_form"):
        api_key = st.text_input(
            "OpenAI API Key",
            value=config.LLM_API_KEY,
            type="password",
            help="支持 OpenAI 兼容接口（通义千问、DeepSeek 等）"
        )
        api_base = st.text_input(
            "API Base URL",
            value=config.LLM_API_BASE,
            help="默认 OpenAI，国内可替换为兼容接口地址"
        )
        llm_model = st.text_input(
            "LLM 模型",
            value=config.LLM_MODEL,
            help="如 gpt-4o-mini、qwen-plus、deepseek-chat 等"
        )
        embedding_model = st.text_input(
            "Embedding 模型",
            value=config.EMBEDDING_API_MODEL,
            help="如 text-embedding-3-small"
        )

        submitted = st.form_submit_button("💾 保存配置", use_container_width=True)

        if submitted and api_key:
            config.LLM_API_KEY = api_key
            config.LLM_API_BASE = api_base
            config.LLM_MODEL = llm_model
            config.EMBEDDING_API_KEY = api_key
            config.EMBEDDING_API_BASE = api_base
            config.EMBEDDING_API_MODEL = embedding_model
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = api_base
            st.session_state.api_configured = True
            st.success("✅ 配置已保存！")
            st.rerun()


# ──────────────────────────────────────────
# 引擎初始化（懒加载 + 异常保护）
# ──────────────────────────────────────────

def do_init_engine():
    """加载已有索引，不执行重建操作"""
    placeholder = st.empty()
    placeholder.info("⏳ 正在加载已有索引...")

    try:
        from rag_engine import load_engine

        engine = load_engine()
        if engine is None:
            placeholder.empty()
            st.warning("⚠️ 未找到已有索引，请在命令行运行以下命令构建索引：")
            st.code("python build_index.py", language="bash")
            st.caption(f"知识库路径：{config.KB_ROOT}")
            with st.expander("📋 说明"):
                st.markdown("""
                索引构建会在后台执行以下操作：
                1. 加载本地 Embedding 模型（首次需下载）
                2. 扫描知识库目录中的所有 Markdown 文件
                3. 解析 YAML frontmatter、Obsidian 链接
                4. 分块并生成向量索引
                5. 保存至 `chroma_db/` 目录
                
                构建过程约需 1-2 分钟，请在命令行运行：
                ```
                cd research-rag-agent
                python build_index.py
                ```
                构建完成后刷新本页面即可。
                """)
            return False

        st.session_state.engine = engine
        st.session_state.api_configured = True
        st.session_state.init_error = None
        placeholder.empty()

        stats = engine.get_kb_stats()
        if stats.get("total_chunks", 0) == 0:
            st.warning("⚠️ 索引为空，需要重建索引：`python build_index.py --force`")
        else:
            st.success(f"✅ 索引加载成功！{stats['total_chunks']} 个向量块")
        return True

    except Exception as e:
        st.session_state.init_error = str(e)
        placeholder.empty()
        st.error(f"索引加载失败：{e}")
        with st.expander("🔍 详细错误信息"):
            st.code(traceback.format_exc())
        return False


def do_rebuild_index():
    """重建索引 - 提示用户在命令行运行"""
    st.warning("⚠️ 索引重建需要 1-2 分钟，请在命令行运行：")
    st.code("python build_index.py --force", language="bash")
    st.caption("重建完成后刷新本页面即可加载新索引。")


# ──────────────────────────────────────────
# 侧边栏
# ──────────────────────────────────────────

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 🔬 研知通")
        st.caption("Research KB RAG Agent")

        # 模式切换
        mode = st.radio(
            "交互模式",
            ["💬 对话问答", "🔍 语义搜索", " 知识库浏览"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # 知识库状态
        if st.session_state.engine:
            try:
                stats = st.session_state.engine.get_kb_stats()
                st.markdown("### 📊 知识库状态")
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['total_chunks']}</div>
                    <div class="stat-label">索引块总数</div>
                </div>
                """, unsafe_allow_html=True)

                for note_type, count in stats.get("type_counts", {}).items():
                    label = config.NOTE_TYPE_LABELS.get(note_type, note_type)
                    color = config.NOTE_TYPE_COLORS.get(note_type, "#999")
                    st.markdown(
                        f'<span class="source-tag" style="background:{color}">{label}：{count}</span>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.caption("知识库状态获取失败")

            st.divider()

        # 显示初始化错误
        if st.session_state.init_error:
            st.error(f"上次初始化失败：{st.session_state.init_error[:200]}")
            st.divider()

        # API 配置
        with st.expander("⚙️ API 配置", expanded=not st.session_state.api_configured):
            render_api_config()

        # 操作按钮
        st.divider()

        if not st.session_state.engine:
            if st.button("🚀 初始化引擎", use_container_width=True, type="primary"):
                do_init_engine()
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重建索引", use_container_width=True):
                if st.session_state.engine or config.LLM_API_KEY:
                    do_rebuild_index()
                    st.rerun()
                else:
                    st.warning("请先配置 API Key 或点击「初始化引擎」")
        with col2:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.engine:
                    st.session_state.engine.clear_memory()
                st.rerun()

        # 快捷问题
        st.divider()
        st.markdown("### ⚡ 快捷提问")
        quick_questions = [
            "知识库有哪些学习路线？",
            "Transformer 怎么学？",
            "VLA 评测怎么做？",
            "有什么 AI 产品 idea？",
            "今天该做什么实验？",
            "LoRA 微调怎么跑？",
            "为什么要入库 vla-eval？",
            "怎么做 GitHub 项目分析？",
        ]
        for q in quick_questions:
            if st.button(q, key=f"quick_{q}", use_container_width=True):
                st.session_state.quick_question = q

        # 底部信息
        st.divider()
        st.caption(f"知识库：{config.KB_ROOT}")
        st.caption(f"模型：{config.LLM_MODEL}")

    return mode


# ──────────────────────────────────────────
# 对话模式
# ──────────────────────────────────────────

INTENT_LABELS = {
    "learn": "📚 学习",
    "experiment": "🧪 实验",
    "decision": "🤔 决策",
    "workflow": "⚙️ 流程",
    "explore": "💡 探索",
    "general": "🔎 通用",
}

NOTE_TYPE_COLORS = config.NOTE_TYPE_COLORS


def render_chat_mode():
    """渲染对话问答界面"""
    st.markdown('<div class="app-title">🔬 研知通</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">你的 Obsidian 科研知识库智能助手 — 搜索、检索、对话学习</div>', unsafe_allow_html=True)

    # 显示历史对话
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if "sources" in msg and msg["sources"]:
                    with st.expander(f"📎 引用来源（{len(msg['sources'])} 条）"):
                        for src in msg["sources"]:
                            type_label = src.get("type_label", "")
                            color = NOTE_TYPE_COLORS.get(src.get("type", ""), "#999")
                            st.markdown(
                                f'<span class="source-tag" style="background:{color}">{type_label}</span>'
                                f' **{src.get("file", "")}**',
                                unsafe_allow_html=True,
                            )
                            st.caption(src.get("content_preview", ""))
                if "intent" in msg:
                    intent_label = INTENT_LABELS.get(msg["intent"], msg["intent"])
                    st.caption(f"识别意图：{intent_label}")

    # 输入框
    prompt = st.chat_input("向研知通提问...")

    # 处理快捷问题
    if "quick_question" in st.session_state:
        prompt = st.session_state.quick_question
        del st.session_state.quick_question

    if prompt:
        if not st.session_state.engine:
            st.error("️ 请先在侧边栏点击「初始化引擎」")
            return

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                answer_placeholder = st.empty()
                sources = []
                intent = "general"
                answer_parts = []

                for chunk in st.session_state.engine.query_stream(prompt):
                    if isinstance(chunk, dict) and "__meta__" in chunk:
                        meta = chunk["__meta__"]
                        sources = meta["sources"]
                        intent = meta["intent"]
                    else:
                        answer_parts.append(chunk)
                        answer_placeholder.markdown("".join(answer_parts))

                answer = "".join(answer_parts)

                if sources:
                    with st.expander(f"📎 引用来源（{len(sources)} 条）"):
                        for src in sources:
                            type_label = src.get("type_label", "")
                            color = NOTE_TYPE_COLORS.get(src.get("type", ""), "#999")
                            st.markdown(
                                f'<span class="source-tag" style="background:{color}">{type_label}</span>'
                                f' **{src.get("file", "")}**',
                                unsafe_allow_html=True,
                            )
                            st.caption(src.get("content_preview", ""))

                intent_label = INTENT_LABELS.get(intent, intent)
                st.caption(f"识别意图：{intent_label}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "intent": intent,
                })

            except Exception as e:
                error_msg = f"❌ 生成失败：{e}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                })


# ──────────────────────────────────────────
# 搜索模式
# ──────────────────────────────────────────

def render_search_mode():
    """渲染语义搜索界面"""
    st.markdown("### 🔍 语义搜索")
    st.caption("直接检索知识库内容，不调用 LLM 生成")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("搜索关键词", placeholder="输入搜索内容...")
    with col2:
        note_type = st.selectbox(
            "类型过滤",
            ["全部", "知识卡", "实验", "日报", "工作流", "原始资料", "地图"],
            index=0,
        )

    type_map = {
        "全部": None, "知识卡": "knowledge", "实验": "experiment",
        "日报": "daily", "工作流": "skill", "原始资料": "source", "地图": "map",
    }

    if query and st.session_state.engine:
        with st.spinner("搜索中..."):
            try:
                results = st.session_state.engine.search_only(
                    query, top_k=15, note_type=type_map.get(note_type),
                )
            except Exception as e:
                st.error(f"搜索失败：{e}")
                return

        if results:
            st.success(f"找到 {len(results)} 条相关结果")
            for i, res in enumerate(results):
                type_label = res.get("type_label", "")
                color = NOTE_TYPE_COLORS.get(res.get("type", ""), "#999")
                tags = res.get("tags", [])
                tag_str = " ".join([f"`{t}`" for t in (tags or [])[:5]])

                with st.expander(
                    f"**{i+1}. {res.get('file', '未知')}** — {type_label} {tag_str}",
                    expanded=(i < 3),
                ):
                    st.markdown(f"""
                    <span class="source-tag" style="background:{color}">{type_label}</span>
                    <span style="color:#666;font-size:0.85rem;">{res.get('file', '')}</span>
                    """, unsafe_allow_html=True)
                    st.markdown(res.get("content", ""))
        else:
            st.info("未找到相关结果")


# ──────────────────────────────────────────
# 浏览模式
# ─────────────────────────────────────────

def render_browse_mode():
    """渲染知识库浏览界面"""
    st.markdown("### 📊 知识库浏览")

    if not st.session_state.engine:
        st.warning("请先初始化引擎")
        return

    try:
        stats = st.session_state.engine.get_kb_stats()
    except Exception:
        st.error("获取知识库统计失败")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("索引块总数", stats["total_chunks"])
    with col2:
        st.metric("笔记类型", len(stats.get("type_counts", {})))
    with col3:
        kb_path = Path(config.KB_ROOT)
        md_count = len(list(kb_path.rglob("*.md"))) if kb_path.exists() else 0
        st.metric("MD 文件数", md_count)

    st.divider()

    type_counts = stats.get("type_counts", {})
    for note_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        label = config.NOTE_TYPE_LABELS.get(note_type, note_type)
        color = NOTE_TYPE_COLORS.get(note_type, "#999")

        with st.expander(f"{label}（{count} 块）"):
            try:
                results = st.session_state.engine.search_only(
                    "", top_k=5, note_type=note_type
                )
                for res in results:
                    st.markdown(f"- **{res.get('file', '')}**")
                    if res.get("tags"):
                        st.caption("标签：" + " ".join([f"`{t}`" for t in res["tags"][:5]]))
            except Exception:
                st.info("暂无数据")

    st.divider()
    st.markdown("#### 📅 最近日报")
    try:
        daily_results = st.session_state.engine.search_only(
            "今日完成 明日行动", top_k=5, note_type="daily"
        )
        for res in daily_results:
            st.markdown(f"**{res.get('file', '')}**")
            preview = res.get("content", "")[:200]
            st.caption(preview + "...")
    except Exception:
        pass


# ──────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────

mode = render_sidebar()

if "💬" in mode:
    render_chat_mode()
elif "🔍" in mode:
    render_search_mode()
elif "📊" in mode:
    render_browse_mode()
