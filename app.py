"""
研知通 · Research KB RAG Agent
基于 Streamlit 的 RAG 对话引擎 — 自动连接知识库，感知更新
"""

import os
import time
import traceback

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
# CSS 样式
# ──────────────────────────────────────────

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .stApp { background: #f8fafc; }

    /* Header */
    .header { text-align: center; padding: 2rem 0 0.5rem 0; }
    .header-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .header-sub { color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem; }

    /* Status bar */
    .status-bar {
        display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap;
        margin: 1rem 0;
    }
    .status-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.35rem 0.85rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 500;
    }
    .status-badge.online { background: #dcfce7; color: #16a34a; }
    .status-badge.offline { background: #fef3c7; color: #d97706; }
    .status-badge.building { background: #dbeafe; color: #2563eb; }
    .status-badge.info { background: #f1f5f9; color: #64748b; }

    .status-dot {
        width: 7px; height: 7px; border-radius: 50%; display: inline-block;
    }
    .status-dot.online { background: #16a34a; animation: pulse 2s infinite; }
    .status-dot.offline { background: #d97706; }
    .status-dot.building { background: #2563eb; animation: pulse 1s infinite; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Chat bubbles */
    .stChatMessage { border-radius: 16px !important; }
    .chat-bubble-user {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white; border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1.1rem; display: inline-block;
    }
    .chat-bubble-assistant {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1.1rem;
    }

    /* Source cards */
    .source-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 0.6rem 1rem; margin: 0.35rem 0;
        transition: box-shadow 0.15s, border-color 0.15s;
    }
    .source-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-color: #cbd5e1; }
    .source-type-tag {
        display: inline-block; padding: 2px 8px; border-radius: 6px;
        font-size: 0.7rem; font-weight: 600; color: white; margin-right: 0.5rem;
    }
    .source-file { font-weight: 600; font-size: 0.85rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] button {
        background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important; transition: all 0.15s;
    }
    section[data-testid="stSidebar"] button:hover {
        background: rgba(255,255,255,0.15) !important; border-color: rgba(255,255,255,0.25) !important;
    }
    section[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
    .sidebar-logo { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.15rem; }
    .sidebar-sub { font-size: 0.75rem; color: #94a3b8 !important; }

    /* Stats */
    .kb-stat { text-align: center; padding: 0.5rem; }
    .kb-stat-number { font-size: 1.8rem; font-weight: 700; color: #fff !important; }
    .kb-stat-label { font-size: 0.7rem; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Expander */
    .streamlit-expanderHeader { font-size: 0.85rem !important; }

    /* Input */
    .stChatInput textarea { border-radius: 14px !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────
# Session State
# ──────────────────────────────────────────

def init_session():
    defaults = {
        "chat_history": [],
        "initialized": False,
        "init_error": None,
        "build_info": None,
        "last_check_time": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ──────────────────────────────────────────
# 引擎缓存与自动初始化
# ──────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_cached_engine():
    """自动加载引擎，检测知识库变更并增量更新"""
    from rag_engine import load_engine

    engine = load_engine()
    if engine is None:
        return None
    return engine


def check_and_update_engine():
    """检查知识库是否变更，如有变更则增量更新"""
    from ingest import load_manifest, save_manifest, compute_file_hash
    from pathlib import Path
    import config as cfg

    persist_dir = config.CHROMA_PERSIST_DIR
    manifest_path = os.path.join(persist_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        return None

    old_manifest = load_manifest(persist_dir)

    # 扫描当前文件
    kb = Path(cfg.KB_ROOT)
    if not kb.exists():
        return None

    current_manifest = {}
    for md_file in kb.rglob("*.md"):
        if any(p.startswith(".") for p in md_file.relative_to(kb).parts):
            continue
        try:
            rel = str(md_file.relative_to(kb))
            current_manifest[rel] = compute_file_hash(str(md_file))
        except Exception:
            continue

    # 也扫描顶层文件
    for filename in cfg.KB_TOP_FILES:
        fp = kb / filename
        if fp.exists():
            current_manifest[filename] = compute_file_hash(str(fp))

    added = set(current_manifest.keys()) - set(old_manifest.keys())
    removed = set(old_manifest.keys()) - set(current_manifest.keys())
    changed = {
        k for k in set(current_manifest.keys()) & set(old_manifest.keys())
        if current_manifest[k] != old_manifest[k]
    }

    if added or removed or changed:
        return {
            "added": list(added), "removed": list(removed), "changed": list(changed),
            "total_changes": len(added) + len(removed) + len(changed),
            "manifest": current_manifest,
        }
    return None


# ──────────────────────────────────────────
# 初始化逻辑
# ──────────────────────────────────────────

def initialize_app():
    """自动初始化：加载引擎、检测更新"""
    status = {"state": "loading", "message": "正在连接知识库..."}

    # Phase 1: 加载引擎缓存
    engine = get_cached_engine()
    if engine is None:
        status = {"state": "no_index", "message": "未找到索引"}
        return status

    status["chunks"] = engine.get_kb_stats().get("total_chunks", 0)

    # Phase 2: 检查知识库变更
    changes = check_and_update_engine()
    if changes and changes["total_changes"] > 0:
        status["state"] = "outdated"
        status["changes"] = changes
        return status

    status["state"] = "ready"
    return status


# ──────────────────────────────────────────
# 侧边栏
# ──────────────────────────────────────────

NOTE_TYPE_COLORS = config.NOTE_TYPE_COLORS
NOTE_TYPE_MAP = config.NOTE_TYPE_LABELS
INTENT_LABELS = {
    "learn": "📚 学习", "experiment": "🧪 实验", "decision": "🤔 决策",
    "workflow": "⚙️ 流程", "explore": "💡 探索", "general": "🔎 通用",
}


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🔬 研知通</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Research KB RAG Agent</div>', unsafe_allow_html=True)
        st.divider()

        # 模式
        mode = st.radio("", ["💬 对话", "🔍 搜索", "📊 浏览"], label_visibility="collapsed")
        st.divider()

        # 引擎状态
        status = initialize_app()
        st.markdown("### 📡 引擎状态")

        if status["state"] == "ready":
            chunks = status.get("chunks", 0)
            st.markdown(f"""
            <div class="kb-stat">
                <div class="kb-stat-number">{chunks}</div>
                <div class="kb-stat-label">向量块已就绪</div>
            </div>
            <div style="text-align:center;margin-top:0.25rem;">
                <span class="status-dot online"></span>
                <span style="font-size:0.75rem;color:#86efac;">已连接</span>
            </div>
            """, unsafe_allow_html=True)

            # KB 统计
            with st.expander("📊 类型分布"):
                engine = get_cached_engine()
                if engine:
                    stats = engine.get_kb_stats()
                    for nt, cnt in sorted(stats.get("type_counts", {}).items(), key=lambda x: -x[1]):
                        label = NOTE_TYPE_MAP.get(nt, nt)
                        color = NOTE_TYPE_COLORS.get(nt, "#999")
                        st.markdown(
                            f'<span class="source-type-tag" style="background:{color}">'
                            f'{label}</span> {cnt} 块',
                            unsafe_allow_html=True,
                        )

            # 检查更新
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 检查更新", use_container_width=True):
                    st.cache_resource.clear()
                    st.rerun()
            with col2:
                if st.button("🔨 完整重建", use_container_width=True):
                    with st.spinner("重建中..."):
                        from rag_engine import init_engine as rebuild
                        rebuild(force_rebuild=True)
                        st.cache_resource.clear()
                    st.rerun()

        elif status["state"] == "outdated":
            changes = status["changes"]
            st.warning(f"🔄 检测到 {changes['total_changes']} 个文件变更")
            st.markdown(f"""
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem;">
            新增 {len(changes['added'])} · 修改 {len(changes['changed'])} · 删除 {len(changes['removed'])}
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ 同步更新", use_container_width=True, type="primary"):
                with st.spinner("正在增量更新索引..."):
                    from rag_engine import init_engine as rebuild
                    rebuild(force_rebuild=True)
                    st.cache_resource.clear()
                st.rerun()

        elif status["state"] == "no_index":
            st.error("⚠️ 索引不存在")
            st.caption("在命令行运行以下命令构建索引：")
            st.code("python build_index.py", language="bash")

        elif status["state"] == "loading":
            st.info("⏳ 正在加载...")

        st.divider()

        # 清空对话
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # 快捷问题
        st.divider()
        st.markdown("### ⚡ 快捷提问")
        quick_questions = [
            "知识库有哪些学习路线？",
            "Transformer 怎么学？",
            "VLA 评测怎么做？",
            "有什么 AI 产品 idea？",
            "LoRA 微调怎么跑？",
        ]
        for q in quick_questions:
            if st.button(q, key=f"quick_{q}", use_container_width=True):
                st.session_state.quick_question = q

        st.divider()
        st.caption(f"📂 {config.KB_ROOT}")
        st.caption(f"🤖 {config.LLM_MODEL}")

    return mode


# ──────────────────────────────────────────
# 对话模式
# ──────────────────────────────────────────

def render_chat():
    st.markdown('<div class="header"><div class="header-title">研知通</div>'
                '<div class="header-sub">你的科研知识库智能助手</div></div>',
                unsafe_allow_html=True)

    # 状态栏
    engine = get_cached_engine()
    chunks = engine.get_kb_stats()["total_chunks"] if engine else 0
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-badge online">
            <span class="status-dot online"></span> 知识库已连接
        </div>
        <div class="status-badge info">
            {chunks} 个向量块
        </div>
        <div class="status-badge info">
            {config.LLM_MODEL}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 历史消息
    for msg in st.session_state.chat_history:
        role = msg["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>',
                            unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📎 来源 ({len(msg['sources'])} 条)"):
                        for src in msg["sources"]:
                            tl = src.get("type_label", "")
                            clr = NOTE_TYPE_COLORS.get(src.get("type", ""), "#999")
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="source-type-tag" style="background:{clr}">{tl}</span>
                                <span class="source-file">{src.get("file", "")}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption(src.get("content_preview", ""))
                if msg.get("intent"):
                    st.caption(INTENT_LABELS.get(msg["intent"], msg["intent"]))

    # 输入
    prompt = st.chat_input("输入你的问题...")
    if "quick_question" in st.session_state:
        prompt = st.session_state.quick_question
        del st.session_state.quick_question

    if prompt:
        if not engine:
            st.error("⚠️ 引擎未就绪，请先构建索引：`python build_index.py`")
            return

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            try:
                placeholder = st.empty()
                sources, intent = [], "general"
                parts = []

                for chunk in engine.query_stream(prompt):
                    if isinstance(chunk, dict) and "__meta__" in chunk:
                        sources = chunk["__meta__"]["sources"]
                        intent = chunk["__meta__"]["intent"]
                    else:
                        parts.append(chunk)
                        placeholder.markdown("".join(parts))

                answer = "".join(parts)

                if sources:
                    with st.expander(f"📎 来源 ({len(sources)} 条)"):
                        for src in sources:
                            clr = NOTE_TYPE_COLORS.get(src.get("type", ""), "#999")
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="source-type-tag" style="background:{clr}">{src.get('type_label','')}</span>
                                <span class="source-file">{src.get('file','')}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption(src.get("content_preview", ""))

                st.caption(INTENT_LABELS.get(intent, intent))

                st.session_state.chat_history.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "intent": intent,
                })
            except Exception as e:
                st.error(f"❌ {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"❌ {e}"})


# ──────────────────────────────────────────
# 搜索模式
# ──────────────────────────────────────────

def render_search():
    st.markdown("### 🔍 语义搜索")
    engine = get_cached_engine()

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("", placeholder="输入搜索关键词...", label_visibility="collapsed")
    with c2:
        nt = st.selectbox("", ["全部", "知识卡", "实验", "日报", "工作流", "原始资料"], label_visibility="collapsed")

    type_map = {"全部": None, "知识卡": "knowledge", "实验": "experiment",
                "日报": "daily", "工作流": "skill", "原始资料": "source"}

    if query and engine:
        with st.spinner("搜索中..."):
            results = engine.search_only(query, top_k=15, note_type=type_map.get(nt))

        if results:
            st.success(f"找到 {len(results)} 条")
            for i, r in enumerate(results):
                clr = NOTE_TYPE_COLORS.get(r.get("type", ""), "#999")
                with st.expander(f"**{i+1}. {r.get('file','')}**", expanded=i < 2):
                    st.markdown(f'<span class="source-type-tag" style="background:{clr}">{r.get("type_label","")}</span>',
                                unsafe_allow_html=True)
                    st.markdown(r.get("content", ""))
        else:
            st.info("无结果")


# ──────────────────────────────────────────
# 浏览模式
# ──────────────────────────────────────────

def render_browse():
    st.markdown("### 📊 知识库浏览")
    engine = get_cached_engine()
    if not engine:
        st.warning("引擎未就绪")
        return

    stats = engine.get_kb_stats()
    c1, c2 = st.columns(2)
    c1.metric("向量块", stats["total_chunks"])
    c2.metric("类型数", len(stats.get("type_counts", {})))

    st.divider()
    for nt, cnt in sorted(stats.get("type_counts", {}).items(), key=lambda x: -x[1]):
        label = NOTE_TYPE_MAP.get(nt, nt)
        clr = NOTE_TYPE_COLORS.get(nt, "#999")
        with st.expander(f"{label} ({cnt} 块)"):
            try:
                res = engine.search_only("", top_k=3, note_type=nt)
                for r in res:
                    st.markdown(f"- **{r.get('file','')}**")
            except Exception:
                st.caption("暂无预览")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

mode = render_sidebar()

if "对话" in mode:
    render_chat()
elif "搜索" in mode:
    render_search()
elif "浏览" in mode:
    render_browse()
