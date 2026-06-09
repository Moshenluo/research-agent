import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import streamlit as st
import config

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.PAGE_ICON, layout="wide")

st.markdown("""
<style>
.source-tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin: 2px; color: white; }
.intent-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 500; background: #e8f4fd; color: #1976d2; }
.stat-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #667eea; }
.stat-card .stat-number { font-size: 1.5rem; font-weight: 700; color: #333; }
.stat-card .stat-label { font-size: 0.85rem; color: #666; }
</style>
""", unsafe_allow_html=True)

try:
    if hasattr(st, "secrets") and st.secrets:
        if "LLM_API_KEY" in st.secrets:
            config.LLM_API_KEY = st.secrets["LLM_API_KEY"]
except Exception:
    pass

if "engine" not in st.session_state:
    st.session_state.engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Step 2: config + CSS + session_state")
st.write("Process should still be alive")

with st.sidebar:
    mode = st.radio("Mode", ["A", "B", "C"])
    if st.session_state.engine:
        st.write("Engine loaded")
    else:
        st.write("Engine not loaded")
    if st.button("Init"):
        st.write("Clicked")
