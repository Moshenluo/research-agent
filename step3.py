import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import streamlit as st
import config

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.PAGE_ICON, layout="wide")
st.title("Step 3: Test rag_engine import")

if "engine" not in st.session_state:
    st.session_state.engine = None

if st.button("Import rag_engine"):
    st.write("Importing...")
    from rag_engine import load_engine, init_engine
    st.write("Import OK!")

if st.button("Load engine"):
    st.write("Loading...")
    from rag_engine import load_engine, init_engine
    engine = load_engine()
    if engine:
        st.write(f"Loaded: {engine.get_kb_stats()['total_chunks']} chunks")
        st.session_state.engine = engine
    else:
        st.write("No existing index, trying to build...")
        engine = init_engine(force_rebuild=False)
        st.write(f"Built: {engine.get_kb_stats()['total_chunks']} chunks")
        st.session_state.engine = engine
