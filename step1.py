import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import streamlit as st
import config

st.set_page_config(page_title=config.APP_TITLE, page_icon=config.PAGE_ICON, layout="wide")
st.title("Step 1: config imported")
st.write("Process should still be alive")

st.sidebar.button("Test")
