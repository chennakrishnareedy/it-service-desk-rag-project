"""
Streamlit UI for the IT Service Desk RAG Assistant.

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_pipeline import RagAssistant  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@st.cache_resource(show_spinner="Loading models and building the index...")
def get_assistant() -> RagAssistant:
    return RagAssistant(DOCS_DIR, load_llm=True)


st.set_page_config(page_title="IT Service Desk Assistant", page_icon="🛠️")
st.title("🛠️ IT Service Desk RAG Assistant")
st.caption("Answers come only from approved IT documentation, with source citations.")

assistant = get_assistant()

with st.sidebar:
    st.header("Sample questions")
    samples = [
        "How can I reset my corporate password?",
        "What should I do if the VPN is not connecting?",
        "How do I install the approved Python version?",
        "What does error code VPN-403 mean?",
        "When should this issue be escalated to Level 2 support?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True):
            st.session_state["question"] = s
    st.divider()
    st.caption(f"Knowledge base: {len(assistant.index.chunks)} chunks")

question = st.text_input(
    "Ask an IT question:", value=st.session_state.get("question", "")
)

if st.button("Ask", type="primary") or question:
    if question.strip():
        with st.spinner("Searching approved docs and generating an answer..."):
            ans = assistant.ask(question)

        if ans.blocked:
            st.error(ans.text)
        else:
            st.markdown("### Answer")
            st.write(ans.text)

            if ans.escalated:
                st.warning("This issue is flagged for escalation to the support team.")

            if ans.citations:
                st.markdown("### 📑 Sources")
                for c in ans.citations:
                    st.markdown(f"- {c}")

            with st.expander("Retrieved evidence (debug)"):
                for r in ans.retrieved:
                    st.markdown(
                        f"**{r.chunk.source} p.{r.chunk.page}** "
                        f"(similarity {r.score:.2f})"
                    )
                    st.caption(r.chunk.text)
