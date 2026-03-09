import sys, logging
from pathlib import Path
import streamlit as st

# MUST be the very first Streamlit call
st.set_page_config(
    page_title="NASABot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

# ── Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; }
.user-bubble {
    background: linear-gradient(135deg, #1a3a6b, #0d2147);
    border-left: 3px solid #4a90e2;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px; margin: 8px 0; color: #e8f0fe; font-size: 15px;
}
.bot-bubble {
    background: linear-gradient(135deg, #0d1f0d, #122b12);
    border-left: 3px solid #4caf50;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px; margin: 8px 0; color: #e8f5e9; font-size: 15px;
}
.source-card {
    background: #1a1f2e; border: 1px solid #2d3654;
    border-radius: 8px; padding: 10px 14px; margin: 4px 0;
    font-size: 13px; color: #90a4c8;
}
.source-card strong { color: #7eb8f7; }
</style>
""", unsafe_allow_html=True)


# ── Load RAG (cached so it only loads once) ───────────────────────
@st.cache_resource(show_spinner="Loading NASABot...")
def load_rag():
    try:
        from src.rag_chain import RAGChain
        chain = RAGChain()
        stats = chain.retriever.get_stats()
        return chain, stats, None
    except Exception as e:
        return None, {}, str(e)


# ── Session State ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    from src.memory import ConversationMemory
    st.session_state.memory = ConversationMemory()
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 NASABot")
    st.markdown("*NASA Space Research Expert*")
    st.divider()

    chain, stats, err = load_rag()

    if err:
        st.error(f"Error: {err}")
        st.info("Run: python -m src.ingest")
        st.stop()

    st.markdown("### 📊 Knowledge Base")
    st.metric("Chunks indexed", f"{stats.get('total_chunks', 0):,}")
    st.metric("Questions asked", st.session_state.query_count)
    st.markdown("⚡ **Hybrid Search** (BM25 + Semantic + RRF)")
    st.divider()

    st.markdown("### 💡 Try asking:")
    examples = [
        "What is the Artemis program?",
        "How does the James Webb Telescope work?",
        "Tell me about Mars exploration",
        "What propulsion is used in deep space?",
        "What is the ISS?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex[:20]):
            st.session_state._pending = ex
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.query_count = 0
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────
st.markdown("# 🚀 NASABot — Space Research Expert")
st.markdown("Answers powered by **60 NASA documents** with source citations.")
st.divider()

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;color:#4b5563;padding:60px 0">
        <div style="font-size:56px">🛸</div>
        <div style="font-size:20px;margin-top:16px">
            Ask me anything about NASA!</div>
    </div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">👤 <b>You:</b> {msg["content"]}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="bot-bubble">🤖 <b>NASABot:</b><br><br>{msg["content"]}</div>',
            unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} sources cited"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card"><strong>[Source {s["num"]}]</strong> '
                        f'{s["title"]}<br>📄 {s["source"]} · Page {s["page"]}</div>',
                        unsafe_allow_html=True)


# ── Handle Input ──────────────────────────────────────────────────
pending = st.session_state.pop("_pending", None)
user_input = st.chat_input("Ask about NASA missions, research, technology...")
query = pending or user_input

if query and chain:
    st.session_state.messages.append(
        {"role": "user", "content": query, "sources": []}
    )
    with st.spinner("🔍 Searching NASA documents..."):
        result = chain.answer(query, session_memory=st.session_state.memory)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.session_state.query_count += 1
    st.rerun()
