# app.py
# ------
# This is the main entry point for our RAG application.
# It creates the Streamlit web interface that users interact with.
#
# Streamlit is a Python library that turns a Python script into a web app.
# It handles all the HTML, CSS, and JavaScript for you — you just write Python.
#
# Why is the UI in a separate file?
# Keeping the UI separate from the logic makes the code easier to maintain.
# If we want to change how the app looks, we only edit app.py.
# If we want to improve the search algorithm, we edit the logic files.
# This is called "separation of concerns" — each file has one job.

import streamlit as st
from datetime import datetime
from rag_pipeline import initialize_vector_store, run_rag, get_feature_status
from conversation import ConversationHistory

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Learning App",
    page_icon="🔍",
    layout="wide",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    h1 { font-size: 1.8rem !important; font-weight: 700 !important; color: #e6edf3 !important; }
    h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }
    p, li, label { color: #8b949e !important; }

    [data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #388bfd !important;
    }

    .stButton > button {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
    }

    [data-testid="stExpander"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    hr { border-color: #30363d !important; }

    .stat-badge {
        display: inline-block;
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        color: #8b949e;
        margin: 2px 0;
    }

    .rewritten-query {
        font-size: 0.78rem;
        color: #388bfd;
        margin-top: 4px;
        font-style: italic;
    }

    .msg-time {
        font-size: 0.72rem;
        color: #484f58;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
# Streamlit re-runs the entire script on every user interaction (like a button click).
# "Session state" lets us persist data between re-runs, like a conversation history.
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ConversationHistory()

if "store_initialized" not in st.session_state:
    st.session_state.store_initialized = False
    st.session_state.doc_count = 0

if "chat_messages" not in st.session_state:
    # This list stores messages for display purposes in the chat UI
    st.session_state.chat_messages = []

# --- Initialize Vector Store (once per session) ---
# We only load and embed documents on the very first run.
# After that, the vector store stays in memory for the whole session.
if not st.session_state.store_initialized:
    with st.spinner("Loading knowledge base... (this may take a moment the first time)"):
        doc_count = initialize_vector_store()
        st.session_state.store_initialized = True
        st.session_state.doc_count = doc_count

# --- Sidebar ---
with st.sidebar:
    st.markdown("### RAG Learning App")
    st.divider()

    st.markdown("**System**")
    st.markdown(f'<div class="stat-badge">📚 {st.session_state.doc_count} documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-badge">🧠 text-embedding-3-small</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-badge">⚡ gpt-4o-mini</div>', unsafe_allow_html=True)

    st.divider()

    msg_count = len([m for m in st.session_state.chat_messages if m["role"] == "user"])
    st.markdown(f'<div class="stat-badge">💬 {msg_count} message{"s" if msg_count != 1 else ""} this session</div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.conversation_history.clear()
        st.session_state.chat_messages = []
        st.rerun()

    st.divider()

    st.markdown("**How It Works**")
    st.markdown("""
    1. Query is rewritten for better search
    2. Embedded into a vector
    3. ChromaDB finds closest documents
    4. GPT-4o-mini answers from context
    5. Second model checks for hallucinations
    """)

    st.divider()

    st.markdown("**Progress**")
    status = get_feature_status()
    for feature, done in status.items():
        if done is True:
            st.success(f"✅ {feature}", icon=None)
        elif done is False:
            st.warning(f"⬜ {feature}", icon=None)
        else:
            st.info(f"🔲 {feature}", icon=None)

# --- Main Content ---
st.markdown("## RAG Learning App")
st.markdown('<p style="color:#8b949e;margin-top:-8px;">Ask questions about Python, machine learning, databases, and AI</p>', unsafe_allow_html=True)
st.divider()

def render_assistant_meta(message):
    """Render sources, confidence, grounding for an assistant message."""
    if message.get("rewritten_query") and message["rewritten_query"] != message.get("original_query"):
        st.markdown(f'<div class="rewritten-query">🔄 Searched as: "{message["rewritten_query"]}"</div>', unsafe_allow_html=True)

    if message.get("sources"):
        with st.expander(f"Sources — {len(message['sources'])} documents retrieved"):
            for i, (source, distance) in enumerate(zip(message["sources"], message["distances"])):
                similarity = max(0, 1 - distance / 2)
                st.markdown(f"**Source {i+1}** · similarity `{similarity:.2f}`")
                st.markdown(f"> {source}")
                if i < len(message["sources"]) - 1:
                    st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            confidence = message.get("confidence", 0)
            color = "green" if confidence > 0.6 else "orange" if confidence > 0.3 else "red"
            st.markdown(f":{color}[**{confidence:.0%}** confidence]")
        with col2:
            grounding = message.get("grounding", {})
            verdict = grounding.get("verdict", "")
            if verdict == "GROUNDED":
                st.markdown(":green[✓ Grounded]")
            elif verdict == "PARTIAL":
                st.markdown(":orange[⚠ Partial]")
            elif verdict == "HALLUCINATED":
                st.markdown(":red[✗ Hallucinated]")
        with col3:
            if message.get("timestamp"):
                st.markdown(f'<span style="color:#484f58;font-size:0.75rem;">{message["timestamp"]}</span>', unsafe_allow_html=True)

        warning = grounding.get("warning", "")
        if warning:
            st.warning(warning)


# --- Display Chat History ---
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        if message.get("timestamp") and message["role"] == "user":
            st.markdown(f'<div class="msg-time">{message["timestamp"]}</div>', unsafe_allow_html=True)
        st.write(message["content"])
        if message["role"] == "assistant":
            render_assistant_meta(message)

# --- Query Input ---
query = st.chat_input("Ask a question...")

if query:
    timestamp = datetime.now().strftime("%I:%M %p")

    with st.chat_message("user"):
        st.markdown(f'<div class="msg-time">{timestamp}</div>', unsafe_allow_html=True)
        st.write(query)

    st.session_state.chat_messages.append({
        "role": "user",
        "content": query,
        "timestamp": timestamp,
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_rag(query, st.session_state.conversation_history)

        if result["error"]:
            st.error(result["answer"])
        else:
            st.write(result["answer"])

        msg = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "distances": result["distances"],
            "confidence": result["confidence"],
            "grounding": result["grounding"],
            "rewritten_query": result.get("rewritten_query", ""),
            "original_query": query,
            "timestamp": timestamp,
        }
        render_assistant_meta(msg)

    st.session_state.chat_messages.append(msg)
