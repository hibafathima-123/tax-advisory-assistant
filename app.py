# ============================================================
# app.py — Streamlit Frontend for Tax Advisory Assistant
# ============================================================
# This file creates a clean web UI using Streamlit.
# It connects to rag_core.py to handle all AI/RAG logic.
#
# Run with:  streamlit run app.py
# ============================================================

import html
from pathlib import Path

import streamlit as st
from rag_core import initialize_rag, get_answer

# -------------------------------------------------------
# PAGE CONFIGURATION
# Must be the FIRST Streamlit command in the script
# -------------------------------------------------------
st.set_page_config(
    page_title="Tax Advisory Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS — Makes the UI look polished
# -------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
        color: #e8eaf0;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(90deg, #1e3a5f 0%, #0d2137 100%);
        border-left: 5px solid #f5a623;
        border-radius: 8px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(245,166,35,0.15);
    }
    .header-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: #a0b4c8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Answer card */
    .answer-card {
        background: #1a2744;
        border: 1px solid #2a3f6f;
        border-left: 4px solid #f5a623;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 16px;
        font-size: 1rem;
        line-height: 1.7;
        color: #dde4f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }

    /* Source card */
    .source-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #6b8aad;
        line-height: 1.5;
    }

    /* Sample question chips */
    .question-chip {
        display: inline-block;
        background: #1e2d45;
        color: #7eb3e8;
        border: 1px solid #2a3f6f;
        border-radius: 20px;
        padding: 5px 14px;
        margin: 4px;
        font-size: 0.82rem;
        cursor: pointer;
    }

    /* Info box */
    .info-box {
        background: #0d2137;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #7ea8c8;
        margin-bottom: 16px;
    }

    /* Streamlit input overrides */
    .stTextArea textarea {
        background: #1a2744 !important;
        color: #e8eaf0 !important;
        border: 1px solid #2a3f6f !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #f5a623, #e8920f) !important;
        color: #0d1117 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #1e2d45 !important;
    }

    /* Section label */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #f5a623;
        margin-bottom: 10px;
    }

    /* Status badge */
    .status-ready {
        display: inline-block;
        background: #0a2e1a;
        color: #4ade80;
        border: 1px solid #166534;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-loading {
        display: inline-block;
        background: #1c1a08;
        color: #fbbf24;
        border: 1px solid #854d0e;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Divider */
    hr { border-color: #1e2d45 !important; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# PATH TO DATA FILE
# -------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_FILE_CANDIDATES = ("tax_data.txt", "tax_data_.txt")


def resolve_data_file() -> Path:
    """Return the first available knowledge-base file in the app directory."""
    for candidate in DATA_FILE_CANDIDATES:
        candidate_path = APP_DIR / candidate
        if candidate_path.exists():
            return candidate_path

    checked_paths = ", ".join(str(APP_DIR / candidate) for candidate in DATA_FILE_CANDIDATES)
    raise FileNotFoundError(f"No knowledge base file found. Checked: {checked_paths}")

# -------------------------------------------------------
# CACHED INITIALIZATION
# @st.cache_resource ensures the RAG chain is built ONLY ONCE
# even if the user interacts with the UI multiple times.
# Without caching, it would reload models on every interaction.
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag_chain(data_file_path: str, data_file_mtime: float):
    """Load and cache the entire RAG pipeline."""
    _ = data_file_mtime
    return initialize_rag(data_file_path)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <h1>💼 Tax Advisory Assistant</h1>
    <p>AI-powered Q&A on Indian Income Tax — powered by RAG (Retrieval-Augmented Generation)</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR — About & Sample Questions
# -------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧾 About This App")
    st.markdown("""
    <div class="info-box">
    This assistant uses <strong>RAG</strong> to answer your Indian income tax questions.<br><br>
    It retrieves relevant information from a curated tax knowledge base and generates accurate, grounded answers.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Tech Stack")
    st.markdown("""
    <div class="info-box">
    🔹 <strong>LangChain</strong> — RAG pipeline<br>
    🔹 <strong>FAISS + Sentence-Transformers</strong> — Local semantic retrieval<br>
    🔹 <strong>Groq</strong> — Hosted LLM inference<br>
    🔹 <strong>Streamlit</strong> — Web interface<br>
    🔹 <strong>Python 3.10+</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")

    sample_questions = [
        "What are the income tax slabs for FY 2026-27?",
        "What is Section 80C deduction limit?",
        "What is the difference between old and new tax regime?",
        "How much can I save under Section 80D?",
        "What is the tax rebate under Section 87A?",
        "How is HRA exemption calculated?",
        "When is the last date to file ITR?",
        "What is TDS and how does it work?",
        "Are capital gains taxable?",
        "What is standard deduction for salaried employees?",
    ]

    # Clicking a sample question pre-fills the input box
    for q in sample_questions:
        if st.button(q, key=f"btn_{q[:20]}", use_container_width=True):
            st.session_state["user_query"] = q

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a6080; text-align:center;'>
    ⚠️ For educational purposes only.<br>Consult a CA for professional advice.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# MAIN AREA — Load RAG chain
# -------------------------------------------------------

# Initialize session state for query if not present
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

# Load the RAG chain (cached after first load)
st.markdown('<p class="section-label">System Status</p>', unsafe_allow_html=True)

with st.spinner("🔄 Initializing Groq connection and knowledge base..."):
    try:
        data_file = resolve_data_file()
        qa_chain = load_rag_chain(str(data_file), data_file.stat().st_mtime)
        st.markdown('<span class="status-ready">✅ Knowledge Base Ready</span>', unsafe_allow_html=True)
        st.caption(f"Using knowledge file: `{data_file.name}`")
    except FileNotFoundError:
        st.error(
            "❌ Could not find the tax knowledge base file.\n\n"
            f"Checked: `{', '.join(DATA_FILE_CANDIDATES)}` in `{APP_DIR}`."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG pipeline:\n\n`{str(e)}`")
        st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------
# QUERY INPUT BOX
# -------------------------------------------------------
st.markdown('<p class="section-label">Ask a Tax Question</p>', unsafe_allow_html=True)

user_query = st.text_area(
    label="Your Question",
    value=st.session_state.get("user_query", ""),
    placeholder="e.g. What is the income tax slab for someone earning Rs. 8 lakh per year?",
    height=100,
    label_visibility="collapsed",
    key="query_box"
)

# -------------------------------------------------------
# SUBMIT BUTTON
# -------------------------------------------------------
ask_button = st.button("🔍 Get Answer", use_container_width=True)

# -------------------------------------------------------
# PROCESS QUERY AND DISPLAY ANSWER
# -------------------------------------------------------
if ask_button:
    query = user_query.strip()

    if not query:
        st.warning("⚠️ Please type a question before clicking **Get Answer**.")
    else:
        with st.spinner("🤔 Searching knowledge base and generating answer..."):
            result = get_answer(qa_chain, query)

        answer = result["answer"]
        sources = result["sources"]
        formatted_answer = html.escape(answer).replace("\n", "<br>")

        # ---- Display the answer ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-card">{formatted_answer}</div>',
            unsafe_allow_html=True,
        )

        # ---- Optionally show retrieved source chunks ----
        if sources:
            with st.expander("📚 View Source Context (retrieved chunks)", expanded=False):
                st.markdown("""
                <div style='font-size:0.82rem; color:#6b8aad; margin-bottom:10px;'>
                These are the relevant sections retrieved from the knowledge base that were used to generate the answer above.
                </div>
                """, unsafe_allow_html=True)
                for i, src in enumerate(sources, 1):
                    st.markdown(f'<div class="source-card"><strong>Chunk {i}:</strong><br>{src}</div>', unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER NOTES
# -------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-size:0.8rem; color:#3a5070; padding-bottom:10px;'>
    Built with ❤️ using LangChain + FAISS + Groq + Streamlit &nbsp;|&nbsp;
    Data: Indian Income Tax Guidelines FY 2026-27<br>
    <em>This tool is for educational/demonstration purposes only. Not a substitute for professional tax advice.</em>
</div>
""", unsafe_allow_html=True)
