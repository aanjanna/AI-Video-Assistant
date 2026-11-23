"""
AI Video Assistant — Streamlit UI
Run with:  streamlit run app.py
Place this file in the project root (next to utils/ and core/).
"""

import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
    --paper:   #EEF1F7;
    --surface: #FFFFFF;
    --ink:     #1A2138;
    --muted:   #5E6784;
    --line:    #D9DEEB;
    --cobalt:  #3347E0;
    --cobalt-soft: #E4E8FD;
    --teal:    #14917F;
    --teal-soft: #DDF3EF;
    --amber:   #B8741A;
    --amber-soft: #FBEBD3;
    --rose:    #B8385A;
    --rose-soft: #F9E1E8;
}

html, body, [class*="css"], .stApp {
    font-family: 'Instrument Sans', sans-serif;
    color: var(--ink);
}
.stApp { background: var(--paper); }

h1, h2, h3, .display {
    font-family: 'Bricolage Grotesque', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--ink);
}

/* Hide default chrome */
#MainMenu, footer { visibility: hidden; }

/* Keep the header so the sidebar toggle still works, but make it blend in */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }

/* Make sure the open-sidebar button is always visible */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg { color: var(--ink); }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1080px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--ink);
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #E7EAF5; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] label p { color: #AEB5D1 !important; font-size: 0.85rem; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #262E4A !important;
    border: 1px solid #3A4466 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #262E4A;
    border: 1px dashed #4A5580;
    border-radius: 12px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: #3A4466; color: #fff; border: none;
}
.sb-brand { display:flex; align-items:center; gap:.6rem; margin-bottom:1.4rem; }
.sb-brand .mark {
    width:34px; height:34px; border-radius:10px; background:var(--cobalt);
    display:flex; align-items:center; justify-content:center; font-size:1.1rem;
}
.sb-brand .name { font-family:'Bricolage Grotesque',sans-serif; font-size:1.25rem; font-weight:700; color:#fff; }

/* Primary button */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid var(--line);
    font-weight: 600;
    padding: .6rem 1.1rem;
    transition: transform .08s ease, box-shadow .15s ease;
}
.stButton > button[kind="primary"] {
    background: var(--cobalt);
    border: none;
    color: #fff;
    box-shadow: 0 6px 16px rgba(51,71,224,.35);
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); background:#2A3CC8; }
.stButton > button:focus-visible { outline: 3px solid #9AA6F5; outline-offset: 2px; }

/* Hero */
.hero { padding: 2.4rem 0 1rem; }
.hero h1 { font-size: 3.1rem; line-height: 1.04; margin: 0 0 .9rem; font-weight: 700; max-width: 14ch; }
.hero p { font-size: 1.1rem; color: var(--muted); max-width: 52ch; line-height: 1.55; }

.steps { display:grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem; }
.step {
    background: var(--surface); border:1px solid var(--line); border-radius: 16px;
    padding: 1.1rem 1.2rem;
}
.step b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.05rem; display:block; margin-bottom:.25rem; }
.step span { color: var(--muted); font-size:.92rem; line-height:1.45; }

/* Result header */
.result-head { margin-bottom: 1.2rem; }
.result-head h1 { font-size: 2.2rem; margin: 0 0 .5rem; font-weight: 700; }
.chips { display:flex; gap:.5rem; flex-wrap:wrap; }
.chip {
    background: var(--surface); border:1px solid var(--line); border-radius: 999px;
    padding: .25rem .8rem; font-size:.82rem; color: var(--muted);
}

/* Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
}
.panel h3 { margin: 0 0 .7rem; font-size: 1.15rem; }
.panel.summary { border-left: 6px solid var(--cobalt); }
.panel.actions { border-left: 6px solid var(--teal); }
.panel.decisions { border-left: 6px solid var(--amber); }
.panel.questions { border-left: 6px solid var(--rose); }
.panel p, .panel li { line-height: 1.6; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    height: 2.8rem; padding: 0 1.1rem; font-weight: 600; color: var(--muted);
    border-radius: 10px 10px 0 0;
}
.stTabs [aria-selected="true"] { color: var(--cobalt) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--cobalt) !important; }

/* Transcript */
.transcript {
    background: var(--surface); border:1px solid var(--line); border-radius: 16px;
    padding: 1.3rem 1.5rem; line-height: 1.75; max-height: 520px; overflow-y: auto;
    white-space: pre-wrap; font-size: .97rem;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1.2rem;
}
[data-testid="stChatInput"] { border-radius: 14px; }

/* Empty-state / small text */
.hint { color: var(--muted); font-size: .9rem; }

@media (max-width: 760px) {
    .steps { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2.3rem; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "messages": [],
    "source_label": "",
    "language": "english",
    "processed_at": None,
}.items():
    st.session_state.setdefault(key, default)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    """Write an uploaded file to a temp path so process_input can read it."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def build_report(result: dict) -> str:
    """Markdown export of the full analysis."""
    return (
        f"# {result['title']}\n\n"
        f"## Summary\n{result['summary']}\n\n"
        f"## Action items\n{result['action_items']}\n\n"
        f"## Key decisions\n{result['key_decisions']}\n\n"
        f"## Open questions\n{result['open_questions']}\n\n"
        f"## Transcript\n{result['transcript']}\n"
    )


def run_pipeline_ui(source: str, language: str) -> dict:
    """Same steps as the CLI pipeline, with live progress in the UI."""
    with st.status("Working on your video…", expanded=True) as status:
        st.write("Fetching and splitting audio")
        chunks = process_input(source)

        st.write("Transcribing speech")
        transcript = transcribe_all(chunks, language)

        st.write("Writing the summary")
        summary = summarize(transcript)
        try:
            title = generate_title(transcript) or "AI Video Summary"
        except Exception:
            title = "AI Video Summary"

        st.write("Finding action items, decisions and open questions")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        st.write("Preparing the chat")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Analysis ready", state="complete", expanded=False)

    return {
        "title": str(title).strip().strip('"'),
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def panel(css_class: str, heading: str, body):
    st.markdown(f'<div class="panel {css_class}"><h3>{heading}</h3>', unsafe_allow_html=True)
    st.markdown(body if body else "_Nothing found in this video._")
    st.markdown("</div>", unsafe_allow_html=True)


def answer(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        reply = ask_question(st.session_state.result["rag_chain"], question)
    except Exception as e:
        reply = f"I couldn't answer that. Details: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})


# ──────────────────────────────────────────────────────────────
# Sidebar — input
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="mark">🎬</div>'
        '<div class="name">Video Assistant</div></div>',
        unsafe_allow_html=True,
    )

    input_mode = st.radio("Source", ["YouTube link", "Upload a file"], label_visibility="collapsed")

    url, uploaded = None, None
    if input_mode == "YouTube link":
        url = st.text_input("Video URL", placeholder="https://youtube.com/watch?v=…")
    else:
        uploaded = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a"],
        )

    language = st.selectbox("Spoken language", ["english", "hinglish"])

    go = st.button("Analyze video", type="primary", use_container_width=True)

    if st.session_state.result:
        st.divider()
        if st.button("Start over", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.rerun()

# ──────────────────────────────────────────────────────────────
# Run pipeline
# ──────────────────────────────────────────────────────────────
if go:
    source = url.strip() if url else (save_upload(uploaded) if uploaded else "")
    if not source:
        st.sidebar.error("Add a YouTube link or upload a file first.")
    else:
        try:
            st.session_state.result = run_pipeline_ui(source, language)
            st.session_state.messages = []
            st.session_state.source_label = url.strip() if url else uploaded.name
            st.session_state.language = language
            st.session_state.processed_at = datetime.now().strftime("%d %b %Y, %H:%M")
        except Exception as e:
            st.error(f"Something went wrong while processing the video: {e}")

# ──────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────
result = st.session_state.result

if not result:
    st.markdown(
        """
        <div class="hero">
            <h1>Turn any video into notes you can talk to.</h1>
            <p>Paste a YouTube link or upload a recording. You get a summary, the action items,
            the decisions that were made and the questions left open. Then ask anything about it.</p>
        </div>
        <div class="steps">
            <div class="step"><b>1. Add a video</b><span>A YouTube link, or a video or audio file from your device.</span></div>
            <div class="step"><b>2. Read the breakdown</b><span>Summary, action items, decisions and open questions in one place.</span></div>
            <div class="step"><b>3. Ask questions</b><span>Chat with the transcript and get answers grounded in what was said.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="result-head">
            <h1>{result['title']}</h1>
            <div class="chips">
                <span class="chip">{st.session_state.source_label}</span>
                <span class="chip">{st.session_state.language.capitalize()}</span>
                <span class="chip">Processed {st.session_state.processed_at}</span>
                <span class="chip">{len(result['transcript'].split()):,} words</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_chat, tab_transcript = st.tabs(["Overview", "Ask questions", "Transcript"])

    # Overview
    with tab_overview:
        panel("summary", "Summary", result["summary"])
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            panel("actions", "Action items", result["action_items"])
            panel("questions", "Open questions", result["open_questions"])
        with col2:
            panel("decisions", "Key decisions", result["key_decisions"])
            st.download_button(
                "Download full report (.md)",
                data=build_report(result),
                file_name="video_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # Chat
    with tab_chat:
        if not st.session_state.messages:
            st.markdown(
                '<p class="hint">Try one of these, or type your own question below.</p>',
                unsafe_allow_html=True,
            )
            suggestions = [
                "What were the main points?",
                "Who is responsible for what?",
                "What deadlines were mentioned?",
            ]
            cols = st.columns(len(suggestions))
            for col, text in zip(cols, suggestions):
                if col.button(text, key=f"sg_{text}", use_container_width=True):
                    with st.spinner("Thinking…"):
                        answer(text)
                    st.rerun()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎬"):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask something about this video"):
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar="🎬"):
                with st.spinner("Thinking…"):
                    answer(prompt)
                st.markdown(st.session_state.messages[-1]["content"])

    # Transcript
    with tab_transcript:
        st.markdown(
            f'<div class="transcript">{result["transcript"]}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download transcript (.txt)",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )