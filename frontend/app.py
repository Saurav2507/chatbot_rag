import json
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Sanskrit Document RAG", layout="wide")
st.title("Sanskrit Document RAG")


def render_sources(citations: list[dict], chunks: list[dict] | None = None):
    if citations:
        with st.expander("Sources"):
            for citation in citations:
                score = citation.get("relevance_score", 0.0)
                st.markdown(
                    f"**{citation.get('filename', '')}, page {citation.get('page_number', '?')}** "
                    f"(score: {score:.3f})"
                )
                st.caption(citation.get("text_snippet", ""))

    if chunks:
        with st.expander("Retrieved Chunks"):
            for index, chunk in enumerate(chunks, start=1):
                dense = chunk.get("dense_score", chunk.get("score", 0.0))
                lexical = chunk.get("lexical_score", 0.0)
                st.markdown(
                    f"**{index}. {chunk.get('filename', '')}, page {chunk.get('page_number', '?')}** "
                    f"(dense: {dense:.3f}, lexical: {lexical:.3f})"
                )
                st.text(chunk.get("text", ""))


def render_latency(latency: dict):
    if not latency:
        return
    with st.sidebar:
        st.subheader("Last Query Latency")
        df_latency = pd.DataFrame([latency]).T
        df_latency.columns = ["ms"]
        st.dataframe(df_latency, use_container_width=True)


def fallback_chat(prompt: str, top_k: int):
    with st.spinner("Generating answer..."):
        response = requests.post(f"{API_URL}/chat", json={"query": prompt, "top_k": top_k}, timeout=180)
        response.raise_for_status()
        data = response.json()

    answer = data.get("answer", "")
    citations = data.get("citations", [])
    chunks = data.get("retrieved_chunks", [])
    latency = data.get("latency_ms", {})

    st.markdown(answer)
    render_sources(citations, chunks)
    render_latency(latency)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations, "retrieved_chunks": chunks}
    )


with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
    if uploaded_file is not None and st.button("Upload and Ingest"):
        with st.spinner("Uploading document..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/octet-stream")}
            try:
                response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                response.raise_for_status()
                st.success(f"{uploaded_file.name} is being indexed in the background.")
            except requests.exceptions.RequestException as exc:
                error_message = f"Upload failed: {exc}"
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                    try:
                        detail = exc.response.json().get("detail")
                        if detail:
                            error_message += f"\n\n**Reason:** {detail}"
                    except json.JSONDecodeError:
                        pass  # Response was not JSON
                st.error(error_message)

    if st.button("Ingest Local Folder"):
        with st.spinner("Starting ingestion..."):
            try:
                response = requests.post(f"{API_URL}/ingest_folder", timeout=60)
                response.raise_for_status()
                st.success(response.json().get("message", "Ingestion started."))
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.header("Retrieval")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)
    stream_answers = st.toggle("Stream answers", value=True)

    try:
        status = requests.get(f"{API_URL}/status", timeout=5).json()
        st.caption(f"Collection: {status.get('collection')}")
        st.caption(f"Device: {status.get('device')}")
    except Exception:
        st.warning("Backend is not reachable.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("citations", []), message.get("retrieved_chunks", []))


if prompt := st.chat_input("Ask in Sanskrit, Devanagari, IAST, or transliteration..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not stream_answers:
            try:
                fallback_chat(prompt, top_k)
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend on port 8000.")
            except Exception as exc:
                st.error(f"Chat failed: {exc}")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/chat/stream",
                    json={"query": prompt, "top_k": top_k},
                    stream=True,
                    timeout=180,
                )
                response.raise_for_status()

                answer_placeholder = st.empty()
                full_answer = ""
                citations = []
                chunks = []
                latency = {}

                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if not data_str:
                        continue

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")
                    if event_type == "citations":
                        citations = event.get("data", [])
                    elif event_type == "chunks":
                        chunks = event.get("data", [])
                    elif event_type == "token":
                        full_answer += event.get("data", "")
                        answer_placeholder.markdown(full_answer + "...")
                    elif event_type == "done":
                        latency = event.get("data", {}).get("latency_ms", {})

                answer_placeholder.markdown(full_answer)
                render_sources(citations, chunks)
                render_latency(latency)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "citations": citations,
                        "retrieved_chunks": chunks,
                    }
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend on port 8000.")
            except Exception:
                try:
                    fallback_chat(prompt, top_k)
                except Exception as exc:
                    st.error(f"Chat failed: {exc}")
