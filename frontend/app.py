import streamlit as st
import requests
import json
import time
import pandas as pd

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

st.title("📚 Private PDF Knowledge Base")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Ingest"):
            with st.spinner("Uploading and processing..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"File {uploaded_file.name} is being processed in the background!")
                    else:
                        st.error("Failed to upload file.")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

    st.divider()
    st.header("Evaluation & Latency")
    st.write("Latency metrics will appear here after a query.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("Sources Used"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['filename']} - Page {c['page_number']}** (Score: {c['relevance_score']:.3f})")
                    st.caption(f"> {c['text_snippet']}")

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Try streaming endpoint first for real-time token display
        try:
            response = requests.post(
                f"{API_URL}/chat/stream",
                json={"query": prompt, "top_k": 3},
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                answer_placeholder = st.empty()
                full_answer = ""
                citations = []
                latency = {}
                
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    
                    if event.get("type") == "citations":
                        citations = event["data"]
                    elif event.get("type") == "token":
                        full_answer += event["data"]
                        answer_placeholder.markdown(full_answer + "▌")
                    elif event.get("type") == "done":
                        latency = event["data"].get("latency_ms", {})
                
                # Final render without cursor
                answer_placeholder.markdown(full_answer)
                
                # Show citations
                if citations:
                    with st.expander("Sources Used"):
                        for c in citations:
                            st.markdown(f"**{c['filename']} - Page {c['page_number']}** (Score: {c['relevance_score']:.3f})")
                            st.caption(f"> {c['text_snippet']}")
                
                # Update sidebar latency
                if latency:
                    with st.sidebar:
                        st.subheader("Last Query Latency")
                        df_latency = pd.DataFrame([latency]).T
                        df_latency.columns = ["ms"]
                        st.dataframe(df_latency)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_answer,
                    "citations": citations
                })
            else:
                # Fallback to non-streaming endpoint
                _fallback_chat(prompt)
                
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Make sure the server is running on port 8000.")
        except Exception as e:
            # Fallback to non-streaming on any streaming error
            try:
                _fallback_chat(prompt)
            except Exception as e2:
                st.error(f"Error connecting to backend: {e2}")


def _fallback_chat(prompt: str):
    """Non-streaming fallback using the original /chat endpoint."""
    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/chat",
            json={"query": prompt, "top_k": 3},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            citations = data["citations"]
            latency = data["latency_ms"]
            
            st.markdown(answer)
            
            if citations:
                with st.expander("Sources Used"):
                    for c in citations:
                        st.markdown(f"**{c['filename']} - Page {c['page_number']}** (Score: {c['relevance_score']:.3f})")
                        st.caption(f"> {c['text_snippet']}")
            
            with st.sidebar:
                st.subheader("Last Query Latency")
                df_latency = pd.DataFrame([latency]).T
                df_latency.columns = ["ms"]
                st.dataframe(df_latency)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "citations": citations
            })
        else:
            st.error("Failed to get response from server.")
