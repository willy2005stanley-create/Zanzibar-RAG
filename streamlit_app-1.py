"""
Bahari — Zanzibar Aqua Tourism RAG Dashboard
Run:  streamlit run streamlit_app.py

Reuses the vector store built in zanzibar_rag_pipeline.ipynb (vector_store.pkl).
Set GROQ_API_KEY as an environment variable, or paste one into the sidebar for
this session only.
"""

import os
import pickle
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from a local .env file if present

STORE_PATH = "vector_store.pkl"
GROQ_MODEL = os.environ.get("GROQ_MODEL","openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are Bahari, a friendly, practical assistant for people planning aqua tourism activities "
    "(diving, snorkeling, boat trips, island trips) in Zanzibar. Answer using ONLY the context passages "
    "provided — do not invent details that aren't in the context. If the context doesn't cover something, "
    "say so plainly. Keep answers concise (roughly 3-6 sentences), warm, and specific to Zanzibar. Any prices "
    "mentioned are approximate and should be confirmed locally."
)

st.set_page_config(page_title="Bahari — Zanzibar Aqua Tourism", page_icon="🌊", layout="centered")


# ---------- Load vector store (built by the notebook) ----------
@st.cache_resource
def load_store():
    if not os.path.exists(STORE_PATH):
        return None
    with open(STORE_PATH, "rb") as f:
        return pickle.load(f)


store = load_store()

if store is None:
    st.error(
        f"Couldn't find `{STORE_PATH}`. Run `zanzibar_rag_pipeline.ipynb` first — the last "
        "cells in Step 4 / \"Persist the vector store\" create this file."
    )
    st.stop()

chunks = store["chunks"]
embeddings = store["embeddings"]
use_sentence_transformers = store["use_sentence_transformers"]


# ---------- Embedding function (must match the notebook) ----------
@st.cache_resource
def get_embedder():
    if use_sentence_transformers:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(store["embed_model_name"])

        def embed(texts):
            return np.array(model.encode(texts, normalize_embeddings=True))

        return embed
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf.fit([c["text"] for c in chunks])

        def embed(texts):
            return normalize(tfidf.transform(texts).toarray())

        return embed


embed_texts = get_embedder()


def retrieve(question, k=4):
    q_vec = embed_texts([question])[0]
    sims = embeddings @ q_vec
    top_idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def build_context(retrieved):
    return "\n\n".join(f"[{c['title']}]\n{c['text']}" for c, _ in retrieved)


def ask_groq(question, context, api_key, model=GROQ_MODEL):
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"},
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    default_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API key",
        value=default_key,
        type="password",
        help="Get a free key at console.groq.com/keys. Read from GROQ_API_KEY if set.",
    )
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=4)
    st.caption(f"Vector store: {len(chunks)} chunks · "
               f"{'sentence-transformers' if use_sentence_transformers else 'TF-IDF fallback'} embeddings")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🌊 Bahari")
st.caption("Ask about diving, snorkeling and boat trips in Zanzibar — grounded in a local knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Render history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Retrieved sources"):
                for title, score in msg["sources"]:
                    st.markdown(f"- **{title}** — similarity {score:.3f}")

# ---------- Chat input ----------
question = st.chat_input("e.g. Where can I swim with dolphins?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            answer = "Please add a Groq API key in the sidebar to get an answer."
            sources = []
            st.markdown(answer)
        else:
            with st.spinner("Retrieving relevant info and generating an answer..."):
                retrieved = retrieve(question, k=top_k)
                context = build_context(retrieved)
                try:
                    answer = ask_groq(question, context, api_key)
                except Exception as e:
                    answer = f"Something went wrong calling Groq: {e}"
                sources = [(c["title"], s) for c, s in retrieved]
            st.markdown(answer)
            if sources:
                with st.expander("Retrieved sources"):
                    for title, score in sources:
                        st.markdown(f"- **{title}** — similarity {score:.3f}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
