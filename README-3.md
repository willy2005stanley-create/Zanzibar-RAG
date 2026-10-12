# Bahari — Zanzibar Aqua Tourism RAG

A full Retrieval-Augmented Generation pipeline (documents → chunks → embeddings →
vector database → question → retrieval → context → LLM → answer) for a Zanzibar
aqua-tourism assistant, using **Groq** for generation and a **Streamlit** dashboard
for the UI.

## Files

- `zanzibar_rag_pipeline.ipynb` — the notebook, one section per pipeline stage. Builds
  the embeddings + vector index and saves them to `vector_store.pkl`.
- `streamlit_app.py` — chat UI that loads `vector_store.pkl` and calls Groq.
- `documents/` — 24 `.txt` knowledge files (dive sites, tours, seasons, prices, safety, etc.)
- `requirements.txt`
- `.env.example` — template for your API key

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key: https://console.groq.com/keys

Copy the env template and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your-key-here
```

Both the notebook and `streamlit_app.py` load `.env` automatically via `python-dotenv`.
`.env` is already in `.gitignore` — it's never meant to be committed or shared.

## Run

1. Open and run all cells in `zanzibar_rag_pipeline.ipynb` top to bottom.
   This builds the embeddings/index and writes `vector_store.pkl`.
2. Launch the dashboard:
   ```bash
   streamlit run streamlit_app.py
   ```
   (If you didn't set `GROQ_API_KEY` as an environment variable, paste it into
   the sidebar field — it's kept only for that browser session.)

## Notes

- Embeddings default to `sentence-transformers` (`all-MiniLM-L6-v2`) and the vector
  index to `faiss`. If either package isn't installed, the notebook and app both fall
  back automatically to a TF-IDF + cosine-similarity approach so things still run —
  install the real packages for better semantic retrieval.
- To use your own knowledge base, replace the files in `documents/` with your own
  `.txt` files (title on the first line, body after a blank line), then re-run the
  notebook to rebuild `vector_store.pkl`.
- Prices and specifics in the sample knowledge base are illustrative — verify anything
  you plan to show real travelers.
