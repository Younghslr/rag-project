# RAG Application — Architecture

![Architecture Diagram](docs/rag-architecture.svg)

---

## Components

### Startup — Document Ingestion (runs once per session)

ChromaDB is **in-memory** (`chromadb.Client()`). It resets every time the app restarts, so ingestion runs automatically at startup each session.

| Component | File | What it does |
|-----------|------|--------------|
| Source Docs | `data_loader.py` | Returns 20 hardcoded paragraphs covering Python, ML, databases, and AI |
| Embed Documents | `embeddings.py` | Batch-embeds all 20 docs via OpenAI `text-embedding-3-small` |
| ChromaDB | `vector_store.py` | Stores doc vectors in an in-memory collection named `"tech_docs"` |

---

### Per-Request — Query Pipeline

Every user question flows through these steps in order:

| Step | Component | File | What it does |
|------|-----------|------|--------------|
| 1 | Security Validator | `security.py` | Rejects empty input, queries over 500 chars, and 10 prompt-injection patterns |
| 2 | Query Rewriter | `workflow.py` | Calls `gpt-4o-mini` (temp=0.1) to rephrase the query for better semantic search; resolves pronouns using conversation history |
| 3 | Embed Query | `embeddings.py` | Embeds the rewritten query via `text-embedding-3-small` |
| 4 | ChromaDB Search | `vector_store.py` | Finds top-3 most similar documents by L2 distance (same collection as ingestion) |
| 5 | Similarity Filter | `filters.py` | Drops documents where L2 distance > 1.0 (`SIMILARITY_THRESHOLD`) |
| 5a | *(fallback)* | `filters.py` | If **no** documents pass the filter, returns a fallback message immediately — GPT-4o-mini is never called |
| 6 | GPT-4o-mini | `rag_pipeline.py` | Generates an answer using the filtered docs as context, plus conversation history injected into the prompt |
| 7 | Hallucination Monitor | `monitoring.py` | Second OpenAI call (temp=0.0) acting as LLM-as-judge — returns GROUNDED / PARTIAL / HALLUCINATED |
| 8 | Response | `app.py` | Displays answer, source documents, confidence score, grounding verdict, and timestamp |

---

### Memory — Conversation History

`conversation.py` stores the last 10 turns (user + assistant messages) in session state.

It feeds into **two** places each request:
- **Query Rewriter** — so vague follow-ups like "what else can it do?" resolve correctly
- **GPT-4o-mini generation prompt** — so the model has full prior context when answering

After each successful response, the rewritten query and answer are saved back to history.

---

### External Service — OpenAI API

Four components make API calls to OpenAI per query:

| Call | Component | Model | Purpose |
|------|-----------|-------|---------|
| ① | Embed Documents (startup) | `text-embedding-3-small` | Vectorize all 20 source docs |
| ② | Query Rewriter | `gpt-4o-mini` | Rephrase query for better retrieval |
| ③ | Embed Query | `text-embedding-3-small` | Vectorize the rewritten query |
| ④ | GPT-4o-mini + Hallucination Monitor | `gpt-4o-mini` | Generate answer + verify grounding |

---

## Data Flow Summary

```
STARTUP:
  Source Docs → [OpenAI embed] → ChromaDB (in-memory)

PER REQUEST:
  User query
    → Security (block injections / bad input)
    → Query Rewriter [OpenAI ②] + Conversation History
    → Embed Query [OpenAI ③]
    → ChromaDB similarity search (same store as startup)
    → Similarity Filter
        ├── no match → Fallback response (pipeline stops here)
        └── match → GPT-4o-mini [OpenAI ④] + Conversation History in prompt
                        → Hallucination Monitor [OpenAI ④]
                        → Response displayed
                        → Save turn to Conversation History
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local, in-memory) |
| Language | Python 3.14 |
| Security | Custom prompt-injection detection (`security.py`) |
| Monitoring | LLM-as-judge hallucination detection (`monitoring.py`) |
