# NLP_PROJECT_SEM6
# 📚 Live RAG System — Retrieval Depth Sensitivity Analysis

> A **Live Retrieval-Augmented Generation (RAG)** system that dynamically ingests documents, answers questions using Google Gemini, and rigorously evaluates how **retrieval depth (K)** affects answer quality — all through an interactive Streamlit dashboard.

---

## What This Project Does

Most RAG systems treat **K** (the number of retrieved passages) as a fixed, ignored constant. This project challenges that assumption.

**Key capabilities:**
-  **Live Document Ingestion** — Upload PDFs, DOCX, or TXT files; they are instantly chunked, embedded, and searchable without restarting.
-  **Semantic Search** — Uses `all-MiniLM-L6-v2` embeddings stored in **ChromaDB** for fast, accurate vector retrieval.
-  **LLM Answer Generation** — Passes retrieved chunks through **Google Gemini 1.5 Flash** with relevance filtering to prevent hallucination.
- **Retrieval Depth Evaluation** — Automatically evaluates K = 1 → 5, measuring how retrieval depth impacts answer quality.
-  **Document Contribution Analysis** — Shows which uploaded documents contributed most to a given answer.
-  **Interactive Dashboard** — Plotly charts inside Streamlit tabs for visual analysis of all metrics.

---

##  System Architecture

```text
User Query
    │
    ▼
Semantic Search (ChromaDB + MiniLM)
    │
    ▼
Relevance Filter (cosine similarity threshold)
    │
    ▼
Google Gemini 1.5 Flash (LLM)
    │
    ▼
Answer + Evaluation Pipeline (K=1..5)
    │
    ▼
Streamlit Dashboard (metrics, charts, contribution analysis)
