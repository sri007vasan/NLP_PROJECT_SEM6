"""
config.py - Central configuration for the Live RAG System
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ─── Document Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 64        # character overlap between chunks

# ─── Retrieval ────────────────────────────────────────────────────────────────
K_VALUES = [1, 2, 3, 4, 5]          # retrieval depths to evaluate
DEFAULT_K = 3

# Adaptive retrieval thresholds (cosine similarity)
ADAPTIVE_HIGH_THRESHOLD = 0.85      # → K=1
ADAPTIVE_MID_THRESHOLD = 0.70       # → K=3
ADAPTIVE_LOW_K = 5                  # default when below mid threshold

# ─── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION = "live_rag_docs"

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_RESULTS_FILE = "evaluation_results.json"

# Quality score weights (must sum to 1.0)
QUALITY_WEIGHTS = {
    "recall":     0.35,
    "grounding":  0.35,
    "hallucination": 0.20,   # inverted: lower hallucination → higher score
    "latency":    0.10,      # inverted: lower latency → higher score
}
LATENCY_NORM_CAP = 10.0     # seconds — latency above this → 0 score

# ─── Upload ───────────────────────────────────────────────────────────────────
UPLOAD_DIR = "./uploaded_docs"
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]
