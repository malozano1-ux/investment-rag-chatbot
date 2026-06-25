# rag_retriever.py
# Standalone FAISS-based retriever using OpenAI embeddings.
# Used for searching a pre-built knowledge base of documents.
#
# Setup:
# 1. Place your docs in the `knowledge_base/` folder (see build_index.py).
# 2. Set OPENAI_API_KEY in your .env file.
# 3. Import and call search() or build_context() from your main app.

import json
import numpy as np
import faiss
import os
from openai import OpenAI

BASE = "knowledge_base"
DOCS = json.load(open(f"{BASE}/docs.json"))
EMBS = np.load(f"{BASE}/vectors.npy").astype("float32")
index = faiss.read_index(f"{BASE}/faiss.index")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search(query: str, k=6):
    """
    Returns the top-k most relevant document chunks for a given query.
    Each result includes the chunk text, source URL/path, and cosine similarity score.
    """
    q = client.embeddings.create(model="text-embedding-3-large", input=[query])
    qv = np.array([q.data[0].embedding], dtype="float32")
    faiss.normalize_L2(qv)
    D, I = index.search(qv, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(DOCS):
            hits.append({**DOCS[idx], "score": float(score)})
    return hits

def build_context(query: str):
    """
    Returns a formatted context string and list of source citations for a query.
    Suitable for direct injection into an LLM prompt.
    """
    hits = search(query, k=6)
    blocks = []
    titles = []
    for h in hits:
        titles.append(h["url"])
        blocks.append(f"[{h['url']}]\n{h['text']}")
    context = "\n\n--- RETRIEVED CONTEXT ---\n" + "\n\n---\n".join(blocks) if hits else ""
    citations = list(dict.fromkeys(titles))
    return context, citations
