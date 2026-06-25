# iwm_retriever.py
import json, numpy as np, faiss, os
from openai import OpenAI

BASE = "iwm_memory"
DOCS = json.load(open(f"{BASE}/docs.json"))
EMBS = np.load(f"{BASE}/vectors.npy").astype("float32")
index = faiss.read_index(f"{BASE}/faiss.index")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search(query: str, k=6):
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
    hits = search(query, k=6)
    blocks = []
    titles = []
    for h in hits:
        titles.append(h["url"])
        blocks.append(f"[{h['url']}]\n{h['text']}")
    context = "\n\n--- IWM CONTEXT ---\n" + "\n\n---\n".join(blocks) if hits else ""
    citations = list(dict.fromkeys(titles))
    return context, citations
