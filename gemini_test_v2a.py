"""
gemini_test.py — Routed RAG for WhatsApp (Meta Cloud API) + Gemini

Key improvements vs current version:
- Intent routing (funding / investing / academy / premium / paises / onboarding / general)
- Separate FAISS index per topic (retrieval only from relevant sources)
- Per-intent prompts (prevents unwanted academy/premium segways)
- Hard limits: TRANSCRIPT_MAX_CHUNKS + TRANSCRIPT_LOAD_TIMEOUT_SEC
- Safer WhatsApp formatting
"""

import os
import json
import logging
import datetime
import time
logging.info("BOOT CHECK: gemini_test_v1 loaded, time=%s", time.time())
from typing import List, Dict, Tuple, Optional, Any, Set

import numpy as np
import requests

from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

import faiss
from google.oauth2.service_account import Credentials
import gspread

from pypdf import PdfReader
import fitz

import re
import logging

import asyncio

# -----------------------------
# Logging
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# -----------------------------
# Env / Config
# -----------------------------
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "testtoken").strip()

META_TOKEN = os.getenv("WHATSAPP_TOKEN", "").strip()  # WhatsApp Cloud API token
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_ID", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
_raw = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001").strip()
GEMINI_EMBED_MODEL = _raw.split("models/", 1)[1] if _raw.startswith("models/") else _raw
logging.info("Using GEMINI_EMBED_MODEL=%s", GEMINI_EMBED_MODEL)

TRANSCRIPT_DISABLE_RAG = os.getenv("TRANSCRIPT_DISABLE_RAG", "0").strip() == "1"
TRANSCRIPT_TOP_K = int(os.getenv("TRANSCRIPT_TOP_K", "6"))
TRANSCRIPT_MAX_CHUNKS = int(os.getenv("TRANSCRIPT_MAX_CHUNKS", "800"))  # cap chunks across all docs
TRANSCRIPT_LOAD_TIMEOUT_SEC = float(os.getenv("TRANSCRIPT_LOAD_TIMEOUT_SEC", "120.0"))  # time budget

MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "1400"))
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "2400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dedupe for incoming WhatsApp messages (prevents double-processing on retries)
SEEN_LOCK = asyncio.Lock()
SEEN_MESSAGE_IDS: set[str] = set()

# Optional: cap memory so it doesn't grow forever
SEEN_MAX = int(os.getenv("SEEN_MAX", "5000"))

def pick_faiss_dir() -> str:
    # If you add a Render Disk later, you can point FAISS_DIR to it via env var.
    preferred = os.getenv("FAISS_DIR", "/var/data/faiss")

    try:
        os.makedirs(preferred, exist_ok=True)
        testfile = os.path.join(preferred, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        return preferred
    except Exception:
        # Always writable on Render, but not persistent across deploys
        fallback = "/tmp/faiss"
        os.makedirs(fallback, exist_ok=True)
        return fallback

FAISS_DIR = pick_faiss_dir()
CHUNKS_PATH = os.path.join(FAISS_DIR, "faiss_chunks.json")

# -----------------------------
# Gemini init
# -----------------------------
from google import genai

genai_client = None
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        # For text generation:
        gemini_model = GEMINI_MODEL  # keep as a string; e.g. "gemini-1.5-flash"
        logging.info("Gemini client ready (model=%s)", GEMINI_MODEL)
    except Exception:
        logging.exception("Failed to initialize Gemini (google-genai)")
else:
    logging.warning("No GEMINI_API_KEY set. Bot will respond with a fallback message.")

# -----------------------------
# Transcript sources
# Each tuple: (relative_or_abs_path, topic_name)
# IMPORTANT: topic_name is used for routing filters.
# -----------------------------
TRANSCRIPT_PDF_PATH     = os.getenv("TRANSCRIPT_PDF_PATH", "")    # fondeo luis fernando
TRANSCRIPT_PDF_PATH_3   = os.getenv("TRANSCRIPT_PDF_PATH_3", "")  # fondeo av
TRANSCRIPT_PDF_PATH_4   = os.getenv("TRANSCRIPT_PDF_PATH_4", "")  # insights av
TRANSCRIPT_PDF_PATH_5   = os.getenv("TRANSCRIPT_PDF_PATH_5", "")  # productos av
TRANSCRIPT_PDF_PATH_6   = os.getenv("TRANSCRIPT_PDF_PATH_6", "")  # productos lista
TRANSCRIPT_PDF_PATH_7   = os.getenv("TRANSCRIPT_PDF_PATH_7", "")  # academy isa
TRANSCRIPT_PDF_PATH_8   = os.getenv("TRANSCRIPT_PDF_PATH_8", "")  # academy lista
TRANSCRIPT_PDF_PATH_9   = os.getenv("TRANSCRIPT_PDF_PATH_9", "")  # premium isa
TRANSCRIPT_PDF_PATH_10   = os.getenv("TRANSCRIPT_PDF_PATH_10", "")  # home isa
TRANSCRIPT_PDF_PATH_13   = os.getenv("TRANSCRIPT_PDF_PATH_13", "")  # onboarding av
TRANSCRIPT_PDF_PATH_14   = os.getenv("TRANSCRIPT_PDF_PATH_14", "")  # paises av
TRANSCRIPT_PDF_PATH_15   = os.getenv("TRANSCRIPT_PDF_PATH_15", "")  # fondeos_ach av

TRANSCRIPT_SOURCES: List[Tuple[str, str]] = [
    # Onboarding
    ("transcripts/onboarding av.pdf", "onboarding"),
    
    # Funding
    ("transcripts/fondeos luisf.pdf", "fondeo"),
    ("transcripts/fondeos av.pdf", "fondeo"),
    ("transcripts/fondeos_ach av.pdf", "fondeo"),
    
    # Investing
    ("transcripts/productos av.pdf", "productos"),
    ("transcripts/productos lista.pdf", "productos"),
    
    # Academy 
    ("transcripts/academy isa.pdf", "academy"),
    ("transcripts/academy lista.pdf", "academy"),
    
    # Premium
    ("transcripts/premium isa.pdf", "premium"),
    
    #General
    ("transcripts/insights av.pdf", "insights"),
    ("transcripts/home isa.pdf", "home"),
    ("transcripts/paises av.pdf", "paises"),
]

# -----------------------------
# DB (conversation logging)
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatlog.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    wa_id = Column(String(64), index=True)
    direction = Column(String(10), index=True)  # "user" or "bot"
    conversation_id = Column(String(128), index=True, nullable=True)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)


def log_message(
    wa_id: str,
    direction: str,
    text: str,
    conversation_id: str | None = None,
) -> None:
    try:
        db = SessionLocal()
        db_msg = Message(
            wa_id=wa_id,
            direction=direction,
            conversation_id=conversation_id,
            text=text,
        )
        db.add(db_msg)
        db.commit()
    except Exception:
        logging.exception("Failed to log message")
    finally:
        try:
            db.close()
        except Exception:
            pass

def get_recent_history(wa_id: str, limit: int = 8) -> str:
    """
    Returns the last `limit` messages for this user in a simple transcript format.
    Uses your existing SQLAlchemy model: Message(wa_id, direction, text, created_at).
    """
    try:
        db = SessionLocal()
        rows = (
            db.query(Message)
              .filter(Message.wa_id == wa_id)
              .order_by(Message.created_at.desc())
              .limit(limit)
              .all()
        )
        rows = list(reversed(rows))  # oldest -> newest

        lines = []
        for r in rows:
            role = "Usuario" if r.direction == "user" else "Asistente"
            lines.append(f"{role}: {r.text}")

        return "\n".join(lines).strip()
    except Exception:
        logging.exception("Failed to load chat history")
        return ""
    finally:
        try:
            db.close()
        except Exception:
            pass

# -----------------------------
# In-memory indexes (per topic)
# -----------------------------
TOPIC_CHUNKS: Dict[str, List[Dict[str, str]]] = {}   # topic -> [{"text","source"}, ...]
TOPIC_INDEX: dict[str, faiss.Index] = {}                    # topic -> faiss index
TOPIC_DIM: Optional[int] = None

# -----------------------------
# Helpers
# -----------------------------
def _abs_path(p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(BASE_DIR, p)

def chunk_text(text: str, min_chars: int, max_chars: int, overlap: int) -> List[str]:
    """
    Chunk by paragraphs with overlap, aiming for stable retrieval.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]

    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
            else:
                # force append even if small (avoid losing info)
                if buf:
                    chunks.append(buf)
            buf = p

    if buf:
        chunks.append(buf)

    # Add overlap by trailing chars
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        prev_tail = ""
        for c in chunks:
            if prev_tail:
                overlapped.append((prev_tail + "\n\n" + c).strip())
            else:
                overlapped.append(c)
            prev_tail = c[-overlap:] if len(c) > overlap else c
        chunks = overlapped

    return chunks

def read_pdf_text(path: str) -> str:
    """
    PDF -> text (best effort).
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception:
        logging.exception("Failed to read PDF: %s", path)
        return ""

def embed_texts_gemini(texts: List[str]) -> np.ndarray:
    """
    Embed multiple documents for retrieval (google-genai).
    Robust to missing client / transient API errors.
    Returns float32 array of shape (n_texts, dim) when successful,
    or (0, 0) when nothing could be embedded.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    if genai_client is None:
        logging.warning("genai_client is None; returning empty embeddings.")
        return np.zeros((0, 0), dtype="float32")

    vectors: List[np.ndarray] = []
    expected_dim = None

    for i, t in enumerate(texts):
        try:
            resp = genai_client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=t,
            )
            vec = np.asarray(resp.embeddings[0].values, dtype="float32")

            if expected_dim is None:
                expected_dim = int(vec.shape[0])
            elif int(vec.shape[0]) != expected_dim:
                logging.error(
                    "Embedding dim mismatch at i=%d got=%d expected=%d; skipping.",
                    i, int(vec.shape[0]), expected_dim
                )
                continue

            vectors.append(vec)

        except Exception:
            logging.exception("Embedding failed at i=%d; skipping chunk.", i)

    if not vectors:
        return np.zeros((0, 0), dtype="float32")

    return np.vstack(vectors)

def embed_query_gemini(q: str) -> np.ndarray:
    if not q:
        return np.zeros((1, 0), dtype="float32")
    if genai_client is None:
        logging.warning("genai_client is None; returning empty query embedding.")
        return np.zeros((1, 0), dtype="float32")

    resp = genai_client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=q)
    vec = np.asarray(resp.embeddings[0].values, dtype="float32")
    return vec.reshape(1, -1)

# -----------------------------
# Intent Routing (the “training” lever)
# -----------------------------
def classify_intent(user_text: str) -> str:
    """
    Returns one of: funding | investing | academy | premium | general
    """
    t = (user_text or "").lower().strip()

    # Funding / deposits / withdrawals
    if any(k in t for k in [
        "deposit", "depósito", "deposito", "fonde", "fondeo", "recargar",
        "retirar", "retiro", "withdraw", "withdrawal"
    ]):
        return "funding"

    # Academy / learning
    if any(k in t for k in [
        "academy", "curso", "cursos", "clase", "clases", "aprender",
        "educación", "educacion"
    ]):
        return "academy"

    # Premium / advisor
    if any(k in t for k in [
        "premium", "asesor", "asesoría", "asesoria", "llamada", "advisor", "wealth", "suscripción", "anual", "mensual", "planes"
    ]):
        return "premium"

    # Investing basics
    if any(k in t for k in [
        "invert", "inversión", "inversion", "etf", "acciones", "bonos",
        "portafolio", "riesgo", "diversific", "diversificación", "diversificacion"
    ]):
        return "investing"
    
    # Onboarding
    if any(k in t for k in [
        "abrir cuenta", "cuenta abierta", "cedula", "pasaporte", "identificación", "proof of address", "comprobante de domicilio", "PPT", "permiso de proteccion temporal", "cedula de extranjeria"
    ]):
        return "cuenta abierta"

    return "general"

# Map intent → transcript topic labels
TOPIC_MAP = {
    "funding": "fondeo",
    "investing": "productos",
    "academy": "academy",
    "premium": "premium",
    "general": "insights",  # or "home"
    "cuenta abierta": "onboarding",
    "country": "paises",
}

def retrieve_context_by_topic(query: str, topic: str, k: int):
    mapped_topic = TOPIC_MAP.get(topic, topic)
    return retrieve_context(query, topic=mapped_topic, k=k)

# -----------------------------
# Build per-topic FAISS indices (with limits)
# -----------------------------
def load_chunks_cache(path: str):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logging.exception("Failed to load chunks cache at %s", path)
        return {}


def save_chunks_cache(chunks_dict: dict, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks_dict, f, ensure_ascii=False)
        logging.info("Saved chunks cache to %s", path)
    except Exception:
        logging.exception("Failed to save chunks cache at %s", path)

def load_faiss_cache(base_dir: str):
    topic_index = {}

    if not os.path.exists(base_dir):
        return topic_index

    for fname in os.listdir(base_dir):
        if not fname.endswith(".index"):
            continue

        topic = fname.replace(".index", "")
        path = os.path.join(base_dir, fname)
        topic_index[topic] = faiss.read_index(path)

    logging.info("Loaded FAISS indices: %s", list(topic_index.keys()))
    return topic_index


def save_faiss_cache(topic_index_dict: dict, base_dir: str):
    os.makedirs(base_dir, exist_ok=True)

    for topic, index in topic_index_dict.items():
        if not isinstance(index, faiss.Index):
            logging.warning("Skipping FAISS save for topic=%s (not a faiss.Index)", topic)
            continue

        path = os.path.join(base_dir, f"{topic}.index")
        faiss.write_index(index, path)
        logging.info("Saved FAISS index for topic=%s → %s", topic, path)


def build_transcript_indexes() -> None:
    global TOPIC_CHUNKS, TOPIC_INDEX, TOPIC_DIM

    if TRANSCRIPT_DISABLE_RAG:
        logging.info("RAG disabled (TRANSCRIPT_DISABLE_RAG=1).")
        TOPIC_CHUNKS, TOPIC_INDEX, TOPIC_DIM = {}, {}, None
        return

    start_time = time.time()
    loaded_chunks = 0

    topic_docs: Dict[str, List[Dict[str, str]]] = {}

    logging.info("Building transcript indices (timeout=%.1fs max_chunks=%d)", TRANSCRIPT_LOAD_TIMEOUT_SEC, TRANSCRIPT_MAX_CHUNKS)

    for raw_path, topic in TRANSCRIPT_SOURCES:
        if time.time() - start_time > TRANSCRIPT_LOAD_TIMEOUT_SEC:
            logging.warning("Transcript load timed out early. Loaded_chunks=%d", loaded_chunks)
            break
        if loaded_chunks >= TRANSCRIPT_MAX_CHUNKS:
            logging.warning("Reached TRANSCRIPT_MAX_CHUNKS cap (%d).", TRANSCRIPT_MAX_CHUNKS)
            break

        p = _abs_path(raw_path)
        if not p or not os.path.exists(p):
            logging.warning("Missing transcript topic=%s path=%s", topic, p)
            continue

        text = read_pdf_text(p)
        if not text:
            logging.warning("Empty/unreadable transcript topic=%s path=%s", topic, p)
            continue

        pieces = chunk_text(text, MIN_CHARS_PER_CHUNK, MAX_CHARS_PER_CHUNK, CHUNK_OVERLAP_CHARS)
        for piece in pieces:
            if loaded_chunks >= TRANSCRIPT_MAX_CHUNKS:
                break
            topic_docs.setdefault(topic, []).append({"text": piece, "source": os.path.basename(p)})
            loaded_chunks += 1

    if not topic_docs:
        logging.warning("No transcript chunks loaded. RAG will be empty.")
        TOPIC_CHUNKS, TOPIC_INDEX, TOPIC_DIM = {}, {}, None
        return

    # Build each topic index
    TOPIC_CHUNKS = {}
    TOPIC_INDEX = {}
    TOPIC_DIM = None

    for topic, docs in topic_docs.items():
        texts = [d["text"] for d in docs]
        X = embed_texts_gemini(texts)
        if X.size == 0:
            continue

        dim = int(X.shape[1])
        if TOPIC_DIM is None:
            TOPIC_DIM = dim
        elif TOPIC_DIM != dim:
            logging.error("Embedding dim mismatch for topic=%s (got %d expected %d)", topic, dim, TOPIC_DIM)
            continue

        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(dim)
        index.add(X)

        TOPIC_CHUNKS[topic] = docs
        TOPIC_INDEX[topic] = index

        logging.info("Index ready topic=%s chunks=%d dim=%d", topic, len(docs), dim)

RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))  # ajusta luego con pruebas

def retrieve_context(query: str, topic: str, k: int):
    """
    Returns: (context_str, best_score)
    """
    if TRANSCRIPT_DISABLE_RAG:
        return "", 0.0
    q = (query or "").strip()
    if not q:
        return "", 0.0
    if topic not in TOPIC_INDEX or topic not in TOPIC_CHUNKS:
        return "", 0.0

    idx = TOPIC_INDEX[topic]
    chunks = TOPIC_CHUNKS[topic]

    qvec = embed_query_gemini(q)
    faiss.normalize_L2(qvec)
    D, I = idx.search(qvec, k)

    # FAISS IndexFlatIP con vectores normalizados => score ~ cosine similarity
    best_score = float(D[0][0]) if D is not None and len(D) and len(D[0]) else 0.0

    picked = []
    for j in I[0].tolist():
        if j < 0 or j >= len(chunks):
            continue
        c = chunks[j]
        picked.append(f"[{topic} | {c['source']}]\n{c['text']}")

    return "\n\n---\n\n".join(picked).strip(), best_score

# -----------------------------
# Prompts (per intent)
# -----------------------------
BASE_GUARDRAILS = """
Eres asesorIA de Insights.

Jerarquía de conocimiento (OBLIGATORIA):
1) Estas reglas del sistema (SYSTEM RULES / BASE GUARDRAILS) tienen máxima prioridad.
2) Luego, usa como fuente principal la información interna provista en el contexto (transcripts/FAQ internos recuperados por RAG).
3) Solo si NO hay información relevante en los transcripts/FAQ internos, puedes usar conocimiento general para responder.

Reglas generales:
- Responde en español claro y accionable.
- Educación general: NO des recomendaciones personalizadas ni “lo mejor para ti”.
- NO recomiendes tickers específicos.
- Si falta un dato clave para dar pasos concretos (ej: país, método de depósito, etc.), pide SOLO 1 pregunta corta.
- NO promociones Academy/Premium a menos que el usuario pregunte por educación/asesoría o sea un follow-up natural y relevante.
"""

SYSTEM_RULES = """
Eres consultorIA de Insights.

Misión:
- Ayudar a usuarios con dudas sobre: depósitos/retiros, uso de la app, conceptos generales de inversión y productos de Insights.

Jerarquía de fuentes (OBLIGATORIA):
- Prioridad 1: Reglas de este sistema (cumplimiento, tono, límites).
- Prioridad 2: Información interna provista en el contexto (transcripts/FAQ internos del onboarding y otros materiales internos). Esta es la fuente principal de conocimiento.
- Prioridad 3: Conocimiento general SOLO si el contexto interno no contiene información relevante o suficiente.

Reglas de uso de transcripts (RAG-first):
- Siempre asume que la respuesta puede estar en los transcripts y prioriza buscar/usar esa información.
- Si el contexto interno incluye información relevante, responde basándote en esa información y NO completes con suposiciones de conocimiento general.
- Si NO hay información relevante en el contexto interno:
  - Puedes responder con conocimiento general, pero aclara de forma breve que es una explicación general y puede variar según políticas internas.
  - Si falta un dato clave para poder responder o para recuperar mejor información (por ejemplo país, tipo de cuenta, método), haz SOLO 1 pregunta corta.

Reglas de respuesta:
- Responde en español claro, directo y accionable.
- Responde de forma conversacional, como una persona por WhatsApp (no como un manual).
- Explica bien, pero sin hacer la respuesta larga o densa.

ESTILO Y LONGITUD (MUY IMPORTANTE):
- Máximo 6-10 líneas de texto total.
- Evita párrafos largos o explicaciones innecesarias.
- No respondas ni demasiado corto ni demasiado largo.
- Es importante que respondas las preguntas con claridad y detalle, sin necesidad de ser muy largo.  

TONO:
- Natural, cercano y útil.
- Evita frases formales como:
  "A continuación te explico..." o "Es importante destacar que..."
  
  FLUJO IDEAL:
1. 1 frase corta y natural
2. Explicación clara (bullets o frases)
3. 1 pregunta corta para continuar la conversación

Reglas adicionales:
- Educación general: NO des recomendaciones personalizadas ni “lo mejor para ti”.
- NO recomiendes tickers, acciones específicas, o portafolios concretos.
- Prioriza responder la pregunta del usuario. No cambies de tema.
- NO menciones Academy a menos que:
  (a) el usuario lo pregunte explícitamente, o
  (b) sea un siguiente paso natural.
- Puedes mencionar Premium solo si es natural en la conversación y de forma breve.
- Si falta un dato clave para dar pasos concretos, haz SOLO 1 pregunta corta.
- Si detectas frustración (ej. "no funciona", "estoy harto", "no entiendo nada"), 
  pide disculpas y ofrece el link del centro de ayuda: https://help.insightswm.com/hc/es-419
- Si el usuario pide explícitamente hablar con un humano o soporte técnico, indícale que puede encontrar todas las opciones de contacto en nuestro centro de ayuda: https://help.insightswm.com/hc/es-419
  
Formato WhatsApp:
- Usa bullets “•” cuando tenga sentido.
- Párrafos cortos.
- Evita bloques de texto largos.
""".strip()

def classify_intent(user_text: str) -> str:
    t = (user_text or "").lower()
    if any(k in t for k in ["deposit", "depósito", "deposito", "fonde", "fondeo", "retirar", "retiro", "withdraw"]):
        return "funding"
    if any(k in t for k in ["academy", "curso", "cursos", "clase", "aprender", "educación", "educacion"]):
        return "academy"
    if any(k in t for k in ["premium", "asesor", "asesoría", "asesoria", "llamada", "advisor", "suscripción", "anual", "mensual", "planes"]):
        return "premium"
    if any(k in t for k in ["invert", "inversión", "inversion", "acciones", "bonos", "etf", "portafolio", "riesgo", "diversific", "efectivo rentable", "money market", "single stocks", "acciones y ETFs", "portafolio gestionado", "managed portfolio", "portafolio temático", "themed portfolio", "portafolio IA", "AI portfolio", "portafolio BlackRock", "Crypto", "Cash"]):
        return "investing"
    if any(k in t for k in ["onboarding", "abrir cuenta", "apertura", "crear cuenta", "documentos", "documento", "identificación", "identificacion", "cédula", "cedula", "ine", "dni", "licencia", "id", "pasaporte", "proof of address", "comporbante de domicilio", "PPT", "permiso de protección temporal", "cedula de extranjería"]):
        return "onboarding"
    if any(k in t for k in ["pais", "paises", "residencia"]):
        return "paises"
    if any(k in t for k in [
        "ayuda adicional", "hablar con alguien", "soporte", "humano", "agente", "no entiendo", "no me ayudas", "estoy frustrado", "queja", "reclamo","centro de ayuda", "link de ayuda", "servicio al cliente"
    ]):
        return "ayuda_externa"
    return "general"

def mode_instruction(intent: str) -> str:
    if intent == "funding":
        return "Modo: FONDEO/RETIROS. Da pasos dentro de la app, tiempos, comisiones si aplica. Checklist + 1 pregunta si falta algo."
    if intent == "investing":
        return "Modo: CÓMO INVERTIR. Explica el proceso general + pasos en la app + los productos que tenemos en el app. Sin recomendaciones específicas."
    if intent == "academy":
        return "Modo: ACADEMY. Explica qué es, cómo acceder y qué incluye."
    if intent == "premium":
        return "Modo: PREMIUM. Explica qué incluye, cómo funciona, cuanto cuesta, y cómo solicitarlo. No des todo el detalle en un solo mensaje."
    if intent == "onboarding":
        return "Modo: ONBOARDING. Da pasos dentro de la app, documentos que necesitan y preguntas que tengan."
    if intent == "paises":
        return "Modo: PAISES. Da los paises en los que Insights está disponible."
    if intent == "ayuda_externa":
        return (
            "Modo: SOPORTE EXTERNO. El usuario parece frustrado o necesita ayuda que no puedes proveer. "
            "Debes ser empático, pedir disculpas si es necesario y proporcionar OBLIGATORIAMENTE "
            "el link al centro de ayuda: https://help.insightswm.com/hc/es-419"
        )
    return "Modo: GENERAL. Resuelve la duda y guía al siguiente paso."

def build_prompt(user_text: str, intent: str, context: str) -> str:
    return f"""
{SYSTEM_RULES}

{mode_instruction(intent)}

Contexto interno (úsalo solo si es relevante):
{context if context else "(sin contexto recuperado)"}

Usuario: {user_text}

Respuesta:
""".strip()

def _sanitize_for_whatsapp(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)           # **bold** -> *bold*
    text = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", text)          # "- item" -> "• item"
    text = re.sub(r"\n{3,}", "\n\n", text)                   # collapse blank lines
    return text.strip()

# -----------------------------
# Google Sheets (logging)
# -----------------------------
import base64

# Accept both env var naming conventions
GSHEET_ID = (
    os.getenv("GSHEET_ID", "").strip()
    or os.getenv("GOOGLE_SHEET_ID", "").strip()
)

GSHEET_TAB = (
    os.getenv("GSHEET_TAB", "").strip()
    or os.getenv("GOOGLE_SHEET_NAME", "Sheet1").strip().strip('"')
)

GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON", "").strip()
GOOGLE_CREDS_JSON_B64 = os.getenv("GOOGLE_CREDS_JSON_B64", "").strip()

_gsheet = None


def _load_creds_dict():
    if GOOGLE_CREDS_JSON:
        return json.loads(GOOGLE_CREDS_JSON)

    if GOOGLE_CREDS_JSON_B64:
        decoded = base64.b64decode(GOOGLE_CREDS_JSON_B64).decode("utf-8")
        return json.loads(decoded)

    if GOOGLE_CREDS_FILE:
        path = GOOGLE_CREDS_FILE
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def get_sheet():
    global _gsheet
    if _gsheet is not None:
        return _gsheet

    logging.info(
        "Sheets config check: GSHEET_ID_set=%s GSHEET_TAB=%s CREDS_JSON_set=%s CREDS_B64_set=%s",
        bool(GSHEET_ID),
        GSHEET_TAB,
        bool(GOOGLE_CREDS_JSON),
        bool(GOOGLE_CREDS_JSON_B64),
    )

    if not GSHEET_ID:
        logging.error("GSHEET_ID missing.")
        return None

    creds_dict = _load_creds_dict()
    if not creds_dict:
        logging.error("No Google Sheets credentials available.")
        return None

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        sh = client.open_by_key(GSHEET_ID)
        try:
            ws = sh.worksheet(GSHEET_TAB)
        except Exception:
            logging.warning("Worksheet %r not found. Falling back to first sheet.", GSHEET_TAB)
            ws = sh.get_worksheet(0)

        _gsheet = ws
        logging.info("Google Sheet connected OK.")
        return _gsheet

    except Exception:
        logging.exception("Failed to connect to Google Sheet.")
        return None


def log_to_sheet(wa_id: str, direction: str, text: str, message_id: str = ""):
    ws = get_sheet()
    if ws is None:
        logging.error("log_to_sheet skipped because sheet connection is None.")
        return

    try:
        row = [datetime.datetime.utcnow().isoformat(), wa_id, direction, message_id, text]
        ws.append_row(row, value_input_option="RAW")
        logging.info("Appended row to Google Sheet: wa_id=%s direction=%s msg_id=%s", wa_id, direction, message_id)
    except Exception:
        logging.exception("Failed to append row to Google Sheet.")

def log_loaded_transcripts():
    """
    Logs how many chunks were loaded per topic. Safe if indexes aren't built.
    """
    try:
        if not TOPIC_CHUNKS:
            logging.warning("No transcript chunks loaded (TOPIC_CHUNKS empty).")
            return

        total = 0
        for topic, docs in TOPIC_CHUNKS.items():
            n = len(docs)
            total += n
            logging.info("Transcript topic=%s chunks=%d", topic, n)
        logging.info("Transcript total chunks=%d", total)

    except Exception:
        logging.exception("log_loaded_transcripts failed")

# -----------------------------
# Core reply function
# -----------------------------
def generate_reply(user_text: str, wa_id: str):
    try:
        if genai_client is None or not gemini_model:
            return "¡Hola! 😊 Soy consultorIA de Insights. ¿En qué te puedo ayudar hoy?", {}

        user_text = (user_text or "").strip()
        if not user_text:
            return "¿Qué te gustaría hacer hoy: depositar, retirar o invertir?", {"intent": "empty"}

        # 1) Route intent (new)
        intent = classify_intent(user_text)

        # 2) Retrieve transcript context (RAG) — topic-limited (new)
        # NOTE: implement retrieve_context_by_topic(...) using your existing retrieval + filtered chunks/index
        context, best_score = retrieve_context_by_topic(user_text, topic=intent, k=TRANSCRIPT_TOP_K)
        has_rag = bool(context) and (best_score >= RAG_MIN_SCORE)

        logging.info(
            "RAG topic=%s has_rag=%s best_score=%.3f context_length=%d",
            intent, has_rag, best_score, len(context or "")
        )

        # 3) Retrieve recent chat history (keep your existing function)
        history = get_recent_history(wa_id, limit=8)

        # 4) One-line “mode” steering (no PROMPTS dict)
        mode = mode_instruction(intent)

        # 5) Use transcripts as primary source of truth
        if has_rag:
            rag_policy = (
                "POLÍTICA DE FUENTES: Usa como fuente principal el Contexto interno (transcripts). "
                "Si el contexto interno contiene información relevante, NO completes con suposiciones de conocimiento general."
            )
        else:
            rag_policy = (
                "POLÍTICA DE FUENTES: No se encontró contexto interno relevante para esta pregunta. "
                "Puedes responder con conocimiento general, y si falta un dato clave haz SOLO 1 pregunta corta."
            )

        # 6) Build the prompt (still SYSTEM_RULES)
        prompt = f"""
{SYSTEM_RULES}

{rag_policy}
{mode}

Historial reciente (útil para continuidad):
{history if history else "(sin historial)"}

Contexto interno (transcripts recuperados):
{context if has_rag else "(sin contexto interno relevante)"}

Usuario: {user_text}
Asistente:
""".strip()

        # 7) Call Gemini
        response = genai_client.models.generate_content(
            model=gemini_model,   # aquí gemini_model ES un string tipo "gemini-1.5-flash"
            contents=prompt,
        )

        # Extraer texto de forma robusta
        reply_text = (getattr(response, "text", "") or "").strip()
        if not reply_text:
            try:
                reply_text = (
                    response.candidates[0].content.parts[0].text
                ).strip()
            except Exception:
                reply_text = ""

        # 8) WhatsApp formatting cleanup (use your existing helper)
        reply_text = _sanitize_for_whatsapp(reply_text)

        return reply_text, {
            "intent": intent,
            "context_chars": len(context or ""),
            "prompt_chars": len(prompt),
        }

    except Exception:
        logging.exception("generate_reply failed")
        return "Tuve un problema generando la respuesta. ¿Puedes intentar de nuevo?", {"error": "generate_reply_failed"}

# -----------------------------
# WhatsApp send helpers
# -----------------------------
def send_whatsapp_text(to_number: str, message: str, phone_number_id: Optional[str] = None) -> Dict[str, Any]:
    if not META_TOKEN:
        return {"error": "META_TOKEN missing"}
    sender_phone_id = phone_number_id or PHONE_NUMBER_ID
    if not sender_phone_id:
        return {"error": "PHONE_NUMBER_ID missing"}

    url = f"https://graph.facebook.com/v22.0/{sender_phone_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message},
    }
    headers = {"Authorization": f"Bearer {META_TOKEN}", "Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        data = {"status_code": resp.status_code, "text": resp.text}
        if resp.status_code >= 300:
            logging.error("WhatsApp send failed: %s", data)
        return data
    except Exception:
        logging.exception("WhatsApp send exception")
        return {"error": "exception"}

def send_whatsapp_template(
    to_number: str,
    template_name: str,
    language_code: str = "es",
    phone_number_id: Optional[str] = None,
    components: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Send a WhatsApp template message.
    Log it immediately after sending (recommended) because webhook won't always
    include the full outbound template content.
    """
    sender_phone_id = phone_number_id or os.getenv("WHATSAPP_PHONE_ID")
    token = os.getenv("WHATSAPP_TOKEN")

    if not sender_phone_id:
        raise RuntimeError("Missing sender phone_number_id (and WHATSAPP_PHONE_ID not set).")
    if not token:
        raise RuntimeError("Missing WHATSAPP_TOKEN.")

    url = f"https://graph.facebook.com/v22.0/{sender_phone_id}/messages"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    template_obj: Dict[str, Any] = {
        "name": template_name,
        "language": {"code": language_code},
    }
    if components:
        template_obj["components"] = components

    payload: Dict[str, Any] = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "template",
        "template": template_obj,
    }

    logging.info("Outgoing template payload to Meta: %s", json.dumps(payload, ensure_ascii=False))

    r = requests.post(url, headers=headers, json=payload, timeout=20)

    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}

    if r.status_code >= 300:
        logging.error(
            "Template send failed status=%s from_phone_id=%s to=%s resp=%s",
            r.status_code, sender_phone_id, to_number, r.text
        )
        return data

    logging.info(
        "Template send ok from_phone_id=%s to=%s resp=%s",
        sender_phone_id, to_number, r.text
    )
    return data

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

# Meta may retry deliveries; dedup prevents double replies.
# NOTE: This is in-memory (resets on deploy). For production use Redis/DB with TTL.
SEEN_MESSAGE_IDS: Set[str] = set()

import asyncio
from collections import defaultdict

USER_LOCKS = defaultdict(asyncio.Lock)

# If true, ignore webhook events that are not for WHATSAPP_PHONE_ID.
# For debugging, keep this False (so you don’t accidentally ignore all events).
ENFORCE_PHONE_ID = os.getenv("ENFORCE_PHONE_ID", "false").lower() == "true"

@app.on_event("startup")
async def startup_event():
    global TOPIC_INDEX, TOPIC_CHUNKS, TOPIC_DIM

    os.makedirs(FAISS_DIR, exist_ok=True)

    # Load caches
    TOPIC_INDEX = load_faiss_cache(FAISS_DIR) or {}
    TOPIC_CHUNKS = load_chunks_cache(CHUNKS_PATH) or {}

    if TOPIC_INDEX and TOPIC_CHUNKS:
        logging.info("Loaded FAISS (%d topics) + chunks (%d topics) from cache",
                     len(TOPIC_INDEX), len(TOPIC_CHUNKS))
        log_loaded_transcripts()
        logging.info("Application startup complete.")
        return

    logging.info("FAISS cache/chunks missing → attempting to build transcript indexes...")

    try:
        # IMPORTANT: don't let this crash startup
        build_transcript_indexes()
    except Exception:
        logging.exception("Transcript index build failed; starting WITHOUT RAG.")
        TOPIC_INDEX, TOPIC_CHUNKS, TOPIC_DIM = {}, {}, None
        log_loaded_transcripts()
        logging.info("Application startup complete (RAG disabled due to build failure).")
        return

    # Only save if we actually built something
    if TOPIC_INDEX and TOPIC_CHUNKS:
        save_faiss_cache(TOPIC_INDEX, FAISS_DIR)
        save_chunks_cache(TOPIC_CHUNKS, CHUNKS_PATH)
        logging.info("Built + saved transcript indexes (%d topics).", len(TOPIC_INDEX))
    else:
        logging.warning("Index build produced no topics; not saving empty caches.")

    log_loaded_transcripts()
    logging.info("Application startup complete.")

@app.get("/")
async def root():
    return {"status": "ok"}

@app.head("/")
async def head_root():
    return Response(status_code=200)

# -----------------------------
# Webhook verification (GET)
# -----------------------------
@app.get("/debug/sheets-test")
async def sheets_test():
    ws = get_sheet()
    if ws is None:
        return {"ok": False, "error": "get_sheet() returned None. Check logs for the reason."}

    try:
        ws.append_row([datetime.datetime.utcnow().isoformat(), "debug", "debug", "debug", "hello from render"], value_input_option="RAW")
        return {"ok": True, "tab": GSHEET_TAB}
    except Exception as e:
        logging.exception("sheets-test append failed")
        return {"ok": False, "error": str(e)}

@app.get("/debug/rag-stats")
async def rag_stats():
    return {
        "chunks_loaded": len(CHUNKS),
        "index_ready": INDEX is not None,
        "embed_dim": EMBED_DIM,
        "top_k": TRANSCRIPT_TOP_K,
        "rag_disabled": TRANSCRIPT_DISABLE_RAG,
    }

@app.get("/webhook")
async def verify_token(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_challenge: str = Query("", alias="hub.challenge"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
):
    # WhatsApp verification:
    # ?hub.mode=subscribe&hub.verify_token=...&hub.challenge=...
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")

def extract_user_text(message: Dict[str, Any]) -> Optional[str]:
    """
    Normalize inbound WhatsApp message into a single user_text string.

    Supports:
    - text
    - button (older quick reply format)
    - interactive (button_reply, list_reply)  ✅ template buttons land here
    - media captions (image/video/document)
    """
    if not message:
        return None

    mtype = (message.get("type") or "").strip()

    # Normal text
    if mtype == "text":
        text = ((message.get("text") or {}).get("body") or "").strip()
        return text or None

    # Quick reply buttons (older format)
    if mtype == "button":
        btn = message.get("button") or {}
        text = (btn.get("payload") or btn.get("text") or "").strip()
        return text or None

    # Interactive replies (common for template buttons + list replies)
    if mtype == "interactive":
        inter = message.get("interactive") or {}
        itype = (inter.get("type") or "").strip()

        if itype == "button_reply":
            br = inter.get("button_reply") or {}
            # Prefer id for deterministic routing; fall back to title
            val = (br.get("id") or br.get("title") or "").strip()
            return val or None

        if itype == "list_reply":
            lr = inter.get("list_reply") or {}
            val = (lr.get("id") or lr.get("title") or "").strip()
            return val or None

        # If Meta changes structure, at least return something non-empty
        return str(inter).strip() or None

    # Media with captions
    if mtype in {"image", "video", "document"}:
        caption = ((message.get(mtype) or {}).get("caption") or "").strip()
        return caption or f"[{mtype}]"

    return None

async def process_webhook_payload(payload: dict, inbound_phone_number_id: str | None = None):
    try:
        entries = payload.get("entry", []) or []

        for entry in entries:
            changes = entry.get("changes", []) or []

            for change in changes:
                value = change.get("value") or {}

                # ---- Metadata (which phone number received the message) ----
                metadata = value.get("metadata") or {}
                display_phone = metadata.get("display_phone_number")
                meta_phone_number_id = metadata.get("phone_number_id")
                phone_number_id = meta_phone_number_id or inbound_phone_number_id

                logging.info(
                    "Inbound metadata: display_phone=%s phone_number_id=%s",
                    display_phone,
                    phone_number_id,
                )

                # ============================================================
                # 1) INBOUND USER MESSAGES
                # ============================================================
                messages = value.get("messages") or []
                contacts = value.get("contacts") or []

                contact_wa_id = None
                if contacts and contacts[0].get("wa_id"):
                    contact_wa_id = contacts[0]["wa_id"]

                for msg in messages:
                    from_wa = msg.get("from") or contact_wa_id
                    user_msg_id = msg.get("id")
                    user_text = extract_user_text(msg)

                    logging.info(
                        "Inbound message: from=%s msg_id=%s text=%s",
                        from_wa,
                        user_msg_id,
                        user_text,
                    )

                    # Skip unsupported payloads (reactions, media without caption, etc.)
                    if not from_wa or not user_text:
                        continue

                    # ---- DEDUPLICATION (Meta retries webhooks) ----
                    if user_msg_id:
                        async with SEEN_LOCK:
                            if user_msg_id in SEEN_MESSAGE_IDS:
                                logging.info("Duplicate message ignored: %s", user_msg_id)
                                return

                            SEEN_MESSAGE_IDS.add(user_msg_id)

                            # prevent unbounded memory growth
                            if len(SEEN_MESSAGE_IDS) > SEEN_MAX:
                                SEEN_MESSAGE_IDS.clear()

                    # ---- PER-USER ORDERING (avoid overlapping replies) ----
                    async with USER_LOCKS[from_wa]:

                        # Log inbound message to DB
                        try:
                            log_message(
                                wa_id=from_wa,
                                direction="user",
                                text=user_text,
                            )
                        except Exception:
                            logging.exception("Failed to log user message to DB")

                        # Log inbound message to Google Sheets
                        try:
                            log_to_sheet(
                                wa_id=from_wa,
                                direction="user",
                                message_id=user_msg_id,
                                text=user_text,
                            )
                        except Exception:
                            logging.exception("Failed to log user message to Sheet")

                        # ---- Generate reply ----
                        reply_text, _meta = generate_reply(
                            user_text=user_text,
                            wa_id=from_wa,
                        )

                        if not phone_number_id:
                            logging.warning(
                                "Missing phone_number_id; using default sender."
                            )

                        # ---- Send reply ----
                        resp = send_whatsapp_text(
                        to_number=from_wa,
                        message=reply_text,
                        phone_number_id=phone_number_id,
                        )

                        # Capture bot message id if available
                        bot_msg_id = None
                        try:
                            if isinstance(resp, dict):
                                bot_msg_id = (resp.get("messages") or [{}])[0].get("id")
                        except Exception:
                            bot_msg_id = None

                        # Log bot reply to DB
                        try:
                            log_message(
                                wa_id=from_wa,
                                direction="bot",
                                text=reply_text,
                            )
                        except Exception:
                            logging.exception("Failed to log bot message to DB")

                        # Log bot reply to Sheets
                        try:
                            log_to_sheet(
                                wa_id=from_wa,
                                direction="bot",
                                message_id=bot_msg_id or user_msg_id,
                                text=reply_text,
                            )
                        except Exception:
                            logging.exception("Failed to log bot message to Sheet")

                # ============================================================
                # 2) STATUS UPDATES (delivered, read, etc.)
                # ============================================================
                statuses = value.get("statuses") or []
                if statuses and not messages:
                    logging.info(
                        "Received %d WhatsApp status events (no inbound messages).",
                        len(statuses),
                    )

    except Exception:
        logging.exception("process_webhook_payload failed")

# -----------------------------
# Webhook receiver (POST) — UPDATED
# -----------------------------
from fastapi import Body

@app.post("/admin/send-template")
async def admin_send_template(payload: dict = Body(...)):
    wa_id = (payload.get("wa_id") or "").strip()
    template_name = (payload.get("name") or "").strip()
    lang = (payload.get("lang") or "es").strip()
    vars_obj = payload.get("vars") or {}

    if not wa_id:
        return JSONResponse({"ok": False, "error": "Missing wa_id"}, status_code=400)
    if not template_name:
        return JSONResponse({"ok": False, "error": "Missing template name"}, status_code=400)

    # Build template components
    components = None
    if vars_obj:
        # If keys are digits ("1","2") treat as positional; otherwise treat as named vars
        keys = list(vars_obj.keys())
        all_digits = all(str(k).isdigit() for k in keys)

        if all_digits:
            # Order positional params 1,2,3...
            ordered_items = sorted(((int(k), vars_obj[k]) for k in keys), key=lambda x: x[0])
            parameters = [{"type": "text", "text": str(v)} for _, v in ordered_items]
        else:
            # Named variables: Meta expects parameter_name
            # Keep stable ordering (sorted by key)
            ordered_items = sorted(((str(k), vars_obj[k]) for k in keys), key=lambda x: x[0])
            parameters = [
                {"type": "text", "parameter_name": k, "text": str(v)}
                for k, v in ordered_items
            ]

        components = [{"type": "body", "parameters": parameters}]

    resp = send_whatsapp_template(
        to_number=wa_id,
        template_name=template_name,
        language_code=lang,
        components=components,
    )

    out_id = None
    try:
        out_id = (resp.get("messages") or [{}])[0].get("id")
    except Exception:
        pass

    log_to_sheet(
        wa_id=wa_id,
        direction="bot",
        message_id=out_id,
        text=f"[TEMPLATE SENT] {template_name}",
    )

    return {"ok": True, "resp": resp, "message_id": out_id}


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp Cloud API webhook.
    ACK immediately, then process asynchronously to avoid timeouts.
    """
    payload = await request.json()
    logging.info("Incoming webhook keys=%s", list(payload.keys()))

    # Extract inbound phone_number_id (which WA number received this message)
    inbound_phone_number_id = None
    try:
        value = (
            payload.get("entry", [{}])[0]
                  .get("changes", [{}])[0]
                  .get("value", {})
        )
        metadata = value.get("metadata", {}) or {}
        inbound_phone_number_id = metadata.get("phone_number_id")
        display_phone = metadata.get("display_phone_number")

        logging.info(
            "Inbound metadata: display_phone_number=%s phone_number_id=%s",
            display_phone, inbound_phone_number_id
        )

        msg = (value.get("messages") or [{}])[0]
        sender = msg.get("from")
        text = (msg.get("text") or {}).get("body")
        logging.info("Inbound msg: from=%s text=%s", sender, text)

    except Exception:
        logging.exception("Failed to parse/log webhook metadata")

    # Fire-and-forget async processing
    asyncio.create_task(process_webhook_payload(payload, inbound_phone_number_id))

    # ACK fast
    return JSONResponse({"status": "ok"}, status_code=200)

# -----------------------------
# Manual test endpoint
# -----------------------------
@app.post("/test")
async def test_api(req: Request):
    data = await req.json()
    q = (data.get("q") or "").strip()
    if not q:
        return JSONResponse({"error": "Missing q"}, status_code=400)
    reply_text, meta = generate_reply(q)
    return {"reply": reply_text, "meta": meta}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)




# Sending template message manually
# curl -X POST "https://consultoria-3.onrender.com/admin/send-template" \
#  -H "Content-Type: application/json" \
#  -d '{"wa_id":"573206720800","name":"bienvenida_insights","lang":"es"}'

# Steps to run:
# 1. open terminal
# 2. Input into terminal: cd /Users/manuelalozano/Documents/INSIGHTS/proyecto\ ia
# 3. Input into terminal: source .venv/bin/activate
# 4. Input into terminal: python test_environment.py

# To exit environment when finished: enter into terminal: deactivate