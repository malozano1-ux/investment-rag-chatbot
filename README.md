# investment-rag-chatbot

A production WhatsApp chatbot for financial education, built with **Retrieval-Augmented Generation (RAG)**, **FAISS vector search**, and the **Gemini API**. Deployed via FastAPI on Render.

---

## Architecture

```
User (WhatsApp)
      │
      ▼
Meta Cloud API (webhook)
      │
      ▼
FastAPI App (main.py)
      │
      ├── Intent Router → classifies message topic
      │         (funding / investing / academy / premium / onboarding / general)
      │
      ├── FAISS Retriever → fetches top-k relevant chunks from topic-specific index
      │         (per-topic indexes prevent cross-topic noise)
      │
      ├── Gemini API → generates grounded response using retrieved context + chat history
      │
      ├── SQLite → logs all messages for conversation history
      │
      └── Google Sheets → real-time logging for ops monitoring
```

**Key design decisions:**
- **Per-topic FAISS indexes**: Each topic (funding, products, academy, etc.) has its own index, so retrieval is scoped to relevant documents only.
- **RAG-first policy**: The LLM is instructed to prioritize retrieved internal docs over general knowledge.
- **Async webhook processing**: FastAPI ACKs Meta's webhook immediately and processes asynchronously to avoid 20s timeout errors.
- **Deduplication**: In-memory set tracks processed message IDs to handle Meta's retry behavior.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM | Google Gemini (gemini-2.0-flash) |
| Embeddings | Gemini Embedding API |
| Vector Search | FAISS (IndexFlatIP + L2 normalization → cosine similarity) |
| Messaging | WhatsApp Cloud API (Meta) |
| Conversation Logging | SQLAlchemy + SQLite |
| Ops Logging | Google Sheets API (gspread) |
| Deployment | Render |

---

## Project Structure

```
investment-rag-chatbot/
├── main.py                     # FastAPI app: webhook handler, RAG pipeline, reply generation
├── rag_retriever.py            # Standalone FAISS retriever (OpenAI embeddings variant)
├── send_campaign.py            # Bulk WhatsApp template sender (CSV → batch API calls)
├── send_single.py              # Single-recipient template sender (for testing)
├── send_onboarding_campaign.py # Onboarding-specific bulk sender
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── docs/                       # Place your PDF knowledge base files here
    ├── funding_faq.pdf
    ├── products_overview.pdf
    └── ...
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/malozano1-ux/investment-rag-chatbot.git
cd investment-rag-chatbot
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual keys
```

### 3. Add your knowledge base

Place PDF files in the `docs/` folder and update the `TRANSCRIPT_SOURCES` list in `main.py` to match your file names and topics.

### 4. Run locally

```bash
uvicorn main:app --reload --port 8000
```

### 5. Expose locally for webhook testing (optional)

```bash
ngrok http 8000
# Use the ngrok URL as your Meta webhook URL
```

---

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `WHATSAPP_TOKEN` | Meta WhatsApp Cloud API bearer token |
| `WHATSAPP_PHONE_ID` | WhatsApp sender phone number ID |
| `VERIFY_TOKEN` | Webhook verification token (you choose this) |
| `GSHEET_ID` | Google Sheet ID for conversation logging |
| `GOOGLE_CREDS_JSON_B64` | Base64-encoded Google service account JSON |
| `DATABASE_URL` | SQLAlchemy DB URL (defaults to SQLite) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/webhook` | Meta webhook verification |
| `POST` | `/webhook` | Receive inbound WhatsApp messages |
| `POST` | `/test` | Test the bot locally (`{"q": "your question"}`) |
| `POST` | `/admin/send-template` | Send a WhatsApp template message |
| `GET` | `/debug/rag-stats` | Check RAG index status |
| `GET` | `/debug/sheets-test` | Verify Google Sheets connection |

---

## Campaign Scripts

### Bulk campaign
```bash
CAMPAIGN_CSV=contacts.csv TPL_NAME=your_template python send_campaign.py
```

### Single test message
```bash
WHATSAPP_RECIPIENT=15551234567 TEMPLATE_NAME=your_template python send_single.py
```

### Onboarding re-engagement
```bash
ONBOARDING_CSV=pending_users.csv TPL_ONBOARDING=onboarding_reminder python send_onboarding_campaign.py
```
