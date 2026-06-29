# PDF Intelligence

Ask questions about your PDFs and get answers grounded in the actual text, with citations back to the page they came from. Upload one or more PDFs and the app extracts the text, splits and embeds it, stores the vectors locally in ChromaDB, and answers your questions using only what it can retrieve — so it says it couldn't find something instead of making things up.

The UI is a React + TypeScript single-page app talking to a FastAPI backend. A standalone Streamlit version (`app.py`) is also kept around for a quick single-process demo.

## How it works

1. Extract text from each PDF page with `pypdf`, keeping the file name and page number.
2. Split the page text into overlapping chunks.
3. Embed the chunks and store them in a local ChromaDB collection.
4. For a question, embed it, pull the nearest chunks, and drop anything past a relevance cutoff.
5. Send the surviving chunks to the chat model, which answers only from that context and cites sources inline as `[1]`, `[2]`, …
6. Stream the answer back token by token; cited sources appear next to it.

## Stack

- React, TypeScript, Vite (frontend)
- FastAPI, Python 3.11+ (backend)
- ChromaDB for local vector storage
- Any OpenAI-compatible API for chat + embeddings (defaults assume OpenRouter)
- pypdf for text extraction

## Running it

Backend:

```bash
python -m venv .venv
. .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your API key
uvicorn api.main:app --reload --port 8000
```

Frontend (in a second terminal):

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

The frontend calls the API at `http://localhost:8000` by default; override with `VITE_API_BASE`.

### Streamlit version

```bash
streamlit run app.py
```

## Configuration

Everything is read from `.env` — see `.env.example` for the full list and defaults. The ones you'll usually touch:

| Variable | What it does |
| --- | --- |
| `OPENAI_API_KEY` | API key for your model provider |
| `OPENAI_BASE_URL` | Provider endpoint (e.g. OpenRouter) |
| `OPENAI_CHAT_MODEL` | Model that writes the answers |
| `OPENAI_EMBEDDING_MODEL` | Model that embeds chunks and questions |
| `RETRIEVAL_MAX_DISTANCE` | How strict relevance filtering is (lower = stricter) |
| `MAX_UPLOAD_MB`, `MAX_PAGES_PER_PDF` | Upload size/length guards |

## Tests

```bash
python -m pytest          # backend
cd frontend && npm test   # frontend
```

## Notes

- ChromaDB persists to a local directory, which works fine for development but may reset on some free hosting platforms — use durable storage for anything real.
- Scanned PDFs with no embedded text won't produce useful chunks; there's no OCR step.
