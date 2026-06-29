# AI PDF Question Answering Chatbot

A Streamlit web app for asking questions about uploaded PDF documents using Retrieval-Augmented Generation (RAG). The app extracts page-level text, creates searchable embeddings, retrieves relevant chunks, and generates grounded answers with PDF name and page-number citations.

## Features

- Upload and index one or more PDF files.
- Extract text page by page with source metadata.
- Split PDF text into overlapping chunks for retrieval.
- Generate embeddings with the OpenAI Python SDK.
- Store and search vectors locally with ChromaDB.
- Ask questions in a chat-style interface.
- Display answers with expandable source citations.
- Deduplicate citations from the same PDF page.
- Clear the local vector database and chat history from the UI.
- Run tests without calling the real OpenAI API.

## Architecture Overview

```text
PDF upload
  -> PDF loader
  -> PageText records
  -> Chunker
  -> TextChunk records
  -> OpenAI embeddings
  -> ChromaDB vector store
  -> Retriever
  -> Grounded prompt
  -> OpenAI answer generation
  -> Answer + citations
```

The app is intentionally modular so the Streamlit UI stays separate from PDF parsing, chunking, embeddings, vector storage, and RAG orchestration.

## Tech Stack

- Python 3.11+
- Streamlit
- OpenAI Python SDK
- ChromaDB
- pypdf
- python-dotenv
- pytest
- Ruff

## Folder Structure

```text
.
|-- app.py
|-- src/
|   |-- config.py
|   |-- pdf_loader.py
|   |-- chunker.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- rag_service.py
|   |-- citations.py
|   `-- models.py
|-- tests/
|   |-- test_app.py
|   |-- test_chunker.py
|   |-- test_citations.py
|   |-- test_config.py
|   |-- test_embeddings.py
|   |-- test_pdf_loader.py
|   |-- test_rag_service.py
|   `-- test_vector_store.py
|-- requirements.txt
|-- pyproject.toml
|-- pytest.ini
|-- Makefile
|-- Dockerfile
|-- .dockerignore
|-- .env.example
|-- .gitignore
`-- README.md
```

## How RAG Works in This Project

1. The user uploads PDF files in the Streamlit app.
2. `src/pdf_loader.py` extracts text from each readable page and preserves the file name, document ID, and page number.
3. `src/chunker.py` splits page text into overlapping chunks while keeping source metadata attached.
4. `src/embeddings.py` sends chunk text to the configured OpenAI embedding model.
5. `src/vector_store.py` stores chunks, metadata, and embeddings in a local ChromaDB collection.
6. When the user asks a question, the app embeds the question and retrieves the most relevant chunks.
7. `src/rag_service.py` builds a grounded prompt that tells the model to answer only from retrieved PDF context.
8. The final response includes the answer and citations with PDF file name, page number, and a short preview.

## Local Setup

Clone the repository and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Then set your real environment values in `.env`. Do not commit `.env`.

On macOS/Linux, you can also use:

```bash
make install
```

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Authenticates embedding and answer generation requests. |
| `OPENAI_CHAT_MODEL` | No | Chat model used for answer generation. Defaults to `gpt-4.1-mini`. |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model used for document chunks and questions. Defaults to `text-embedding-3-small`. |
| `CHROMA_PERSIST_DIR` | No | Local directory for ChromaDB persistence. Defaults to `./chroma_db`. |
| `CHUNK_SIZE` | No | Maximum chunk size used during text splitting. Defaults to `1000`. |
| `CHUNK_OVERLAP` | No | Character overlap between adjacent chunks. Defaults to `150`. |
| `RETRIEVAL_TOP_K` | No | Number of chunks retrieved for each question. Defaults to `4`. |

Use `.env.example` as the reference for variable names and defaults. Keep secrets in `.env` locally or in your hosting provider's secret manager.

## Run the App

```bash
streamlit run app.py
```

Or with Make:

```bash
make run
```

The Streamlit entry point is `app.py`.

## Run Tests

```bash
python -m pytest
```

Or with Make:

```bash
make test
```

Run linting and tests together:

```bash
make check
```

Tests use fakes and local fixtures for embeddings, answer generation, PDFs, and vector storage. They do not call the real OpenAI API.

## Deployment

For Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit app from the repository.
3. Set `app.py` as the app entry point.
4. Add required secrets in the Streamlit Cloud settings.
5. Deploy using `requirements.txt`.

For Docker-based hosting:

```bash
docker build -t rag-pdf-chatbot .
docker run --rm -p 8501:8501 --env-file .env rag-pdf-chatbot
```

Local ChromaDB persistence works well for development and demos. On simple hosted platforms, local files may reset during redeploys, restarts, or container rebuilds. For a production deployment, use durable external storage or a managed vector database.

## Screenshots

Add screenshots here after the UI is finalized.

```text
docs/screenshots/upload-and-index.png
docs/screenshots/question-answering.png
```

## Future Improvements

- Add support for OCR-based extraction from scanned PDFs.
- Add authentication for multi-user deployments.
- Store uploaded documents and vectors in durable cloud storage.
- Add streaming answer generation.
- Add richer citation links that jump to source pages.
- Add evaluation tests for retrieval quality.
- Add configurable document collection management.

## Important Limitations

- Scanned PDFs without embedded text may produce little or no extractable content.
- Local ChromaDB persistence is not durable on many free hosting platforms.
- Answers are only as reliable as the extracted text and retrieved context.
- The app is designed for portfolio and learning use, not production compliance workloads.
- Large PDFs may require additional chunking, batching, timeout, and cost controls.

## License

License information has not been added yet. Add a license file before publishing or reusing this project publicly.
