// Types mirroring the FastAPI contract. Keep in sync with the backend.

/** A retrieved chunk of a source PDF that grounds an answer. */
export interface Source {
  id: string;
  file_name: string;
  page_number: number;
  preview: string;
  /** Relevance score; may be null when the backend does not return one. */
  score: number | null;
}

/** Response shape for POST /ask. */
export interface AskResponse {
  answer: string;
  sources: Source[];
}

/** Per-file outcome of an indexing request. */
export interface IndexFileResult {
  file_name: string;
  page_count: number;
  chunk_count: number;
  status: string;
}

/** Response shape for POST /index. */
export interface IndexResponse {
  results: IndexFileResult[];
}

/** A document in the persisted vector store (across sessions, not just this upload). */
export interface DocumentSummary {
  document_id: string;
  file_name: string;
  page_count: number;
  chunk_count: number;
}

/** Snapshot of the vector store from GET /documents (and DELETE /documents). `ready` is true when total_chunks > 0. */
export interface StoreState {
  documents: DocumentSummary[];
  total_chunks: number;
  ready: boolean;
}

/** Response shape for GET /health. */
export interface HealthResponse {
  status: string;
}

/** Response shape for GET /config — upload limits used for client-side pre-flight checks. */
export interface ConfigResponse {
  /** Maximum size of a single upload, in megabytes. */
  max_upload_mb: number;
  /** Maximum page count the server will index per PDF. */
  max_pages_per_pdf: number;
  /** Maximum number of files accepted in a single run. */
  max_files_per_run: number;
}
