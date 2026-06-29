/*
 * Types mirroring the planned FastAPI contract. The backend is not built yet,
 * so these are forward-looking stubs — keep them in sync with the API once it
 * exists.
 */

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

/**
 * One document currently held in the persisted vector store. Reflects the LIVE
 * store (across sessions), not the current upload session.
 */
export interface DocumentSummary {
  document_id: string;
  file_name: string;
  page_count: number;
  chunk_count: number;
}

/**
 * Snapshot of the persisted vector store. Returned by GET /documents and by
 * DELETE /documents (the post-clear state). `ready` is true iff total_chunks > 0.
 */
export interface StoreState {
  documents: DocumentSummary[];
  total_chunks: number;
  ready: boolean;
}

/** Response shape for GET /health. */
export interface HealthResponse {
  status: string;
}

/**
 * Response shape for GET /config — server-advertised upload limits. Drives the
 * client-side pre-flight checks before any file is sent for indexing.
 */
export interface ConfigResponse {
  /** Maximum size of a single upload, in megabytes. */
  max_upload_mb: number;
  /** Maximum page count the server will index per PDF. */
  max_pages_per_pdf: number;
  /** Maximum number of files accepted in a single run. */
  max_files_per_run: number;
}
