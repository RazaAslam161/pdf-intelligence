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

/** Response shape for GET /health. */
export interface HealthResponse {
  status: string;
}
