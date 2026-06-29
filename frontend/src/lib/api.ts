/*
 * Typed API client for the FastAPI backend.
 *
 * NOTE: The backend is not built yet. These functions describe the planned
 * contract and are not called at runtime in this milestone (B3). They exist so
 * later milestones can wire up chat and indexing without re-deriving the shapes.
 */
import type {
  AskResponse,
  HealthResponse,
  IndexResponse,
} from "./types";

const API_BASE: string = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/** Error thrown when the backend responds with a non-2xx status. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new ApiError(
      response.status,
      text || `Request failed with status ${response.status}`,
    );
  }
  return (await response.json()) as T;
}

/** POST /ask — submit a question, receive an answer with grounding sources. */
export async function ask(question: string): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return parseJson<AskResponse>(response);
}

/** POST /index — upload one or more PDFs to be indexed (multipart). */
export async function indexPdfs(files: File[]): Promise<IndexResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file, file.name);
  }
  const response = await fetch(`${API_BASE}/index`, {
    method: "POST",
    body: formData,
  });
  return parseJson<IndexResponse>(response);
}

/** GET /health — lightweight backend liveness check. */
export async function health(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { method: "GET" });
  return parseJson<HealthResponse>(response);
}
