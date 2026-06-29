// Typed API client for the FastAPI backend. /ask streams answers as SSE; the
// rest return plain JSON.
import type {
  ConfigResponse,
  HealthResponse,
  IndexResponse,
  Source,
  StoreState,
} from "./types";
import { parseSseChunk } from "./sse";

const API_BASE: string = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

// Optional tenant id for the X-Tenant-Id header on /ask. Env-driven so one
// build can target multiple tenants; omitted when unset.
const TENANT_ID: string | undefined = import.meta.env.VITE_TENANT_ID;

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

/** Callbacks invoked as the /ask SSE stream is consumed. */
export interface AskStreamHandlers {
  /** A token delta to append to the in-progress answer (called repeatedly). */
  onToken: (text: string) => void;
  /** The grounding sources, delivered once near the end of the stream. */
  onSources: (sources: Source[]) => void;
  /** A generation error from the server, or a transport failure. */
  onError: (message: string) => void;
  /** Abort the request (e.g. on unmount) by aborting this signal. */
  signal?: AbortSignal;
}

// POST /ask — submit a question and consume the streamed answer. Resolves once
// the stream ends, whether cleanly, via a server error event, or on a transport
// failure (all reported through onError). AbortError is swallowed.
export async function askStream(
  question: string,
  handlers: AskStreamHandlers,
): Promise<void> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  if (TENANT_ID) headers["X-Tenant-Id"] = TENANT_ID;

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers,
      body: JSON.stringify({ question }),
      signal: handlers.signal,
    });
  } catch (err) {
    if (isAbort(err)) return;
    handlers.onError(
      err instanceof Error ? err.message : "Unable to reach the backend.",
    );
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    handlers.onError(
      `Request failed (${response.status}). ${text || "The server returned an error."}`,
    );
    return;
  }

  if (!response.body) {
    handlers.onError("The server returned an empty response.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (value) {
        buffer += decoder.decode(value, { stream: true });
        const { events, rest } = parseSseChunk(buffer);
        buffer = rest;
        for (const event of events) {
          switch (event.type) {
            case "token":
              handlers.onToken(event.text);
              break;
            case "sources":
              handlers.onSources(event.sources);
              break;
            case "error":
              handlers.onError(event.message);
              return;
            case "done":
              return;
          }
        }
      }
      if (done) {
        // Flush any complete record left in the buffer at stream end.
        const tail = buffer + decoder.decode();
        const { events } = parseSseChunk(tail.endsWith("\n\n") ? tail : `${tail}\n\n`);
        for (const event of events) {
          if (event.type === "token") handlers.onToken(event.text);
          else if (event.type === "sources") handlers.onSources(event.sources);
          else if (event.type === "error") {
            handlers.onError(event.message);
            return;
          }
        }
        return;
      }
    }
  } catch (err) {
    if (isAbort(err)) return;
    handlers.onError(
      err instanceof Error ? err.message : "The answer stream was interrupted.",
    );
  }
}

/** True when an error is the DOMException raised by aborting a fetch. */
function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
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

/** GET /config — server-advertised upload limits (size, pages, file count). */
export async function getConfig(): Promise<ConfigResponse> {
  const response = await fetch(`${API_BASE}/config`, { method: "GET" });
  return parseJson<ConfigResponse>(response);
}

/** GET /documents — the live persisted vector store (across sessions). */
export async function getDocuments(): Promise<StoreState> {
  const response = await fetch(`${API_BASE}/documents`, { method: "GET" });
  return parseJson<StoreState>(response);
}

/** DELETE /documents — wipe the store; resolves to the post-clear state. */
export async function clearDocuments(): Promise<StoreState> {
  const response = await fetch(`${API_BASE}/documents`, { method: "DELETE" });
  return parseJson<StoreState>(response);
}
