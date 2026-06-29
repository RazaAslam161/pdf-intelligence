/*
 * Pure formatting helpers for the document store UI. No side effects, no network.
 */
import type { StoreState } from "../../lib/types";

/** pluralize("document", 1) -> "1 document". */
export function pluralize(noun: string, count: number): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/**
 * One-line readiness summary. Readiness keys off chunk count (matching the
 * backend's `ready` flag), so documents with zero chunks still read as empty.
 */
export function readinessSummary(state: StoreState): string {
  if (!state.ready || state.total_chunks <= 0) {
    return "No documents indexed yet — add a PDF to start asking questions.";
  }
  const docs = pluralize("document", state.documents.length);
  const chunks = pluralize("chunk", state.total_chunks);
  return `${docs} · ${chunks} ready to query.`;
}

/** Short readiness label for the header. */
export function readinessLabel(state: StoreState): string {
  if (!state.ready || state.total_chunks <= 0) {
    return "No documents";
  }
  return `${pluralize("document", state.documents.length)} indexed`;
}

/** Per-document line, e.g. "12 pages · 48 chunks". */
export function documentMeta(pageCount: number, chunkCount: number): string {
  return `${pluralize("page", pageCount)} · ${pluralize("chunk", chunkCount)}`;
}
