/*
 * Pure formatting/summary helpers for the workspace (document store) UI.
 *
 * Side-effect-free and fully unit-tested (see summary.test.ts) so the
 * presentational components can stay thin. None of this touches the network.
 */
import type { StoreState } from "../../lib/types";

/** Simple count-aware pluralization: pluralize("document", 1) -> "1 document". */
export function pluralize(noun: string, count: number): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/**
 * A concise one-line readiness summary for a store snapshot.
 *
 *   - empty (no chunks)  -> the calm empty-state sentence
 *   - ready (>0 chunks)  -> "N documents · M chunks ready to query."
 *
 * Readiness is driven by chunk count (mirrors the backend's `ready` flag), not
 * the document count, so a store with documents but zero chunks still reads as
 * empty.
 */
export function readinessSummary(state: StoreState): string {
  if (!state.ready || state.total_chunks <= 0) {
    return "No documents indexed yet — add a PDF to start asking questions.";
  }
  const docs = pluralize("document", state.documents.length);
  const chunks = pluralize("chunk", state.total_chunks);
  return `${docs} · ${chunks} ready to query.`;
}

/**
 * A very short readiness label suited to a quiet header status. Empty stores
 * read "No documents"; ready stores read "N documents indexed".
 */
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
