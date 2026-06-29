// State for the persisted document/vector store: load on mount, refresh after
// uploads, and clear (which applies the DELETE response directly).
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, clearDocuments, getDocuments } from "../../lib/api";
import type { StoreState } from "../../lib/types";

/** Loading lifecycle for the store snapshot. */
export type StorePhase = "loading" | "ready" | "error";

export interface UseStoreResult {
  /** The latest store snapshot, or null until the first load resolves. */
  state: StoreState | null;
  phase: StorePhase;
  /** Human-readable load error (e.g. no backend in dev), or null. */
  error: string | null;
  /** True while a DELETE /documents call is in flight. */
  clearing: boolean;
  /** Re-fetch GET /documents (used on mount and after an upload batch). */
  refresh: () => Promise<void>;
  /** Wipe the store; updates state immediately from the DELETE response. */
  clear: () => Promise<void>;
}

function loadMessage(err: unknown): string {
  if (err instanceof ApiError) {
    return `Could not load the document store (${err.status}).`;
  }
  return "Could not load the document store.";
}

export function useStore(): UseStoreResult {
  const [state, setState] = useState<StoreState | null>(null);
  const [phase, setPhase] = useState<StorePhase>("loading");
  const [error, setError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  // Guard against setState after unmount and against overlapping clears.
  const mountedRef = useRef(true);
  const clearingRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    setPhase("loading");
    setError(null);
    try {
      const next = await getDocuments();
      if (!mountedRef.current) return;
      setState(next);
      setPhase("ready");
    } catch (err: unknown) {
      if (!mountedRef.current) return;
      setError(loadMessage(err));
      setPhase("error");
    }
  }, []);

  // Fetch once on mount.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const clear = useCallback(async () => {
    if (clearingRef.current) return;
    clearingRef.current = true;
    setClearing(true);
    try {
      // The DELETE response is the post-clear state, so apply it directly.
      const next = await clearDocuments();
      if (!mountedRef.current) return;
      setState(next);
      setPhase("ready");
      setError(null);
    } catch (err: unknown) {
      if (!mountedRef.current) return;
      setError(loadMessage(err));
      setPhase("error");
    } finally {
      clearingRef.current = false;
      if (mountedRef.current) setClearing(false);
    }
  }, []);

  return { state, phase, error, clearing, refresh, clear };
}
