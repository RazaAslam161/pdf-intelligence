/*
 * Sequential upload state machine for the document panel.
 *
 * Responsibilities:
 *   - fetch + cache GET /config once on mount (drives pre-flight)
 *   - accept dropped / browsed files, run pure pre-flight (preflight.ts), and
 *     turn each into a row with an initial status
 *   - upload eligible rows ONE AT A TIME (indexPdfs([file])) so progress is
 *     per-file and one file's failure never blocks the others
 *   - merge each server result back onto its row
 *
 * The component layer (UploadPanel) is purely presentational over this state.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getConfig, indexPdfs } from "../../lib/api";
import type { ConfigResponse } from "../../lib/types";
import {
  applyFileCountCap,
  validateFiles,
  type PreflightStatus,
} from "./preflight";

/**
 * The lifecycle status of a single queued file. Pre-flight rejections
 * ("too_large" / "unsupported") and the queued/indexing/terminal server states
 * all live in one union so a row is always exactly one status.
 */
export type FileStatus =
  | "queued" // passed pre-flight, awaiting its turn to upload
  | "indexing" // /index call in flight for this file
  | "indexed" // server: success
  | "rejected" // server: P0-2 limit hit (e.g. too many pages/chunks)
  | "failed" // server: other error, or a network/transport error
  | "too_large" // pre-flight: over the size cap (never uploaded)
  | "unsupported"; // pre-flight: not a .pdf (never uploaded)

export interface UploadFile {
  /** Stable per-session id (independent of name; duplicates are allowed). */
  id: string;
  name: string;
  size: number;
  status: FileStatus;
  /** The underlying File, present only for rows we may upload. */
  file?: File;
  /** Populated once the server returns a result for this row. */
  pageCount?: number;
  chunkCount?: number;
  /** Human-readable detail for failed/rejected/error rows. */
  detail?: string;
}

/** A "skipped due to per-run limit" note surfaced once per add. */
export interface SkippedNote {
  id: string;
  count: number;
  cap: number;
}

let fileCounter = 0;
function nextFileId(): string {
  fileCounter += 1;
  return `file-${fileCounter}`;
}

/** Map a pre-flight verdict onto the row's initial status. */
function statusFromPreflight(status: PreflightStatus): FileStatus {
  if (status === "too_large") return "too_large";
  if (status === "unsupported") return "unsupported";
  return "queued";
}

export interface UseUploadResult {
  config: ConfigResponse | null;
  configError: string | null;
  files: UploadFile[];
  /** True while any row is uploading. */
  uploading: boolean;
  /** Most recent "per-run limit" skip note, or null. */
  skippedNote: SkippedNote | null;
  /** Add dropped / browsed files: pre-flight, cap, queue, then auto-upload. */
  addFiles: (files: File[]) => void;
  /** Clear all rows (only when nothing is uploading). */
  clear: () => void;
}

export interface UseUploadOptions {
  /**
   * Called once after the queue finishes draining and at least one file was
   * uploaded in that drain. Lets the caller refresh the live document store so
   * it reflects the just-indexed files (the store, not this session, is the
   * source of truth for what's indexed).
   */
  onBatchComplete?: () => void;
}

export function useUpload(options: UseUploadOptions = {}): UseUploadResult {
  const { onBatchComplete } = options;
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [skippedNote, setSkippedNote] = useState<SkippedNote | null>(null);

  // A pump may already be draining the queue; guard against starting a second.
  const pumpingRef = useRef(false);

  // Keep the latest callback without making `pump` depend on it (which would
  // re-create the pump and risk reentrancy churn).
  const onBatchCompleteRef = useRef(onBatchComplete);
  onBatchCompleteRef.current = onBatchComplete;

  // Fetch + cache config once on mount.
  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then((cfg) => {
        if (!cancelled) setConfig(cfg);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message =
          err instanceof ApiError
            ? `Could not load upload limits (${err.status}).`
            : "Could not load upload limits.";
        setConfigError(message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const patchFile = useCallback(
    (id: string, patch: Partial<UploadFile>) => {
      setFiles((prev) =>
        prev.map((f) => (f.id === id ? { ...f, ...patch } : f)),
      );
    },
    [],
  );

  /**
   * Drain the queue one file at a time. Re-reads the latest state via the
   * functional setState so newly added rows are picked up. Each file is fully
   * isolated: a throw is caught and recorded on that row, then we continue.
   */
  const pump = useCallback(async () => {
    if (pumpingRef.current) return;
    pumpingRef.current = true;
    setUploading(true);
    // Did this drain attempt to index any file? Drives the post-batch refresh.
    let processedAny = false;

    // Pull the next queued row off the current state snapshot.
    const takeNext = (): UploadFile | null => {
      let found: UploadFile | null = null;
      setFiles((prev) => {
        const next = prev.find((f) => f.status === "queued");
        if (next) {
          found = next;
          return prev.map((f) =>
            f.id === next.id ? { ...f, status: "indexing" as const } : f,
          );
        }
        return prev;
      });
      return found;
    };

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const current = takeNext();
      if (!current || !current.file) break;
      processedAny = true;

      try {
        const res = await indexPdfs([current.file]);
        const result = res.results[0];
        if (result) {
          const status: FileStatus =
            result.status === "indexed"
              ? "indexed"
              : result.status === "rejected"
                ? "rejected"
                : result.status === "unsupported"
                  ? "unsupported"
                  : "failed";
          patchFile(current.id, {
            status,
            pageCount: result.page_count,
            chunkCount: result.chunk_count,
            detail:
              status === "rejected"
                ? "Rejected by the server limits."
                : status === "failed"
                  ? "The server could not index this file."
                  : undefined,
          });
        } else {
          patchFile(current.id, {
            status: "failed",
            detail: "No result returned for this file.",
          });
        }
      } catch (err: unknown) {
        // Isolated failure: record it and keep draining the rest.
        const detail =
          err instanceof ApiError
            ? `Upload failed (${err.status}).`
            : err instanceof Error
              ? err.message
              : "Upload failed.";
        patchFile(current.id, { status: "failed", detail });
      }
    }

    pumpingRef.current = false;
    setUploading(false);

    // The batch is done draining. If we touched at least one file, let the
    // caller refresh the live store (it is the source of truth, not the
    // per-file rows). Failures don't block the refresh — the store reflects
    // whatever the server actually persisted.
    if (processedAny) {
      onBatchCompleteRef.current?.();
    }
  }, [patchFile]);

  const addFiles = useCallback(
    (incoming: File[]) => {
      if (incoming.length === 0) return;
      if (!config) return; // pre-flight needs the cached config

      // 1. Per-run count cap, applied to the incoming batch.
      const { accepted, skipped } = applyFileCountCap(incoming, config);
      setSkippedNote(
        skipped.length > 0
          ? {
              id: nextFileId(),
              count: skipped.length,
              cap: config.max_files_per_run,
            }
          : null,
      );

      // 2. Pure pre-flight over the accepted files.
      const verdicts = validateFiles(
        accepted.map((f) => ({ name: f.name, size: f.size })),
        config,
      );

      // 3. Build rows; only "ok" rows carry the File for upload.
      const rows: UploadFile[] = accepted.map((file, i) => {
        const verdict = verdicts[i];
        const status = statusFromPreflight(verdict.status);
        return {
          id: nextFileId(),
          name: file.name,
          size: file.size,
          status,
          file: status === "queued" ? file : undefined,
        };
      });

      setFiles((prev) => [...prev, ...rows]);

      // 4. Kick the pump if anything is uploadable.
      if (rows.some((r) => r.status === "queued")) {
        void pump();
      }
    },
    [config, pump],
  );

  const clear = useCallback(() => {
    if (pumpingRef.current) return;
    setFiles([]);
    setSkippedNote(null);
  }, []);

  return {
    config,
    configError,
    files,
    uploading,
    skippedNote,
    addFiles,
    clear,
  };
}
