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
import type { ConfigResponse, IndexFileResult } from "../../lib/types";
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

export interface DrainCallbacks {
  /** Upload one file and resolve with the server's per-file result. */
  upload: (file: File) => Promise<IndexFileResult | undefined>;
  /** Apply a status patch to the row with the given id. */
  patch: (id: string, patch: Partial<UploadFile>) => void;
}

/**
 * Drain a queue of rows, uploading each row's file in order and mapping the
 * result back onto the row. Mutates `queue` (via shift) so rows pushed mid-drain
 * are still handled. Each file is isolated: a throw is recorded on its row and
 * draining continues. Returns whether any file was processed.
 *
 * Pure (no React) so the core upload loop is unit-tested directly — the earlier
 * regression (rows stuck on "indexing", never uploaded) lived exactly here.
 */
export async function drainUploadQueue(
  queue: UploadFile[],
  callbacks: DrainCallbacks,
): Promise<boolean> {
  let processedAny = false;
  while (queue.length > 0) {
    const row = queue.shift();
    if (!row || !row.file) continue;
    processedAny = true;
    callbacks.patch(row.id, { status: "indexing" });
    try {
      const result = await callbacks.upload(row.file);
      if (result) {
        const status: FileStatus =
          result.status === "indexed"
            ? "indexed"
            : result.status === "rejected"
              ? "rejected"
              : result.status === "unsupported"
                ? "unsupported"
                : "failed";
        callbacks.patch(row.id, {
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
        callbacks.patch(row.id, {
          status: "failed",
          detail: "No result returned for this file.",
        });
      }
    } catch (err: unknown) {
      const detail =
        err instanceof ApiError
          ? `Upload failed (${err.status}).`
          : err instanceof Error
            ? err.message
            : "Upload failed.";
      callbacks.patch(row.id, { status: "failed", detail });
    }
  }
  return processedAny;
}

export function useUpload(options: UseUploadOptions = {}): UseUploadResult {
  const { onBatchComplete } = options;
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [skippedNote, setSkippedNote] = useState<SkippedNote | null>(null);

  // Rows awaiting upload, drained one at a time. Held in a ref (not state) so the
  // drainer reads the queue synchronously — reading it back through a setState
  // updater does not work, because React runs the updater asynchronously.
  const queueRef = useRef<UploadFile[]>([]);
  // A drain may already be running; guard against starting a second.
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
   * Drain the upload queue (a ref, read synchronously) one file at a time, then
   * refresh the live store once if anything was processed. A guarded singleton:
   * rows queued mid-drain are picked up by the same loop.
   */
  const drain = useCallback(async () => {
    if (pumpingRef.current) return;
    pumpingRef.current = true;
    setUploading(true);
    const processedAny = await drainUploadQueue(queueRef.current, {
      upload: async (file) => (await indexPdfs([file])).results[0],
      patch: patchFile,
    });
    pumpingRef.current = false;
    setUploading(false);
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

      // 4. Queue the uploadable rows (with their File objects) and kick the
      // drainer. The queue lives in a ref so the drainer reads it reliably.
      const queued = rows.filter((r) => r.status === "queued" && r.file);
      if (queued.length > 0) {
        queueRef.current.push(...queued);
        void drain();
      }
    },
    [config, drain],
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
