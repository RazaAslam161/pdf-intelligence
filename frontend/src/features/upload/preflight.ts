/*
 * Pure, client-side pre-flight checks for the upload feature.
 *
 * These run BEFORE any /index call so we never ship a file the server is
 * guaranteed to reject. The logic is deliberately side-effect-free and fully
 * unit-tested (see preflight.test.ts).
 *
 * Two server-driven limits are enforced here:
 *   - per-file size cap (max_upload_mb)
 *   - the .pdf extension requirement
 * and one queue-shaping rule:
 *   - the per-run file-count cap (max_files_per_run)
 */
import type { ConfigResponse } from "../../lib/types";

/** Minimal descriptor of a file for categorization — keeps the logic testable
 *  without needing a real File/Blob. */
export interface FileDescriptor {
  name: string;
  size: number;
}

/**
 * Pre-flight verdict for a single file, decided locally:
 *   - "ok"          passes all client checks; eligible to upload
 *   - "too_large"   size exceeds max_upload_mb
 *   - "unsupported" name does not end in .pdf
 */
export type PreflightStatus = "ok" | "too_large" | "unsupported";

export interface PreflightResult extends FileDescriptor {
  status: PreflightStatus;
}

/** Bytes in one megabyte (binary — matches the size shown to the user). */
export const BYTES_PER_MB = 1024 * 1024;

/** True when a file name ends with the .pdf extension (case-insensitive). */
export function isPdfName(name: string): boolean {
  return /\.pdf$/i.test(name.trim());
}

/**
 * Categorize each file against the server config WITHOUT uploading anything.
 *
 * Order of checks (first failure wins):
 *   1. extension — must end in .pdf, else "unsupported"
 *   2. size      — must be <= max_upload_mb * 1MB, else "too_large"
 * Otherwise "ok". A boundary file (exactly at the cap) is accepted; a 0-byte
 * .pdf is "ok" (the server decides whether it is truly empty).
 */
export function validateFiles(
  files: FileDescriptor[],
  config: ConfigResponse,
): PreflightResult[] {
  const maxBytes = config.max_upload_mb * BYTES_PER_MB;

  return files.map((file) => {
    let status: PreflightStatus;
    if (!isPdfName(file.name)) {
      status = "unsupported";
    } else if (file.size > maxBytes) {
      status = "too_large";
    } else {
      status = "ok";
    }
    return { name: file.name, size: file.size, status };
  });
}

/** Outcome of applying the per-run file-count cap. */
export interface CountCapResult<T> {
  /** The files that fit within max_files_per_run (the first N). */
  accepted: T[];
  /** The files skipped because the run was already full. */
  skipped: T[];
}

/**
 * Apply the per-run file-count cap: keep the first `max_files_per_run` items and
 * return the rest as `skipped`. When the count is at or below the cap, `skipped`
 * is empty. Exactly-at-cap keeps everything.
 */
export function applyFileCountCap<T>(
  files: T[],
  config: ConfigResponse,
): CountCapResult<T> {
  const cap = Math.max(0, config.max_files_per_run);
  return {
    accepted: files.slice(0, cap),
    skipped: files.slice(cap),
  };
}

/** Human-readable file size, e.g. 31_457_280 -> "30.0 MB". */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}
