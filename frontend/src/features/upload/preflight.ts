// Client-side pre-flight checks so we don't ship files the server will reject.
// Side-effect-free: enforces the size cap and .pdf requirement per file, plus
// the per-run file-count cap.
import type { ConfigResponse } from "../../lib/types";

// Just enough of a file to categorize, so tests don't need a real File/Blob.
export interface FileDescriptor {
  name: string;
  size: number;
}

export type PreflightStatus = "ok" | "too_large" | "unsupported";

export interface PreflightResult extends FileDescriptor {
  status: PreflightStatus;
}

export const BYTES_PER_MB = 1024 * 1024;

export function isPdfName(name: string): boolean {
  return /\.pdf$/i.test(name.trim());
}

// Extension is checked before size, first failure wins. A file exactly at the
// cap is accepted; the server has final say on empty 0-byte PDFs.
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

export interface CountCapResult<T> {
  accepted: T[];
  skipped: T[];
}

// Keep the first max_files_per_run items; the rest land in skipped.
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
