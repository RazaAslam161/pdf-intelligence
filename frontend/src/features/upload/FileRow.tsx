import type { ReactNode } from "react";
import { formatBytes } from "./preflight";
import type { FileStatus, UploadFile } from "./useUpload";
import styles from "./FileRow.module.css";

export interface FileRowProps {
  file: UploadFile;
  /** Server-advertised size cap (MB), for the "over the N MB limit" message. */
  maxUploadMb: number;
}

/** Per-status presentation: an icon + a short human label. NEVER color-only —
 *  the label always carries the meaning too (WCAG 1.4.1). */
interface StatusMeta {
  label: string;
  icon: ReactNode;
  /** Whether this status reads as a "problem" (muted-emphasis styling). */
  problem: boolean;
  /** Whether to render the small in-progress spinner instead of an icon. */
  spinning?: boolean;
}

function ClockIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <circle cx="12" cy="12" r="8.25" strokeWidth="1.5" />
      <path
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 8v4l2.5 2"
      />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      aria-hidden="true"
      className={styles.spinner}
    >
      <circle cx="12" cy="12" r="8.25" strokeWidth="1.5" opacity="0.3" />
      <path strokeWidth="1.5" strokeLinecap="round" d="M12 3.75a8.25 8.25 0 0 1 8.25 8.25" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <circle cx="12" cy="12" r="8.25" strokeWidth="1.5" />
      <path
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M8.5 12.2l2.4 2.4 4.6-5"
      />
    </svg>
  );
}

function BlockedIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <circle cx="12" cy="12" r="8.25" strokeWidth="1.5" />
      <path strokeWidth="1.5" strokeLinecap="round" d="M6.6 6.6l10.8 10.8" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <path
        strokeWidth="1.5"
        strokeLinejoin="round"
        d="M12 3.5l9 16H3l9-16z"
      />
      <path strokeWidth="1.5" strokeLinecap="round" d="M12 10v4" />
      <circle cx="12" cy="16.6" r="0.6" fill="currentColor" stroke="none" />
    </svg>
  );
}

function statusMeta(status: FileStatus): StatusMeta {
  switch (status) {
    case "queued":
      return { label: "Queued", icon: <ClockIcon />, problem: false };
    case "indexing":
      return {
        label: "Indexing",
        icon: <SpinnerIcon />,
        problem: false,
        spinning: true,
      };
    case "indexed":
      return { label: "Indexed", icon: <CheckIcon />, problem: false };
    case "rejected":
      return { label: "Rejected", icon: <BlockedIcon />, problem: true };
    case "failed":
      return { label: "Failed", icon: <AlertIcon />, problem: true };
    case "too_large":
      return { label: "Too large", icon: <BlockedIcon />, problem: true };
    case "unsupported":
      return { label: "Not a PDF", icon: <BlockedIcon />, problem: true };
    default:
      return { label: status, icon: <AlertIcon />, problem: true };
  }
}

/**
 * One document row: name, size, a status (icon + label + text), and — for
 * indexed rows — page/chunk counts. Each row is a focusable list item so the
 * list is keyboard-reachable. Status is conveyed by icon AND label AND
 * supporting text, never color alone.
 */
export function FileRow({ file, maxUploadMb }: FileRowProps) {
  const meta = statusMeta(file.status);

  // The supporting detail line, by status.
  let detail: string | null = null;
  if (file.status === "too_large") {
    detail = `${formatBytes(file.size)} — over the ${maxUploadMb} MB limit`;
  } else if (file.status === "unsupported") {
    detail = "Only .pdf files can be indexed";
  } else if (file.status === "indexed") {
    detail = `${file.pageCount ?? 0} pages · ${file.chunkCount ?? 0} chunks`;
  } else if (
    (file.status === "rejected" || file.status === "failed") &&
    file.detail
  ) {
    detail = file.detail;
  }

  const rowClass = meta.problem ? styles.rowProblem : styles.row;

  return (
    <li
      className={rowClass}
      tabIndex={0}
      aria-label={`${file.name}, ${formatBytes(file.size)}, ${meta.label}${
        detail ? `. ${detail}` : ""
      }`}
    >
      <span className={styles.icon} aria-hidden="true">
        {meta.icon}
      </span>

      <div className={styles.body}>
        <div className={styles.line}>
          <span className={styles.name}>{file.name}</span>
          <span className={styles.size}>{formatBytes(file.size)}</span>
        </div>
        {detail ? <span className={styles.detail}>{detail}</span> : null}
      </div>

      <span className={styles.status}>{meta.label}</span>
    </li>
  );
}
