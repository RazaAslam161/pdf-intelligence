import type { DocumentSummary } from "../../lib/types";
import { documentMeta } from "./summary";
import styles from "./DocumentRow.module.css";

export interface DocumentRowProps {
  document: DocumentSummary;
}

/** Decorative document glyph. */
function DocIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <path
        strokeWidth="1.5"
        strokeLinejoin="round"
        d="M7 3.75h6.5L18 8.25V20a.75.75 0 0 1-.75.75H7A.75.75 0 0 1 6.25 20V4.5A.75.75 0 0 1 7 3.75z"
      />
      <path strokeWidth="1.5" strokeLinejoin="round" d="M13.25 3.75V8.5H18" />
    </svg>
  );
}

/** One row in the documents list, focusable so the list is keyboard-reachable. */
export function DocumentRow({ document }: DocumentRowProps) {
  const meta = documentMeta(document.page_count, document.chunk_count);

  return (
    <li
      className={styles.row}
      tabIndex={0}
      aria-label={`${document.file_name}. ${meta}`}
    >
      <span className={styles.icon} aria-hidden="true">
        <DocIcon />
      </span>
      <div className={styles.body}>
        <span className={styles.name}>{document.file_name}</span>
        <span className={styles.meta}>{meta}</span>
      </div>
    </li>
  );
}
