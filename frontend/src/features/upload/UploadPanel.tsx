import { Button } from "../../components/Button/Button";
import { IconButton } from "../../components/IconButton/IconButton";
import { Dropzone } from "./Dropzone";
import { FileRow } from "./FileRow";
import { useUpload } from "./useUpload";
import styles from "./UploadPanel.module.css";

export interface UploadPanelProps {
  /** Close the panel and return to the conversation. */
  onClose: () => void;
  /** Fired after an upload batch finishes, so the live store can refresh. */
  onBatchComplete?: () => void;
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <path
        strokeWidth="1.5"
        strokeLinecap="round"
        d="M6 6l12 12M18 6L6 18"
      />
    </svg>
  );
}

/** Dropzone plus the list of documents added this session. Owns its own upload
 * state via useUpload, so it stays independent of the chat feature. */
export function UploadPanel({ onClose, onBatchComplete }: UploadPanelProps) {
  const { config, configError, files, uploading, skippedNote, addFiles, clear } =
    useUpload({ onBatchComplete });

  const hasFiles = files.length > 0;
  const maxMb = config?.max_upload_mb ?? 0;

  return (
    <section className={styles.panel} aria-label="Add documents">
      <header className={styles.head}>
        <div className={styles.titleGroup}>
          <h2 className={styles.title}>Documents</h2>
          {config ? (
            <p className={styles.limits}>
              Up to {config.max_files_per_run} files per run · {config.max_upload_mb}{" "}
              MB each · {config.max_pages_per_pdf} pages max
            </p>
          ) : configError ? (
            <p className={styles.limitsProblem}>{configError}</p>
          ) : (
            <p className={styles.limits}>Loading limits…</p>
          )}
        </div>
        <IconButton aria-label="Close documents panel" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </header>

      <Dropzone onFiles={addFiles} disabled={config === null} />

      {skippedNote ? (
        <p className={styles.note} role="status">
          {skippedNote.count} file{skippedNote.count === 1 ? "" : "s"} skipped —
          per-run limit is {skippedNote.cap}.
        </p>
      ) : null}

      <div className={styles.listHead}>
        <h3 className={styles.listTitle}>
          This session{hasFiles ? ` (${files.length})` : ""}
        </h3>
        {hasFiles ? (
          <Button
            variant="default"
            onClick={clear}
            disabled={uploading}
            aria-label="Clear the document list"
          >
            Clear
          </Button>
        ) : null}
      </div>

      {hasFiles ? (
        <ul className={styles.list}>
          {files.map((file) => (
            <FileRow key={file.id} file={file} maxUploadMb={maxMb} />
          ))}
        </ul>
      ) : (
        <p className={styles.empty}>
          No documents yet. Add PDFs above to index them for this session.
        </p>
      )}
    </section>
  );
}
