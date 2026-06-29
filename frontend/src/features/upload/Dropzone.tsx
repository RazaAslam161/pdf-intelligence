import { useRef, useState } from "react";
import type { DragEvent } from "react";
import styles from "./Dropzone.module.css";

export interface DropzoneProps {
  /** Receives dropped or browsed files. Disabled while config is loading. */
  onFiles: (files: File[]) => void;
  /** Disable interaction (e.g. before config has loaded). */
  disabled?: boolean;
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <path
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 15.5V4.5M8 8l4-4 4 4"
      />
      <path
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4.5 14.5v3a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3"
      />
    </svg>
  );
}

// Drag-and-drop dropzone with a click-to-browse fallback. The zone is a real
// <button> so it stays keyboard operable; the dragover highlight uses neutral
// tokens since the accent is reserved for the citation signal.
export function Dropzone({ onFiles, disabled = false }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  // Track nested dragenter/leave so child elements don't flicker the highlight.
  const dragDepth = useRef(0);

  function openPicker() {
    if (disabled) return;
    inputRef.current?.click();
  }

  function handleDragEnter(e: DragEvent<HTMLButtonElement>) {
    e.preventDefault();
    if (disabled) return;
    dragDepth.current += 1;
    setDragOver(true);
  }

  function handleDragOver(e: DragEvent<HTMLButtonElement>) {
    // Required so the browser fires a drop event.
    e.preventDefault();
    if (!disabled && e.dataTransfer) {
      e.dataTransfer.dropEffect = "copy";
    }
  }

  function handleDragLeave(e: DragEvent<HTMLButtonElement>) {
    e.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragOver(false);
  }

  function handleDrop(e: DragEvent<HTMLButtonElement>) {
    e.preventDefault();
    dragDepth.current = 0;
    setDragOver(false);
    if (disabled) return;
    const dropped = Array.from(e.dataTransfer?.files ?? []);
    if (dropped.length > 0) onFiles(dropped);
  }

  return (
    <button
      type="button"
      className={dragOver ? styles.zoneOver : styles.zone}
      onClick={openPicker}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      disabled={disabled}
      aria-label="Add documents: drop PDF files here, or activate to browse"
    >
      <span className={styles.icon} aria-hidden="true">
        <UploadIcon />
      </span>
      <span className={styles.primary}>
        Drop PDFs here, or <span className={styles.link}>browse</span>
      </span>
      <span className={styles.secondary}>PDF files only</span>

      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        multiple
        className={styles.input}
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          const picked = Array.from(e.target.files ?? []);
          if (picked.length > 0) onFiles(picked);
          // Reset so re-selecting the same file fires change again.
          e.target.value = "";
        }}
      />
    </button>
  );
}
