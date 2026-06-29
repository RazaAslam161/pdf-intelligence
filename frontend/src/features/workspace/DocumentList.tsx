import type { DocumentSummary } from "../../lib/types";
import { DocumentRow } from "./DocumentRow";
import styles from "./DocumentList.module.css";

export interface DocumentListProps {
  documents: DocumentSummary[];
}

/**
 * The live list of documents in the store. When empty, renders the calm
 * empty-state sentence instead of an empty <ul>. Empty vs populated is conveyed
 * by text + structure, never color.
 */
export function DocumentList({ documents }: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <p className={styles.empty}>
        No documents indexed yet — add a PDF to start asking questions.
      </p>
    );
  }

  return (
    <ul className={styles.list}>
      {documents.map((doc) => (
        <DocumentRow key={doc.document_id} document={doc} />
      ))}
    </ul>
  );
}
