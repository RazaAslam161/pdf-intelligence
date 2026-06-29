import type { DocumentSummary } from "../../lib/types";
import { DocumentRow } from "./DocumentRow";
import styles from "./DocumentList.module.css";

export interface DocumentListProps {
  documents: DocumentSummary[];
}

/** Renders the document list, or an empty-state line when there are none. */
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
