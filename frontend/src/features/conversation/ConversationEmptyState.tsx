import styles from "./ConversationEmptyState.module.css";

/**
 * Placeholder for the conversation column before any exchange exists. Chat
 * logic arrives in a later milestone; this is the empty / first-run state only.
 */
export function ConversationEmptyState() {
  return (
    <div className={styles.empty}>
      <h1 className={styles.title}>Ask your documents</h1>
      <p className={styles.lede}>
        Index a set of PDFs, then ask questions in plain language. Every answer
        is traced back to the pages it draws from.
      </p>
      <p className={styles.hint}>
        No conversation yet. Your questions and answers will appear here.
      </p>
    </div>
  );
}
