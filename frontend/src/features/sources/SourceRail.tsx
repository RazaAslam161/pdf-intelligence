import styles from "./SourceRail.module.css";

/**
 * Placeholder for the source rail. Citation rendering arrives in a later
 * milestone; this is the quiet empty state shown before any answer cites a
 * source.
 */
export function SourceRail() {
  return (
    <div className={styles.rail}>
      <h2 className={styles.heading}>Sources</h2>
      <p className={styles.empty}>
        Sources will appear here as answers cite them.
      </p>
    </div>
  );
}
