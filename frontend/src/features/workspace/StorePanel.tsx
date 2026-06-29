import { useState } from "react";
import { Button } from "../../components/Button/Button";
import type { StoreState } from "../../lib/types";
import { DocumentList } from "./DocumentList";
import { readinessSummary } from "./summary";
import type { StorePhase } from "./useStore";
import styles from "./StorePanel.module.css";

export interface StorePanelProps {
  state: StoreState | null;
  phase: StorePhase;
  error: string | null;
  clearing: boolean;
  /** Wipe the store (DELETE /documents). */
  onClear: () => void;
}

function ReadyIcon() {
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

function EmptyIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
      <circle cx="12" cy="12" r="8.25" strokeWidth="1.5" />
      <path strokeWidth="1.5" strokeLinecap="round" d="M9 12h6" />
    </svg>
  );
}

/**
 * The "Documents in store" section. Reads from GET /documents, so it reflects
 * what's actually persisted, not just files uploaded this session.
 */
export function StorePanel({
  state,
  phase,
  error,
  clearing,
  onClear,
}: StorePanelProps) {
  const [confirming, setConfirming] = useState(false);

  const ready = state?.ready === true && state.total_chunks > 0;
  const isEmpty = !ready;
  // Clear is disabled while clearing, while empty, and before the first load.
  const clearDisabled = clearing || isEmpty || state === null;

  const handleClear = () => {
    setConfirming(false);
    onClear();
  };

  return (
    <section className={styles.panel} aria-label="Documents in store">
      <div className={styles.head}>
        <h3 className={styles.title}>Documents in store</h3>
        {state !== null && !confirming ? (
          <Button
            variant="default"
            onClick={() => setConfirming(true)}
            disabled={clearDisabled}
            aria-label="Clear all documents from the store"
          >
            {clearing ? "Clearing…" : "Clear all"}
          </Button>
        ) : null}
        {state !== null && confirming ? (
          <div className={styles.confirm} role="group" aria-label="Confirm clear">
            <span className={styles.confirmText}>Clear all documents?</span>
            <Button
              variant="default"
              onClick={handleClear}
              disabled={clearing}
              aria-label="Confirm: clear all documents"
            >
              Confirm
            </Button>
            <Button
              variant="default"
              onClick={() => setConfirming(false)}
              disabled={clearing}
              aria-label="Cancel clearing documents"
            >
              Cancel
            </Button>
          </div>
        ) : null}
      </div>

      {phase === "loading" && state === null ? (
        <p className={styles.status} role="status">
          Loading the document store…
        </p>
      ) : phase === "error" && state === null ? (
        <p className={styles.problem} role="status">
          {error ?? "Could not load the document store."}
        </p>
      ) : state !== null ? (
        <>
          <p
            className={ready ? styles.summaryReady : styles.summaryEmpty}
            role="status"
          >
            <span className={styles.summaryIcon} aria-hidden="true">
              {ready ? <ReadyIcon /> : <EmptyIcon />}
            </span>
            <span>{readinessSummary(state)}</span>
          </p>
          <DocumentList documents={state.documents} />
        </>
      ) : null}
    </section>
  );
}
