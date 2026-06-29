import { useEffect, useRef } from "react";
import type { ActiveSource, ConversationTurn } from "./state";
import { Composer } from "./Composer";
import { ConversationEmptyState } from "./ConversationEmptyState";
import { Turn } from "./Turn";
import styles from "./ChatThread.module.css";

export interface ChatThreadProps {
  turns: ConversationTurn[];
  active: ActiveSource | null;
  /** True while an assistant answer is being fetched. */
  busy: boolean;
  onSubmit: (question: string) => void;
  onMarkerActivate: (turnId: string, index: number) => void;
  onMarkerDeactivate: () => void;
  onMarkerSelect: (turnId: string, index: number) => void;
}

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

/**
 * The conversation column: a scrollable list of turns above a fixed composer.
 * Shows the empty state when there are no turns. Auto-scrolls to the newest
 * turn as the thread grows.
 */
export function ChatThread({
  turns,
  active,
  busy,
  onSubmit,
  onMarkerActivate,
  onMarkerDeactivate,
  onMarkerSelect,
}: ChatThreadProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const hasTurns = turns.length > 0;

  // Keep the latest turn in view as the thread grows / updates.
  useEffect(() => {
    if (!hasTurns) return;
    endRef.current?.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "end",
    });
  }, [turns.length, busy, hasTurns]);

  return (
    <div className={styles.thread}>
      <div className={styles.scroll}>
        {hasTurns ? (
          <ol className={styles.turns}>
            {turns.map((turn) => (
              <li key={turn.id} className={styles.turnItem}>
                <Turn
                  turn={turn}
                  isActiveTurn={active?.turnId === turn.id}
                  activeIndex={
                    active?.turnId === turn.id ? active.sourceIndex : null
                  }
                  onMarkerActivate={onMarkerActivate}
                  onMarkerDeactivate={onMarkerDeactivate}
                  onMarkerSelect={onMarkerSelect}
                />
              </li>
            ))}
            <div ref={endRef} />
          </ol>
        ) : (
          <ConversationEmptyState />
        )}
      </div>

      <div className={styles.composerSlot}>
        <Composer onSubmit={onSubmit} disabled={busy} />
      </div>
    </div>
  );
}
