import type { ConversationTurn } from "./state";
import { AnswerText } from "./AnswerText";
import styles from "./Turn.module.css";

export interface TurnProps {
  turn: ConversationTurn;
  /** True when this turn's sources are the active set shown in the rail. */
  isActiveTurn: boolean;
  /** Active source index within the active turn, or null. */
  activeIndex: number | null;
  onMarkerActivate: (turnId: string, index: number) => void;
  onMarkerDeactivate: () => void;
  onMarkerSelect: (turnId: string, index: number) => void;
}

function PendingDots() {
  return (
    <span className={styles.pending} aria-label="Generating answer" role="status">
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.dot} />
    </span>
  );
}

/** One message in the thread: a user prompt or an assistant answer. */
export function Turn({
  turn,
  isActiveTurn,
  activeIndex,
  onMarkerActivate,
  onMarkerDeactivate,
  onMarkerSelect,
}: TurnProps) {
  const isUser = turn.role === "user";

  return (
    <article
      className={isUser ? styles.userTurn : styles.assistantTurn}
      data-role={turn.role}
    >
      <div className={styles.roleLabel}>{isUser ? "You" : "Assistant"}</div>

      <div className={styles.bubble}>
        {isUser ? (
          <p className={styles.userText}>{turn.content}</p>
        ) : turn.status === "pending" ? (
          <PendingDots />
        ) : turn.status === "error" ? (
          <p className={styles.error} role="alert">
            {turn.error ?? "Something went wrong fetching this answer."}
          </p>
        ) : (
          <AnswerText
            answer={turn.content}
            sourceCount={turn.sources.length}
            activeIndex={isActiveTurn ? activeIndex : null}
            streaming={turn.status === "streaming"}
            onMarkerActivate={(index) => onMarkerActivate(turn.id, index)}
            onMarkerDeactivate={onMarkerDeactivate}
            onMarkerSelect={(index) => onMarkerSelect(turn.id, index)}
          />
        )}
      </div>
    </article>
  );
}
