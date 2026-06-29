import { forwardRef } from "react";
import type { Source } from "../../lib/types";
import { similarityFromScore } from "../citations/citations";
import { ConfidenceComb } from "./ConfidenceComb";
import styles from "./MarginSlip.module.css";

export interface MarginSlipProps {
  source: Source;
  /** 1-based number shown on the slip and matching the inline marker. */
  number: number;
  /** Whether this slip is the active (highlighted) source. */
  active: boolean;
  /** Hover/focus on a slip highlights its marker(s) in the answer. */
  onActivate: () => void;
  onDeactivate: () => void;
}

/**
 * A "margin slip" card for one grounding source. Shows the source number, file
 * name + page badge, the preview (the whole preview underlined in accent as the
 * evidence span — we have no sub-span offsets), the display similarity as a
 * real number, and a confidence comb.
 *
 * The accent appears here ONLY as the evidence underline, the comb fill, and
 * the active left-edge highlight — all grounding signals.
 */
export const MarginSlip = forwardRef<HTMLLIElement, MarginSlipProps>(
  function MarginSlip(
    { source, number, active, onActivate, onDeactivate },
    ref,
  ) {
    const similarity = similarityFromScore(source.score);

    return (
      <li
        ref={ref}
        className={active ? styles.slipActive : styles.slip}
        tabIndex={0}
        aria-label={`Source ${number}: ${source.file_name}, page ${source.page_number}`}
        onMouseEnter={onActivate}
        onMouseLeave={onDeactivate}
        onFocus={onActivate}
        onBlur={onDeactivate}
      >
        <div className={styles.head}>
          <span className={styles.number} aria-hidden="true">
            {number}
          </span>
          <div className={styles.meta}>
            <span className={styles.fileName}>{source.file_name}</span>
            <span className={styles.pageBadge}>p. {source.page_number}</span>
          </div>
        </div>

        <p className={styles.preview}>
          <span className={styles.evidence}>{source.preview}</span>
        </p>

        <div className={styles.footer}>
          <span className={styles.similarity}>
            {similarity === null ? (
              <span className={styles.noScore}>no score</span>
            ) : (
              <>
                <span className={styles.simLabel}>sim</span>
                <span className={styles.simValue}>{similarity.toFixed(2)}</span>
              </>
            )}
          </span>
          <ConfidenceComb score={source.score} />
        </div>
      </li>
    );
  },
);
