import styles from "./ConfidenceComb.module.css";
import { confidenceQuartile } from "../citations/citations";

export interface ConfidenceCombProps {
  /** Cosine-distance score from the backend; null = no signal. */
  score: number | null;
}

const SEGMENTS = [0, 1, 2, 3];

/**
 * 4-segment indicator for a source's similarity quartile. Meaning is also in the
 * aria-label so it doesn't rely on colour alone; a null score reads as
 * "confidence unavailable".
 */
export function ConfidenceComb({ score }: ConfidenceCombProps) {
  const filled = confidenceQuartile(score);
  const label =
    score === null
      ? "confidence unavailable"
      : `confidence ${filled} of 4`;

  return (
    <span className={styles.comb} role="img" aria-label={label}>
      {SEGMENTS.map((i) => (
        <span
          key={i}
          className={i < filled ? styles.segmentFilled : styles.segmentEmpty}
          aria-hidden="true"
        />
      ))}
    </span>
  );
}
