import styles from "./ConfidenceComb.module.css";
import { confidenceQuartile } from "../citations/citations";

export interface ConfidenceCombProps {
  /** Cosine-distance score from the backend; null = no signal. */
  score: number | null;
}

const SEGMENTS = [0, 1, 2, 3];

/**
 * A 4-segment indicator encoding the similarity quartile of a source.
 *
 * Filled segments use the accent (grounding signal); empty segments are a 1px
 * muted outline. Meaning is also carried by `aria-label` ("confidence N of 4"),
 * so the indicator never relies on colour alone. When score is null the comb is
 * empty and the label reads "confidence unavailable".
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
