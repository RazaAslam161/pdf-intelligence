import { Fragment } from "react";
import { parseAnswerWithBounds } from "../citations/citations";
import styles from "./AnswerText.module.css";

export interface AnswerTextProps {
  answer: string;
  /** Number of sources for this turn — bounds which markers are interactive. */
  sourceCount: number;
  /** Zero-based index of the currently highlighted source, or null. */
  activeIndex: number | null;
  /**
   * True while tokens are still streaming in. Shows a trailing caret and an
   * SR-only status. While streaming `sourceCount` is 0, so markers stay plain
   * text until the sources arrive.
   */
  streaming?: boolean;
  /** Hover/focus/click a marker -> activate that source. */
  onMarkerActivate: (index: number) => void;
  /** Hover-out / blur a marker -> clear the transient highlight. */
  onMarkerDeactivate: () => void;
  /** Click a marker -> make this turn's sources the active set + select. */
  onMarkerSelect: (index: number) => void;
}

/**
 * Renders an answer as prose with inline citation markers. In-range `[n]`
 * markers become superscript buttons; out-of-range ones stay plain text.
 */
export function AnswerText({
  answer,
  sourceCount,
  activeIndex,
  streaming = false,
  onMarkerActivate,
  onMarkerDeactivate,
  onMarkerSelect,
}: AnswerTextProps) {
  const segments = parseAnswerWithBounds(answer, sourceCount);

  return (
    <p className={styles.answer}>
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          return <Fragment key={i}>{seg.value}</Fragment>;
        }
        const number = seg.index + 1;
        const isActive = activeIndex === seg.index;
        return (
          <sup key={i} className={styles.markerWrap}>
            <button
              type="button"
              className={isActive ? styles.markerActive : styles.marker}
              aria-label={`Source ${number}`}
              onMouseEnter={() => onMarkerActivate(seg.index)}
              onMouseLeave={onMarkerDeactivate}
              onFocus={() => onMarkerActivate(seg.index)}
              onBlur={onMarkerDeactivate}
              onClick={() => onMarkerSelect(seg.index)}
            >
              {number}
            </button>
          </sup>
        );
      })}
      {streaming ? (
        <>
          <span className={styles.caret} aria-hidden="true" />
          <span className={styles.srOnly} role="status">
            Generating answer
          </span>
        </>
      ) : null}
    </p>
  );
}
