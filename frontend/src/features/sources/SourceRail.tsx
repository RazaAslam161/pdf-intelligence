import { useEffect, useRef } from "react";
import type { Source } from "../../lib/types";
import { MarginSlip } from "./MarginSlip";
import styles from "./SourceRail.module.css";

export interface SourceRailProps {
  /** Sources of the currently active assistant turn. */
  sources: Source[];
  /** Zero-based index of the highlighted source, or null. */
  activeIndex: number | null;
  /** Called when a slip is hovered/focused (sourceIndex) or left (null). */
  onActiveIndexChange: (index: number | null) => void;
}

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

/**
 * Right-rail list of source slips for the active assistant turn. Hovering or
 * focusing a slip highlights its marker(s) in the answer; when the active
 * source changes from the answer side, the matching slip scrolls into view.
 */
export function SourceRail({
  sources,
  activeIndex,
  onActiveIndexChange,
}: SourceRailProps) {
  const slipRefs = useRef<Array<HTMLLIElement | null>>([]);

  // Scroll the active slip into view when the active source changes.
  useEffect(() => {
    if (activeIndex === null) return;
    const el = slipRefs.current[activeIndex];
    if (!el) return;
    el.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "nearest",
    });
  }, [activeIndex]);

  return (
    <div className={styles.rail}>
      <h2 className={styles.heading}>Sources</h2>

      {sources.length === 0 ? (
        <p className={styles.empty}>
          Sources will appear here as answers cite them.
        </p>
      ) : (
        <ol className={styles.list}>
          {sources.map((source, i) => (
            <MarginSlip
              key={source.id}
              ref={(node) => {
                slipRefs.current[i] = node;
              }}
              source={source}
              number={i + 1}
              active={activeIndex === i}
              onActivate={() => onActiveIndexChange(i)}
              onDeactivate={() => onActiveIndexChange(null)}
            />
          ))}
        </ol>
      )}
    </div>
  );
}
