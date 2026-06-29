/*
 * Pure citation logic, no React or DOM.
 *
 * Answers may contain 1-based inline markers like [1] that reference `sources`
 * by position (so [1] -> sources[0]), or no markers at all. Backend `score` is
 * cosine distance (0 = identical), so display similarity is `1 - score`.
 */

/** Plain prose between or around citation markers. */
export interface TextSegment {
  type: "text";
  value: string;
}

/** A parsed citation marker. `index` is zero-based, so [1] -> index 0. */
export interface CiteSegment {
  type: "cite";
  /** Zero-based index into the sources array. */
  index: number;
}

export type AnswerSegment = TextSegment | CiteSegment;

/** Matches a marker like [1], [12]; captures the 1-based number. */
const CITE_PATTERN = /\[(\d+)\]/g;

/**
 * Parse an answer into ordered text / cite segments. `[n]` becomes a cite with
 * zero-based index (n - 1); `[0]` and surrounding prose stay as text. Doesn't
 * know how many sources exist — see `parseAnswerWithBounds` for the bounded
 * variant that downgrades out-of-range markers.
 */
export function parseAnswer(answer: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  if (!answer) return segments;

  let lastIndex = 0;
  // The regex is module-scoped and stateful, so reset it.
  CITE_PATTERN.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = CITE_PATTERN.exec(answer)) !== null) {
    const oneBased = Number(match[1]);

    // [0] isn't a valid 1-based citation; leave it as literal text.
    if (oneBased < 1) {
      continue;
    }

    if (match.index > lastIndex) {
      segments.push({ type: "text", value: answer.slice(lastIndex, match.index) });
    }

    segments.push({ type: "cite", index: oneBased - 1 });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < answer.length) {
    segments.push({ type: "text", value: answer.slice(lastIndex) });
  }

  return segments;
}

/**
 * Like `parseAnswer`, but a cite whose index falls outside `[0, sourceCount)`
 * is downgraded to its raw `[n]` text. The renderer uses this.
 */
export function parseAnswerWithBounds(
  answer: string,
  sourceCount: number,
): AnswerSegment[] {
  const raw = parseAnswer(answer);
  const out: AnswerSegment[] = [];

  const pushText = (value: string): void => {
    // Coalesce with a preceding text segment so adjacent prose merges.
    const prev = out[out.length - 1];
    if (prev && prev.type === "text") {
      prev.value += value;
    } else {
      out.push({ type: "text", value });
    }
  };

  for (const seg of raw) {
    if (seg.type === "text") {
      pushText(seg.value);
    } else if (seg.index < 0 || seg.index >= sourceCount) {
      pushText(`[${seg.index + 1}]`);
    } else {
      out.push(seg);
    }
  }

  return out;
}

/**
 * Convert a cosine-distance score into a display similarity (1 - score) rounded
 * to 2 dp and clamped to [0, 1]. Returns null when score is null.
 */
export function similarityFromScore(score: number | null): number | null {
  if (score === null) return null;
  const sim = 1 - score;
  const clamped = Math.min(1, Math.max(0, sim));
  return Math.round(clamped * 100) / 100;
}

/**
 * Map a score to a 0..4 confidence quartile by similarity thresholds
 * (0.70 / 0.55 / 0.40). Returns 0 when there's no score.
 */
export function confidenceQuartile(score: number | null): 0 | 1 | 2 | 3 | 4 {
  if (score === null) return 0;
  const sim = similarityFromScore(score);
  if (sim === null) return 0;
  if (sim >= 0.7) return 4;
  if (sim >= 0.55) return 3;
  if (sim >= 0.4) return 2;
  return 1;
}
