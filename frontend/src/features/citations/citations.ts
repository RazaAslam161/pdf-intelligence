/*
 * Pure citation logic. No React, no DOM — fully unit-tested.
 *
 * The assistant answer string MAY contain inline citation markers like [1],
 * [2] — 1-based indices that reference `sources` by POSITION (sources[0] is the
 * target of marker [1]). Markers may be absent entirely (older / uncited
 * answers); callers must handle that gracefully.
 *
 * `score` from the backend is COSINE DISTANCE (0 = identical, higher = less
 * similar), NOT a similarity. Display similarity is `1 - score`.
 */

/** A run of plain prose between (or around) citation markers. */
export interface TextSegment {
  type: "text";
  value: string;
}

/**
 * A parsed citation marker. `index` is the ZERO-BASED source index
 * (marker [1] -> index 0), so it indexes straight into `sources`.
 */
export interface CiteSegment {
  type: "cite";
  /** Zero-based index into the sources array. */
  index: number;
}

export type AnswerSegment = TextSegment | CiteSegment;

/** Matches a citation marker like [1], [12]. Captures the 1-based number. */
const CITE_PATTERN = /\[(\d+)\]/g;

/**
 * Parse an answer into an ordered list of text / cite segments.
 *
 * Behaviour:
 * - `[n]` becomes a CiteSegment with a zero-based `index` (n - 1).
 * - `[0]` is not a valid 1-based marker and is emitted as plain text.
 * - Surrounding prose is preserved verbatim as TextSegments, including the
 *   text between two adjacent markers (which may be empty and is then omitted).
 * - The caller is responsible for treating an `index` that is out of range
 *   (>= sources.length) as plain text — `parseAnswer` does not know how many
 *   sources exist. See `parseAnswerWithBounds` for the bounded variant.
 *
 * @example parseAnswer("a [1] b") ->
 *   [{text,"a "},{cite,0},{text," b"}]
 */
export function parseAnswer(answer: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  if (!answer) return segments;

  let lastIndex = 0;
  // Reset lastIndex defensively — the regex is module-scoped and stateful.
  CITE_PATTERN.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = CITE_PATTERN.exec(answer)) !== null) {
    const oneBased = Number(match[1]);

    // [0] is not a valid 1-based citation; keep it as literal text by skipping.
    if (oneBased < 1) {
      continue;
    }

    // Emit any prose before this marker.
    if (match.index > lastIndex) {
      segments.push({ type: "text", value: answer.slice(lastIndex, match.index) });
    }

    segments.push({ type: "cite", index: oneBased - 1 });
    lastIndex = match.index + match[0].length;
  }

  // Trailing prose after the final marker.
  if (lastIndex < answer.length) {
    segments.push({ type: "text", value: answer.slice(lastIndex) });
  }

  return segments;
}

/**
 * Like `parseAnswer`, but a cite whose zero-based index falls outside
 * `[0, sourceCount)` is downgraded to plain text (the raw `[n]` marker). This
 * is what the renderer uses so out-of-range markers appear as ordinary text.
 */
export function parseAnswerWithBounds(
  answer: string,
  sourceCount: number,
): AnswerSegment[] {
  const raw = parseAnswer(answer);
  const out: AnswerSegment[] = [];

  const pushText = (value: string): void => {
    // Coalesce with a preceding text segment so adjacent prose / downgraded
    // markers always normalize into a single segment.
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
      // Out of range: render the original marker as literal text.
      pushText(`[${seg.index + 1}]`);
    } else {
      out.push(seg);
    }
  }

  return out;
}

/**
 * Convert a backend cosine-DISTANCE score into a display similarity rounded to
 * two decimals. Returns null when score is null (nothing to display).
 *
 *   score 0.17 -> 0.83 ; score 0    -> 1 ; score 1 -> 0
 *
 * Values are clamped to [0, 1] so a distance > 1 never yields a negative
 * similarity.
 */
export function similarityFromScore(score: number | null): number | null {
  if (score === null) return null;
  const sim = 1 - score;
  const clamped = Math.min(1, Math.max(0, sim));
  // Round half-up to 2 dp.
  return Math.round(clamped * 100) / 100;
}

/**
 * Map a cosine-distance score to a 0..4 confidence quartile based on the
 * derived similarity (sim = 1 - score):
 *
 *   sim >= 0.70            -> 4
 *   sim >= 0.55            -> 3
 *   sim >= 0.40            -> 2
 *   sim present but < 0.40 -> 1
 *   score === null         -> 0  (no signal)
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
