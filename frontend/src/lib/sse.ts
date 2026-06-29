/*
 * Pure Server-Sent-Events (SSE) record parsing. No fetch, no DOM — fully
 * unit-tested.
 *
 * The streaming /ask endpoint emits a sequence of SSE records. Each record is a
 * `data: <json>` line terminated by a blank line, so records are separated by
 * the delimiter "\n\n". A single network chunk may contain zero, one, or many
 * complete records, and may end PART-WAY through a record. `parseSseChunk`
 * therefore takes the running buffer (previous leftover + new chunk) and
 * returns the complete events it found plus the unconsumed `rest` to carry into
 * the next call.
 *
 * Only `data:` lines are interpreted; SSE comment lines (`:` ...), `event:`,
 * `id:` and other field lines are ignored. A `data:` payload that is not valid
 * JSON is skipped (no event emitted) rather than throwing — a single malformed
 * record must not abort the stream.
 */
import type { Source } from "./types";

/** A token delta to append to the in-progress answer. */
export interface TokenEvent {
  type: "token";
  text: string;
}

/** The grounding sources, sent once near the end of the stream. */
export interface SourcesEvent {
  type: "sources";
  sources: Source[];
}

/** The stream finished successfully. */
export interface DoneEvent {
  type: "done";
}

/** Generation failed; `message` is human-readable. */
export interface ErrorEvent {
  type: "error";
  message: string;
}

/** Any event the backend can send over the SSE stream. */
export type ParsedEvent = TokenEvent | SourcesEvent | DoneEvent | ErrorEvent;

export interface SseParseResult {
  /** Complete, well-formed events found in this pass, in order. */
  events: ParsedEvent[];
  /** Leftover bytes after the last record delimiter — carry into next call. */
  rest: string;
}

/**
 * SSE records are separated by a blank line. We accept both LF (`\n\n`) and
 * CRLF (`\r\n\r\n`) record delimiters; `findDelimiter` returns the position and
 * length of the next one so partial-record tracking stays exact.
 */
function findDelimiter(s: string): { at: number; len: number } | null {
  const lf = s.indexOf("\n\n");
  const crlf = s.indexOf("\r\n\r\n");
  if (lf === -1 && crlf === -1) return null;
  // Prefer whichever delimiter starts earlier; on a tie the CRLF form is the
  // real boundary (its `\r\n\r\n` contains an `\n\n` at the same-ish spot).
  if (crlf !== -1 && (lf === -1 || crlf <= lf)) return { at: crlf, len: 4 };
  return { at: lf, len: 2 };
}

/**
 * Coerce a parsed JSON value into a known ParsedEvent, or null if it does not
 * match any expected shape (which the caller treats as "ignore this record").
 */
function toEvent(value: unknown): ParsedEvent | null {
  if (typeof value !== "object" || value === null) return null;
  const obj = value as Record<string, unknown>;

  switch (obj.type) {
    case "token":
      return typeof obj.text === "string"
        ? { type: "token", text: obj.text }
        : null;
    case "sources":
      return Array.isArray(obj.sources)
        ? { type: "sources", sources: obj.sources as Source[] }
        : null;
    case "done":
      return { type: "done" };
    case "error":
      return {
        type: "error",
        message:
          typeof obj.message === "string" ? obj.message : "Generation failed.",
      };
    default:
      return null;
  }
}

/**
 * Extract the JSON payload from a single SSE record. A record may contain
 * multiple lines; per the SSE spec, `data:` lines are concatenated with "\n".
 * Non-`data:` lines (comments, `event:`, etc.) are ignored. Returns null when
 * the record carries no data line at all.
 */
function dataPayload(record: string): string | null {
  const dataLines: string[] = [];
  for (const rawLine of record.split("\n")) {
    // Tolerate CRLF line endings.
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line.startsWith("data:")) {
      // A single optional space after the colon is part of the SSE framing.
      const afterColon = line.slice(5);
      dataLines.push(afterColon.startsWith(" ") ? afterColon.slice(1) : afterColon);
    }
    // Any other line (":" comments, "event:", "id:", blank) is ignored.
  }
  if (dataLines.length === 0) return null;
  return dataLines.join("\n");
}

/**
 * Parse complete SSE records out of `buffer`, returning the recognised events
 * and the unconsumed remainder. The remainder always begins immediately after
 * the last complete record delimiter, so the caller should pass
 * `rest + nextChunk` on the following invocation.
 *
 * - Partial records (no trailing delimiter yet) stay in `rest` and emit nothing.
 * - Multiple records in one buffer all emit, in order.
 * - Records whose data line is absent or not valid JSON, or whose JSON does not
 *   match a known event shape, are silently dropped (no throw).
 */
export function parseSseChunk(buffer: string): SseParseResult {
  const events: ParsedEvent[] = [];

  let working = buffer;
  let delimiter = findDelimiter(working);
  while (delimiter !== null) {
    const record = working.slice(0, delimiter.at);
    working = working.slice(delimiter.at + delimiter.len);

    const payload = dataPayload(record);
    if (payload !== null) {
      try {
        const event = toEvent(JSON.parse(payload));
        if (event) events.push(event);
      } catch {
        // Malformed JSON in one record — skip it, keep the stream alive.
      }
    }

    delimiter = findDelimiter(working);
  }

  return { events, rest: working };
}
