/*
 * Pure SSE record parsing — no fetch, no DOM.
 *
 * The /ask stream sends `data: <json>` records separated by a blank line. A
 * chunk can contain any number of complete records and may end mid-record, so
 * parseSseChunk takes the running buffer and returns the events it found plus
 * the leftover to carry into the next call. Non-`data:` lines are ignored, and
 * a record with invalid JSON is skipped rather than killing the stream.
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

// Find the next record boundary, accepting both LF and CRLF blank-line forms.
// Returns its position and length so we know exactly where the record ends.
function findDelimiter(s: string): { at: number; len: number } | null {
  const lf = s.indexOf("\n\n");
  const crlf = s.indexOf("\r\n\r\n");
  if (lf === -1 && crlf === -1) return null;
  // Whichever starts earlier wins; on a tie prefer CRLF, since its \r\n\r\n
  // already contains the \n\n the LF check would match.
  if (crlf !== -1 && (lf === -1 || crlf <= lf)) return { at: crlf, len: 4 };
  return { at: lf, len: 2 };
}

// Coerce parsed JSON into a known ParsedEvent, or null if it matches no shape.
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

// Pull the JSON payload out of a record. Per the SSE spec multiple `data:`
// lines join with "\n"; everything else is ignored. Null if there's no data.
function dataPayload(record: string): string | null {
  const dataLines: string[] = [];
  for (const rawLine of record.split("\n")) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line.startsWith("data:")) {
      // One optional space after the colon is part of the framing.
      const afterColon = line.slice(5);
      dataLines.push(afterColon.startsWith(" ") ? afterColon.slice(1) : afterColon);
    }
  }
  if (dataLines.length === 0) return null;
  return dataLines.join("\n");
}

/**
 * Parse complete records out of `buffer`, returning the events found and the
 * unconsumed remainder. The remainder starts right after the last delimiter, so
 * pass `rest + nextChunk` next time. Partial records stay in `rest`; unparseable
 * ones are dropped silently.
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
        // Skip a malformed record, keep the stream alive.
      }
    }

    delimiter = findDelimiter(working);
  }

  return { events, rest: working };
}
