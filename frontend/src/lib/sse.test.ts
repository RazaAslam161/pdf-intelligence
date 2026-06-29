import { describe, expect, it } from "vitest";
import { parseSseChunk, type ParsedEvent } from "./sse";

/** Build one SSE record (data line + blank-line terminator). */
function record(json: unknown): string {
  return `data: ${JSON.stringify(json)}\n\n`;
}

describe("parseSseChunk", () => {
  it("returns no events and empty rest for an empty buffer", () => {
    expect(parseSseChunk("")).toEqual({ events: [], rest: "" });
  });

  it("parses a single complete token record", () => {
    const result = parseSseChunk(record({ type: "token", text: "Hello" }));
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "Hello" },
    ]);
    expect(result.rest).toBe("");
  });

  it("parses multiple records in a single chunk, in order", () => {
    const buffer =
      record({ type: "token", text: "a" }) +
      record({ type: "token", text: "b" }) +
      record({ type: "done" });
    const result = parseSseChunk(buffer);
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "a" },
      { type: "token", text: "b" },
      { type: "done" },
    ]);
    expect(result.rest).toBe("");
  });

  it("holds a partial record (no trailing delimiter) in rest", () => {
    // Two complete records, then a third that is cut off mid-JSON.
    const buffer =
      record({ type: "token", text: "x" }) + 'data: {"type":"tok';
    const result = parseSseChunk(buffer);
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "x" },
    ]);
    expect(result.rest).toBe('data: {"type":"tok');
  });

  it("resumes a record split mid-record across two chunks", () => {
    const full = record({ type: "token", text: "split" });
    const cut = Math.floor(full.length / 2);
    const first = parseSseChunk(full.slice(0, cut));
    expect(first.events).toEqual([]);
    // Carry the leftover into the next chunk.
    const second = parseSseChunk(first.rest + full.slice(cut));
    expect(second.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "split" },
    ]);
    expect(second.rest).toBe("");
  });

  it("handles the 'data:' prefix with no space after the colon", () => {
    const result = parseSseChunk('data:{"type":"token","text":"tight"}\n\n');
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "tight" },
    ]);
  });

  it("strips exactly one leading space after 'data:' (preserving the rest)", () => {
    // Two spaces -> the payload keeps one space, which is still valid JSON here
    // only because JSON tolerates leading whitespace.
    const result = parseSseChunk('data:  {"type":"token","text":"y"}\n\n');
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "y" },
    ]);
  });

  it("ignores non-data lines (comments, event:, id:)", () => {
    const buffer =
      ": keep-alive comment\n\n" +
      "event: message\ndata: " +
      JSON.stringify({ type: "token", text: "z" }) +
      "\n\n";
    const result = parseSseChunk(buffer);
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "z" },
    ]);
    expect(result.rest).toBe("");
  });

  it("parses a sources event with the source array", () => {
    const sources = [
      {
        id: "s1",
        file_name: "a.pdf",
        page_number: 3,
        preview: "hi",
        score: 0.2,
      },
    ];
    const result = parseSseChunk(record({ type: "sources", sources }));
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "sources", sources },
    ]);
  });

  it("parses an error event and carries the message", () => {
    const result = parseSseChunk(
      record({ type: "error", message: "boom" }),
    );
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "error", message: "boom" },
    ]);
  });

  it("defaults an error event with no message to a generic string", () => {
    const result = parseSseChunk('data: {"type":"error"}\n\n');
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "error", message: "Generation failed." },
    ]);
  });

  it("drops a record with malformed JSON without throwing, keeping later events", () => {
    const buffer =
      "data: {not valid json}\n\n" +
      record({ type: "token", text: "after" });
    const result = parseSseChunk(buffer);
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "after" },
    ]);
    expect(result.rest).toBe("");
  });

  it("drops a record whose JSON has an unknown type", () => {
    const buffer =
      record({ type: "heartbeat" }) +
      record({ type: "token", text: "ok" });
    const result = parseSseChunk(buffer);
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "ok" },
    ]);
  });

  it("tolerates CRLF line endings", () => {
    const result = parseSseChunk(
      'data: {"type":"token","text":"crlf"}\r\n\r\n',
    );
    expect(result.events).toEqual<ParsedEvent[]>([
      { type: "token", text: "crlf" },
    ]);
  });

  it("preserves a leftover record when a delimiter is followed by a partial", () => {
    const buffer =
      record({ type: "token", text: "1" }) + "data: {\"type\":\"to";
    const { events, rest } = parseSseChunk(buffer);
    expect(events).toEqual<ParsedEvent[]>([{ type: "token", text: "1" }]);
    expect(rest).toBe('data: {"type":"to');
  });
});
