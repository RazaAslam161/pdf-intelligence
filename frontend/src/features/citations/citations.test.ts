import { describe, expect, it } from "vitest";
import {
  confidenceQuartile,
  parseAnswer,
  parseAnswerWithBounds,
  similarityFromScore,
} from "./citations";

describe("parseAnswer", () => {
  it("returns an empty array for an empty string", () => {
    expect(parseAnswer("")).toEqual([]);
  });

  it("returns a single text segment when there are no markers", () => {
    expect(parseAnswer("just plain prose")).toEqual([
      { type: "text", value: "just plain prose" },
    ]);
  });

  it("parses a single marker with surrounding text", () => {
    expect(parseAnswer("The sky is blue [1] today.")).toEqual([
      { type: "text", value: "The sky is blue " },
      { type: "cite", index: 0 },
      { type: "text", value: " today." },
    ]);
  });

  it("converts 1-based markers to zero-based indices", () => {
    expect(parseAnswer("a [3] b")).toEqual([
      { type: "text", value: "a " },
      { type: "cite", index: 2 },
      { type: "text", value: " b" },
    ]);
  });

  it("handles multi-digit markers", () => {
    expect(parseAnswer("see [12]")).toEqual([
      { type: "text", value: "see " },
      { type: "cite", index: 11 },
    ]);
  });

  it("handles a marker at the very start", () => {
    expect(parseAnswer("[1] leads")).toEqual([
      { type: "cite", index: 0 },
      { type: "text", value: " leads" },
    ]);
  });

  it("handles adjacent markers with no text between them", () => {
    expect(parseAnswer("[1][2]")).toEqual([
      { type: "cite", index: 0 },
      { type: "cite", index: 1 },
    ]);
  });

  it("handles adjacent markers surrounded by prose", () => {
    expect(parseAnswer("x [1][2] y")).toEqual([
      { type: "text", value: "x " },
      { type: "cite", index: 0 },
      { type: "cite", index: 1 },
      { type: "text", value: " y" },
    ]);
  });

  it("treats [0] as plain text (not a valid 1-based marker)", () => {
    expect(parseAnswer("ref [0] here")).toEqual([
      { type: "text", value: "ref [0] here" },
    ]);
  });

  it("parses multiple separated markers", () => {
    expect(parseAnswer("First [1] then [2] last")).toEqual([
      { type: "text", value: "First " },
      { type: "cite", index: 0 },
      { type: "text", value: " then " },
      { type: "cite", index: 1 },
      { type: "text", value: " last" },
    ]);
  });

  it("is stateless across calls (regex lastIndex reset)", () => {
    const a = parseAnswer("a [1]");
    const b = parseAnswer("b [1]");
    expect(a).toEqual([
      { type: "text", value: "a " },
      { type: "cite", index: 0 },
    ]);
    expect(b).toEqual([
      { type: "text", value: "b " },
      { type: "cite", index: 0 },
    ]);
  });
});

describe("parseAnswerWithBounds", () => {
  it("keeps in-range markers as cites", () => {
    expect(parseAnswerWithBounds("a [1] b [2]", 2)).toEqual([
      { type: "text", value: "a " },
      { type: "cite", index: 0 },
      { type: "text", value: " b " },
      { type: "cite", index: 1 },
    ]);
  });

  it("downgrades an out-of-range marker to plain text", () => {
    // Only 1 source, so [2] is out of range and stays literal.
    expect(parseAnswerWithBounds("a [1] b [2] c", 1)).toEqual([
      { type: "text", value: "a " },
      { type: "cite", index: 0 },
      { type: "text", value: " b [2] c" },
    ]);
  });

  it("downgrades and merges a leading out-of-range marker", () => {
    expect(parseAnswerWithBounds("[5] tail", 2)).toEqual([
      { type: "text", value: "[5] tail" },
    ]);
  });

  it("treats every marker as text when there are zero sources", () => {
    expect(parseAnswerWithBounds("a [1] b", 0)).toEqual([
      { type: "text", value: "a [1] b" },
    ]);
  });
});

describe("similarityFromScore", () => {
  it("returns null for null score", () => {
    expect(similarityFromScore(null)).toBeNull();
  });

  it("converts distance to similarity and rounds to 2 dp", () => {
    expect(similarityFromScore(0.17)).toBe(0.83);
  });

  it("maps distance 0 to similarity 1", () => {
    expect(similarityFromScore(0)).toBe(1);
  });

  it("maps distance 1 to similarity 0", () => {
    expect(similarityFromScore(1)).toBe(0);
  });

  it("rounds half up at the 2nd decimal", () => {
    // 1 - 0.125 = 0.875 -> 0.88
    expect(similarityFromScore(0.125)).toBe(0.88);
  });

  it("clamps a distance > 1 to similarity 0 (no negative)", () => {
    expect(similarityFromScore(1.4)).toBe(0);
  });

  it("clamps a negative distance to similarity 1", () => {
    expect(similarityFromScore(-0.2)).toBe(1);
  });
});

describe("confidenceQuartile", () => {
  it("returns 0 for null score", () => {
    expect(confidenceQuartile(null)).toBe(0);
  });

  it("returns 4 at the sim>=0.70 boundary (score 0.30)", () => {
    expect(confidenceQuartile(0.3)).toBe(4);
  });

  it("returns 4 well above the top boundary", () => {
    expect(confidenceQuartile(0.05)).toBe(4);
  });

  it("returns 3 at the sim>=0.55 boundary (score 0.45)", () => {
    expect(confidenceQuartile(0.45)).toBe(3);
  });

  it("returns 3 just below the top boundary (sim 0.69)", () => {
    expect(confidenceQuartile(0.31)).toBe(3);
  });

  it("returns 2 at the sim>=0.40 boundary (score 0.60)", () => {
    expect(confidenceQuartile(0.6)).toBe(2);
  });

  it("returns 2 just below the 0.55 boundary (sim 0.54)", () => {
    expect(confidenceQuartile(0.46)).toBe(2);
  });

  it("returns 1 for a present-but-low similarity (sim 0.39)", () => {
    expect(confidenceQuartile(0.61)).toBe(1);
  });

  it("returns 1 for very low similarity", () => {
    expect(confidenceQuartile(0.95)).toBe(1);
  });
});
