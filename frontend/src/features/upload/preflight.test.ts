import { describe, expect, it } from "vitest";
import type { ConfigResponse } from "../../lib/types";
import {
  BYTES_PER_MB,
  applyFileCountCap,
  formatBytes,
  isPdfName,
  validateFiles,
} from "./preflight";

const config: ConfigResponse = {
  max_upload_mb: 25,
  max_pages_per_pdf: 200,
  max_files_per_run: 3,
};

describe("isPdfName", () => {
  it("accepts a lowercase .pdf", () => {
    expect(isPdfName("report.pdf")).toBe(true);
  });

  it("accepts an uppercase .PDF (case-insensitive)", () => {
    expect(isPdfName("REPORT.PDF")).toBe(true);
  });

  it("accepts mixed-case .Pdf", () => {
    expect(isPdfName("Report.Pdf")).toBe(true);
  });

  it("ignores surrounding whitespace", () => {
    expect(isPdfName("  spaced.pdf  ")).toBe(true);
  });

  it("rejects a non-pdf extension", () => {
    expect(isPdfName("notes.txt")).toBe(false);
  });

  it("rejects a name with no extension", () => {
    expect(isPdfName("README")).toBe(false);
  });

  it("rejects pdf appearing only mid-name", () => {
    expect(isPdfName("pdf.docx")).toBe(false);
  });

  it("rejects an empty name", () => {
    expect(isPdfName("")).toBe(false);
  });
});

describe("validateFiles", () => {
  it("returns an empty array for no files", () => {
    expect(validateFiles([], config)).toEqual([]);
  });

  it("marks an ordinary in-spec pdf as ok", () => {
    expect(validateFiles([{ name: "a.pdf", size: 1000 }], config)).toEqual([
      { name: "a.pdf", size: 1000, status: "ok" },
    ]);
  });

  it("marks a 0-byte pdf as ok (server decides if truly empty)", () => {
    expect(validateFiles([{ name: "empty.pdf", size: 0 }], config)).toEqual([
      { name: "empty.pdf", size: 0, status: "ok" },
    ]);
  });

  it("marks a non-pdf as unsupported", () => {
    expect(validateFiles([{ name: "notes.txt", size: 10 }], config)).toEqual([
      { name: "notes.txt", size: 10, status: "unsupported" },
    ]);
  });

  it("checks extension before size (a huge .txt is unsupported, not too_large)", () => {
    const huge = 100 * BYTES_PER_MB;
    expect(validateFiles([{ name: "big.txt", size: huge }], config)).toEqual([
      { name: "big.txt", size: huge, status: "unsupported" },
    ]);
  });

  it("accepts a pdf exactly at the size cap (boundary, inclusive)", () => {
    const atCap = 25 * BYTES_PER_MB;
    expect(validateFiles([{ name: "edge.pdf", size: atCap }], config)).toEqual([
      { name: "edge.pdf", size: atCap, status: "ok" },
    ]);
  });

  it("rejects a pdf one byte over the cap as too_large", () => {
    const overCap = 25 * BYTES_PER_MB + 1;
    expect(
      validateFiles([{ name: "over.pdf", size: overCap }], config),
    ).toEqual([{ name: "over.pdf", size: overCap, status: "too_large" }]);
  });

  it("categorizes a mixed batch independently and preserves order", () => {
    const files = [
      { name: "ok.pdf", size: 5 * BYTES_PER_MB },
      { name: "huge.pdf", size: 40 * BYTES_PER_MB },
      { name: "image.png", size: 2 },
      { name: "empty.pdf", size: 0 },
    ];
    expect(validateFiles(files, config)).toEqual([
      { name: "ok.pdf", size: 5 * BYTES_PER_MB, status: "ok" },
      { name: "huge.pdf", size: 40 * BYTES_PER_MB, status: "too_large" },
      { name: "image.png", size: 2, status: "unsupported" },
      { name: "empty.pdf", size: 0, status: "ok" },
    ]);
  });
});

describe("applyFileCountCap", () => {
  it("returns everything when count is below the cap", () => {
    const files = ["a", "b"];
    expect(applyFileCountCap(files, config)).toEqual({
      accepted: ["a", "b"],
      skipped: [],
    });
  });

  it("returns everything when count is exactly at the cap (boundary)", () => {
    const files = ["a", "b", "c"];
    expect(applyFileCountCap(files, config)).toEqual({
      accepted: ["a", "b", "c"],
      skipped: [],
    });
  });

  it("splits the overflow when count exceeds the cap", () => {
    const files = ["a", "b", "c", "d", "e"];
    expect(applyFileCountCap(files, config)).toEqual({
      accepted: ["a", "b", "c"],
      skipped: ["d", "e"],
    });
  });

  it("keeps the FIRST N (queue order is preserved)", () => {
    const files = [1, 2, 3, 4];
    const { accepted, skipped } = applyFileCountCap(files, config);
    expect(accepted).toEqual([1, 2, 3]);
    expect(skipped).toEqual([4]);
  });

  it("handles an empty list", () => {
    expect(applyFileCountCap([], config)).toEqual({
      accepted: [],
      skipped: [],
    });
  });

  it("skips everything when the cap is zero", () => {
    const zeroCap: ConfigResponse = { ...config, max_files_per_run: 0 };
    expect(applyFileCountCap(["a", "b"], zeroCap)).toEqual({
      accepted: [],
      skipped: ["a", "b"],
    });
  });
});

describe("formatBytes", () => {
  it("formats bytes under 1KB with a B suffix", () => {
    expect(formatBytes(512)).toBe("512 B");
  });

  it("formats 0 bytes", () => {
    expect(formatBytes(0)).toBe("0 B");
  });

  it("formats kilobytes to one decimal", () => {
    expect(formatBytes(2048)).toBe("2.0 KB");
  });

  it("formats megabytes to one decimal", () => {
    expect(formatBytes(Math.round(1.5 * BYTES_PER_MB))).toBe("1.5 MB");
  });

  it("formats an over-cap size for the user message", () => {
    expect(formatBytes(Math.round(31.2 * BYTES_PER_MB))).toBe("31.2 MB");
  });
});
