import { describe, expect, it } from "vitest";
import type { StoreState } from "../../lib/types";
import {
  documentMeta,
  pluralize,
  readinessLabel,
  readinessSummary,
} from "./summary";

const emptyState: StoreState = {
  documents: [],
  total_chunks: 0,
  ready: false,
};

const readyState: StoreState = {
  documents: [
    { document_id: "a", file_name: "a.pdf", page_count: 10, chunk_count: 30 },
    { document_id: "b", file_name: "b.pdf", page_count: 4, chunk_count: 18 },
  ],
  total_chunks: 48,
  ready: true,
};

describe("pluralize", () => {
  it("uses the singular for a count of 1", () => {
    expect(pluralize("document", 1)).toBe("1 document");
  });

  it("uses the plural for 0", () => {
    expect(pluralize("document", 0)).toBe("0 documents");
  });

  it("uses the plural for many", () => {
    expect(pluralize("chunk", 48)).toBe("48 chunks");
  });
});

describe("readinessSummary", () => {
  it("returns the calm empty-state sentence when not ready", () => {
    expect(readinessSummary(emptyState)).toBe(
      "No documents indexed yet — add a PDF to start asking questions.",
    );
  });

  it("treats zero chunks as empty even if a document is listed", () => {
    const oddState: StoreState = {
      documents: [
        { document_id: "x", file_name: "x.pdf", page_count: 2, chunk_count: 0 },
      ],
      total_chunks: 0,
      ready: false,
    };
    expect(readinessSummary(oddState)).toBe(
      "No documents indexed yet — add a PDF to start asking questions.",
    );
  });

  it("summarizes a ready store with document and chunk counts", () => {
    expect(readinessSummary(readyState)).toBe(
      "2 documents · 48 chunks ready to query.",
    );
  });

  it("uses singular nouns for a single document and single chunk", () => {
    const oneState: StoreState = {
      documents: [
        { document_id: "a", file_name: "a.pdf", page_count: 1, chunk_count: 1 },
      ],
      total_chunks: 1,
      ready: true,
    };
    expect(readinessSummary(oneState)).toBe("1 document · 1 chunk ready to query.");
  });

  it("defends against ready:true but zero chunks (reads as empty)", () => {
    const inconsistent: StoreState = {
      documents: [],
      total_chunks: 0,
      ready: true,
    };
    expect(readinessSummary(inconsistent)).toBe(
      "No documents indexed yet — add a PDF to start asking questions.",
    );
  });
});

describe("readinessLabel", () => {
  it("reads 'No documents' when empty", () => {
    expect(readinessLabel(emptyState)).toBe("No documents");
  });

  it("reads 'N documents indexed' when ready", () => {
    expect(readinessLabel(readyState)).toBe("2 documents indexed");
  });

  it("uses the singular for a single document", () => {
    const oneState: StoreState = {
      documents: [
        { document_id: "a", file_name: "a.pdf", page_count: 1, chunk_count: 1 },
      ],
      total_chunks: 1,
      ready: true,
    };
    expect(readinessLabel(oneState)).toBe("1 document indexed");
  });
});

describe("documentMeta", () => {
  it("formats pages and chunks", () => {
    expect(documentMeta(12, 48)).toBe("12 pages · 48 chunks");
  });

  it("uses singular nouns for ones", () => {
    expect(documentMeta(1, 1)).toBe("1 page · 1 chunk");
  });

  it("formats zeros as plural", () => {
    expect(documentMeta(0, 0)).toBe("0 pages · 0 chunks");
  });
});
