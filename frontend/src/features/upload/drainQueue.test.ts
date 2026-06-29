import { describe, expect, it } from "vitest";
import { drainUploadQueue, type UploadFile } from "./useUpload";
import type { IndexFileResult } from "../../lib/types";

function row(id: string, withFile = true): UploadFile {
  return {
    id,
    name: `${id}.pdf`,
    size: 10,
    status: "queued",
    file: withFile ? new File(["%PDF-"], `${id}.pdf`) : undefined,
  };
}

function indexed(name: string): IndexFileResult {
  return { file_name: name, page_count: 2, chunk_count: 5, status: "indexed" };
}

describe("drainUploadQueue", () => {
  it("actually uploads every queued file (the regression)", async () => {
    const uploaded: string[] = [];
    const patches: Array<{ id: string; status?: string }> = [];
    const queue = [row("a"), row("b"), row("c")];

    const processed = await drainUploadQueue(queue, {
      upload: async (file) => {
        uploaded.push(file.name);
        return indexed(file.name);
      },
      patch: (id, patch) => patches.push({ id, status: patch.status }),
    });

    expect(processed).toBe(true);
    // every file must reach the uploader
    expect(uploaded).toEqual(["a.pdf", "b.pdf", "c.pdf"]);
    expect(queue.length).toBe(0);
    expect(patches.filter((p) => p.status === "indexing").length).toBe(3);
    expect(patches.filter((p) => p.status === "indexed").length).toBe(3);
  });

  it("isolates a failing file and keeps draining the rest", async () => {
    const queue = [row("ok1"), row("bad"), row("ok2")];
    const final: Record<string, string> = {};

    await drainUploadQueue(queue, {
      upload: async (file) => {
        if (file.name === "bad.pdf") throw new Error("boom");
        return indexed(file.name);
      },
      patch: (id, patch) => {
        if (patch.status && patch.status !== "indexing") final[id] = patch.status;
      },
    });

    expect(final).toEqual({ ok1: "indexed", bad: "failed", ok2: "indexed" });
  });

  it("maps server statuses (rejected / unsupported / failed) onto rows", async () => {
    const queue = [row("r"), row("u"), row("f")];
    const final: Record<string, string> = {};
    const byName: Record<string, IndexFileResult["status"]> = {
      "r.pdf": "rejected",
      "u.pdf": "unsupported",
      "f.pdf": "weird-unknown",
    };

    await drainUploadQueue(queue, {
      upload: async (file) => ({
        file_name: file.name,
        page_count: 0,
        chunk_count: 0,
        status: byName[file.name],
      }),
      patch: (id, patch) => {
        if (patch.status && patch.status !== "indexing") final[id] = patch.status;
      },
    });

    expect(final).toEqual({ r: "rejected", u: "unsupported", f: "failed" });
  });
});
