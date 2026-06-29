import { useCallback, useMemo, useState } from "react";
import { AppShell } from "./components/AppShell/AppShell";
import { Button } from "./components/Button/Button";
import { ChatThread } from "./features/conversation/ChatThread";
import {
  nextTurnId,
  type ActiveSource,
  type ConversationTurn,
} from "./features/conversation/state";
import { SourceRail } from "./features/sources/SourceRail";
import { UploadPanel } from "./features/upload/UploadPanel";
import { StorePanel } from "./features/workspace/StorePanel";
import { useStore } from "./features/workspace/useStore";
import { readinessLabel } from "./features/workspace/summary";
import { askStream, type AskStreamHandlers } from "./lib/api";
import { SAMPLE_ASK_RESPONSE } from "./lib/sampleData";
import type { Source } from "./lib/types";
import styles from "./App.module.css";

type View = "chat" | "documents";

/**
 * Sample mode is an EXPLICIT opt-in (set VITE_USE_SAMPLE=true) for local visual
 * development without a backend: it SIMULATES streaming from the canned sample.
 * By default the app calls the real streaming api.askStream() — consistent with
 * api.ts, which already defaults the API base to http://localhost:8000. (The old
 * gate keyed on VITE_API_BASE being unset, which made answers fake even though
 * uploads were hitting the real backend — the "seeded answer" bug.)
 */
const USE_SAMPLE = import.meta.env.VITE_USE_SAMPLE === "true";

/** Split the sample answer into a handful of chunks for simulated streaming. */
function sampleChunks(answer: string, parts = 6): string[] {
  if (parts <= 1) return [answer];
  const size = Math.ceil(answer.length / parts);
  const chunks: string[] = [];
  for (let i = 0; i < answer.length; i += size) {
    chunks.push(answer.slice(i, i + size));
  }
  return chunks;
}

/**
 * Dev simulation of askStream: emit the SAMPLE answer in chunks over time so
 * local dev shows progressive rendering, then deliver sources and finish.
 * Mirrors askStream's contract (tokens first, sources near the end).
 */
function simulateStream(
  handlers: Pick<AskStreamHandlers, "onToken" | "onSources" | "onError">,
): Promise<void> {
  return new Promise((resolve) => {
    const chunks = sampleChunks(SAMPLE_ASK_RESPONSE.answer);
    let i = 0;
    const tick = () => {
      if (i < chunks.length) {
        handlers.onToken(chunks[i]);
        i += 1;
        setTimeout(tick, 120);
        return;
      }
      // Sources arrive only after all tokens (matching the real backend).
      handlers.onSources(SAMPLE_ASK_RESPONSE.sources as Source[]);
      resolve();
    };
    setTimeout(tick, 200);
  });
}

export default function App() {
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [busy, setBusy] = useState(false);
  // The active grounding link drives the rail + marker/slip sync. `pinned` is
  // the turn whose sources the rail shows (set by clicking a marker, defaults
  // to the latest assistant turn). `sourceIndex` is the transient highlight
  // from hover/focus.
  const [active, setActive] = useState<ActiveSource | null>(null);
  // Which surface fills the main column. The chat stays mounted (just hidden)
  // when the documents panel is open so the conversation is never reset.
  const [view, setView] = useState<View>("chat");

  // The live persisted document store (across sessions). Fetched on mount;
  // refreshed after an upload batch and immediately after a clear.
  const store = useStore();

  const updateTurn = useCallback(
    (id: string, patch: Partial<ConversationTurn>) => {
      setTurns((prev) =>
        prev.map((t) => (t.id === id ? { ...t, ...patch } : t)),
      );
    },
    [],
  );

  const handleSubmit = useCallback(
    async (question: string) => {
      const userTurn: ConversationTurn = {
        id: nextTurnId("user"),
        role: "user",
        content: question,
        sources: [],
        status: "complete",
      };
      const assistantId = nextTurnId("assistant");
      const pendingTurn: ConversationTurn = {
        id: assistantId,
        role: "assistant",
        content: "",
        sources: [],
        status: "pending",
      };

      setTurns((prev) => [...prev, userTurn, pendingTurn]);
      setBusy(true);

      // Track whether an error was reported so completion doesn't override it.
      let errored = false;

      // Append a token delta to this turn's content, flipping it from "pending"
      // to "streaming" so the answer renders progressively with a caret.
      const onToken = (text: string) => {
        setTurns((prev) =>
          prev.map((t) =>
            t.id === assistantId
              ? { ...t, content: t.content + text, status: "streaming" }
              : t,
          ),
        );
      };

      // Sources arrive once near the end: attach them and make this turn the
      // active source set so margin slips populate AFTER the stream.
      const onSources = (sources: Source[]) => {
        updateTurn(assistantId, { sources });
        setActive({ turnId: assistantId, sourceIndex: null });
      };

      const onError = (message: string) => {
        errored = true;
        updateTurn(assistantId, { status: "error", error: message });
      };

      try {
        if (USE_SAMPLE) {
          await simulateStream({ onToken, onSources, onError });
        } else {
          await askStream(question, { onToken, onSources, onError });
        }
      } catch (err) {
        // askStream resolves on failure (it routes errors through onError), so
        // this only guards against an unexpected throw.
        onError(err instanceof Error ? err.message : "Unable to reach the backend.");
      } finally {
        if (!errored) {
          updateTurn(assistantId, { status: "complete" });
          // The latest assistant turn becomes the active source set even when
          // it has no citations (mirrors the pre-streaming behaviour); if a
          // sources event already ran this is a harmless no-op.
          setActive((prev) =>
            prev?.turnId === assistantId
              ? prev
              : { turnId: assistantId, sourceIndex: null },
          );
        }
        setBusy(false);
      }
    },
    [updateTurn],
  );

  // Hover/focus a marker -> highlight its slip (pin that turn's set + index).
  const handleMarkerActivate = useCallback(
    (turnId: string, index: number) => {
      setActive({ turnId, sourceIndex: index });
    },
    [],
  );

  // Hover-out / blur -> keep the active turn but clear the transient highlight.
  const handleMarkerDeactivate = useCallback(() => {
    setActive((prev) => (prev ? { ...prev, sourceIndex: null } : prev));
  }, []);

  // Click a marker -> make that turn's sources the active rail set and select.
  const handleMarkerSelect = useCallback((turnId: string, index: number) => {
    setActive({ turnId, sourceIndex: index });
  }, []);

  // Hover/focus a slip in the rail -> highlight its marker(s) in the answer.
  const handleRailIndexChange = useCallback(
    (index: number | null) => {
      setActive((prev) => (prev ? { ...prev, sourceIndex: index } : prev));
    },
    [],
  );

  // The rail shows the active turn's sources (defaults to latest assistant).
  const activeSources = useMemo(() => {
    if (!active) return [];
    const turn = turns.find((t) => t.id === active.turnId);
    return turn?.sources ?? [];
  }, [active, turns]);

  const documentsOpen = view === "documents";

  // Quiet header status: surface store readiness once it has loaded; fall back
  // to a neutral "ready" while it is still loading or could not load.
  const headerStatus =
    store.state !== null ? readinessLabel(store.state) : "ready";

  return (
    <AppShell
      status={headerStatus}
      headerActions={
        <Button
          variant="default"
          aria-pressed={documentsOpen}
          onClick={() => setView(documentsOpen ? "chat" : "documents")}
        >
          {documentsOpen ? "Back to chat" : "Add documents"}
        </Button>
      }
      rail={
        <SourceRail
          sources={activeSources}
          activeIndex={active?.sourceIndex ?? null}
          onActiveIndexChange={handleRailIndexChange}
        />
      }
    >
      {/* Keep the chat mounted (hidden) while documents is open so the
          conversation is never reset when toggling views. */}
      <div className={styles.chatHost} hidden={documentsOpen}>
        <ChatThread
          turns={turns}
          active={active}
          busy={busy}
          onSubmit={handleSubmit}
          onMarkerActivate={handleMarkerActivate}
          onMarkerDeactivate={handleMarkerDeactivate}
          onMarkerSelect={handleMarkerSelect}
        />
      </div>
      {documentsOpen ? (
        <div className={styles.documentsHost}>
          <UploadPanel
            onClose={() => setView("chat")}
            onBatchComplete={store.refresh}
          />
          <StorePanel
            state={store.state}
            phase={store.phase}
            error={store.error}
            clearing={store.clearing}
            onClear={store.clear}
          />
        </div>
      ) : null}
    </AppShell>
  );
}
