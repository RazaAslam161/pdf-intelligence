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

// Opt-in (VITE_USE_SAMPLE=true) to simulate streaming from a canned sample for
// local dev without a backend. Otherwise we hit the real askStream().
const USE_SAMPLE = import.meta.env.VITE_USE_SAMPLE === "true";

// Split the sample answer into a few chunks for simulated streaming.
function sampleChunks(answer: string, parts = 6): string[] {
  if (parts <= 1) return [answer];
  const size = Math.ceil(answer.length / parts);
  const chunks: string[] = [];
  for (let i = 0; i < answer.length; i += size) {
    chunks.push(answer.slice(i, i + size));
  }
  return chunks;
}

// Dev-only stand-in for askStream: emit the sample answer in timed chunks, then
// deliver sources at the end (tokens first, sources last, like the real thing).
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
      // Sources only after all tokens, matching the real backend.
      handlers.onSources(SAMPLE_ASK_RESPONSE.sources as Source[]);
      resolve();
    };
    setTimeout(tick, 200);
  });
}

export default function App() {
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [busy, setBusy] = useState(false);
  // Drives the rail and marker/slip sync. turnId picks whose sources the rail
  // shows; sourceIndex is the transient hover/focus highlight.
  const [active, setActive] = useState<ActiveSource | null>(null);
  // Which surface fills the main column.
  const [view, setView] = useState<View>("chat");

  // Persisted document store; refreshed after an upload batch and after a clear.
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

      // So completion doesn't override a reported error.
      let errored = false;

      // Append a token delta and flip the turn into "streaming".
      const onToken = (text: string) => {
        setTurns((prev) =>
          prev.map((t) =>
            t.id === assistantId
              ? { ...t, content: t.content + text, status: "streaming" }
              : t,
          ),
        );
      };

      // Attach sources and make this turn the active set once they arrive.
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
        // askStream routes its own errors through onError, so this only catches
        // an unexpected throw.
        onError(err instanceof Error ? err.message : "Unable to reach the backend.");
      } finally {
        if (!errored) {
          updateTurn(assistantId, { status: "complete" });
          // Make the latest assistant turn active even with no citations; a
          // no-op if a sources event already did so.
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

  // Hover/focus a marker: highlight its slip.
  const handleMarkerActivate = useCallback(
    (turnId: string, index: number) => {
      setActive({ turnId, sourceIndex: index });
    },
    [],
  );

  // Hover-out/blur: keep the active turn, clear the transient highlight.
  const handleMarkerDeactivate = useCallback(() => {
    setActive((prev) => (prev ? { ...prev, sourceIndex: null } : prev));
  }, []);

  // Click a marker: make that turn's sources the active rail set.
  const handleMarkerSelect = useCallback((turnId: string, index: number) => {
    setActive({ turnId, sourceIndex: index });
  }, []);

  // Hover/focus a rail slip: highlight its marker(s) in the answer.
  const handleRailIndexChange = useCallback(
    (index: number | null) => {
      setActive((prev) => (prev ? { ...prev, sourceIndex: index } : prev));
    },
    [],
  );

  // The rail shows the active turn's sources.
  const activeSources = useMemo(() => {
    if (!active) return [];
    const turn = turns.find((t) => t.id === active.turnId);
    return turn?.sources ?? [];
  }, [active, turns]);

  const documentsOpen = view === "documents";

  // Show store readiness once loaded, else a neutral "ready".
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
      {/* Keep the chat mounted while documents is open so toggling views
          never resets the conversation. */}
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
