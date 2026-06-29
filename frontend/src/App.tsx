import { useCallback, useMemo, useState } from "react";
import { AppShell } from "./components/AppShell/AppShell";
import { ChatThread } from "./features/conversation/ChatThread";
import {
  nextTurnId,
  type ActiveSource,
  type ConversationTurn,
} from "./features/conversation/state";
import { SourceRail } from "./features/sources/SourceRail";
import { ApiError, ask } from "./lib/api";
import { SAMPLE_ASK_RESPONSE } from "./lib/sampleData";
import type { AskResponse } from "./lib/types";

/**
 * Dev-only fallback: when VITE_API_BASE is unset we have no backend to talk to,
 * so return a canned response for local visual development. The DEFAULT path
 * (VITE_API_BASE set) always calls the real api.ask().
 */
const USE_SAMPLE = import.meta.env.VITE_API_BASE === undefined;

async function fetchAnswer(question: string): Promise<AskResponse> {
  if (USE_SAMPLE) {
    await new Promise((resolve) => setTimeout(resolve, 600));
    return SAMPLE_ASK_RESPONSE;
  }
  return ask(question);
}

export default function App() {
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [busy, setBusy] = useState(false);
  // The active grounding link drives the rail + marker/slip sync. `pinned` is
  // the turn whose sources the rail shows (set by clicking a marker, defaults
  // to the latest assistant turn). `sourceIndex` is the transient highlight
  // from hover/focus.
  const [active, setActive] = useState<ActiveSource | null>(null);

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

      try {
        const res = await fetchAnswer(question);
        updateTurn(assistantId, {
          content: res.answer,
          sources: res.sources,
          status: "complete",
        });
        // The latest assistant turn becomes the active source set.
        setActive({ turnId: assistantId, sourceIndex: null });
      } catch (err) {
        const message =
          err instanceof ApiError
            ? `Request failed (${err.status}). ${err.message}`
            : err instanceof Error
              ? err.message
              : "Unable to reach the backend.";
        updateTurn(assistantId, { status: "error", error: message });
      } finally {
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

  return (
    <AppShell status="ready" rail={
      <SourceRail
        sources={activeSources}
        activeIndex={active?.sourceIndex ?? null}
        onActiveIndexChange={handleRailIndexChange}
      />
    }>
      <ChatThread
        turns={turns}
        active={active}
        busy={busy}
        onSubmit={handleSubmit}
        onMarkerActivate={handleMarkerActivate}
        onMarkerDeactivate={handleMarkerDeactivate}
        onMarkerSelect={handleMarkerSelect}
      />
    </AppShell>
  );
}
