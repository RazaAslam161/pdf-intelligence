import { AppShell } from "./components/AppShell/AppShell";
import { ConversationEmptyState } from "./features/conversation/ConversationEmptyState";
import { emptyConversation } from "./features/conversation/state";
import { SourceRail } from "./features/sources/SourceRail";

export default function App() {
  // Empty conversation state — no data fetching on mount. Later milestones will
  // populate turns and drive the rail from cited sources.
  const conversation = emptyConversation;
  const hasTurns = conversation.turns.length > 0;

  return (
    <AppShell status="ready" rail={<SourceRail />}>
      {hasTurns ? null : <ConversationEmptyState />}
    </AppShell>
  );
}
