// Conversation state for the chat thread and source rail. A turn is one message;
// assistant turns also carry sources and a status since answers arrive async.
import type { Source } from "../../lib/types";

export type TurnRole = "user" | "assistant";

// Lifecycle of an assistant turn: pending (no tokens yet), streaming (answer
// rendering), complete (answer + sources final), or error.
export type TurnStatus = "pending" | "streaming" | "complete" | "error";

export interface ConversationTurn {
  id: string;
  role: TurnRole;
  content: string;
  /** Grounding sources for assistant turns; empty for user turns. */
  sources: Source[];
  /** Only meaningful for assistant turns. User turns are always "complete". */
  status: TurnStatus;
  /** Inline error message for a failed assistant turn. */
  error?: string;
}

export interface ConversationState {
  turns: ConversationTurn[];
}

/** The initial, empty conversation — no turns. */
export const emptyConversation: ConversationState = {
  turns: [],
};

// The active grounding link: which turn's sources show in the rail, and which
// source is highlighted. sourceIndex is null when no individual source is focused.
export interface ActiveSource {
  turnId: string;
  sourceIndex: number | null;
}

let turnCounter = 0;

/** Generate a stable, unique turn id. */
export function nextTurnId(prefix: TurnRole): string {
  turnCounter += 1;
  return `${prefix}-${turnCounter}`;
}
