/*
 * Conversation state shapes for the chat thread and the source rail.
 *
 * A "turn" is one message in the thread. User turns carry only text; assistant
 * turns additionally carry grounding sources and a status (the answer is
 * requested asynchronously, so a turn can be pending or errored before it has
 * content).
 */
import type { Source } from "../../lib/types";

export type TurnRole = "user" | "assistant";

/** Lifecycle of an assistant turn while its answer is being fetched. */
export type TurnStatus = "pending" | "complete" | "error";

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

/**
 * The currently active grounding link: which assistant turn's sources are shown
 * in the rail, and which single source (if any) is highlighted on both ends of
 * the marker <-> slip sync. `sourceIndex` is a zero-based index into that turn's
 * sources, or null when a whole turn is active but no individual source is
 * focused/hovered.
 */
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
