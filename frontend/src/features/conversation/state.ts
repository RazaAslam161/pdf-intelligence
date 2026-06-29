/*
 * Empty conversation state. Message and turn logic arrive in a later milestone;
 * this defines the shape and an empty initial value so the shell can render a
 * grounded "no conversation yet" view without any chat machinery.
 */
import type { Source } from "../../lib/types";

export type TurnRole = "user" | "assistant";

export interface ConversationTurn {
  id: string;
  role: TurnRole;
  content: string;
  /** Grounding sources for assistant turns; empty for user turns. */
  sources: Source[];
}

export interface ConversationState {
  turns: ConversationTurn[];
}

/** The initial, empty conversation — no turns. */
export const emptyConversation: ConversationState = {
  turns: [],
};
