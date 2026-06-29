/*
 * In-memory sample AskResponse for local visual development WITHOUT a live
 * backend. This is ONLY used when `VITE_API_BASE` is unset (see App.tsx). The
 * default runtime path always calls the real `api.ask()`.
 */
import type { AskResponse } from "./types";

export const SAMPLE_ASK_RESPONSE: AskResponse = {
  answer:
    "The reporting deadline is the 15th of each month [1], and late " +
    "submissions incur a 2% penalty applied to the outstanding balance [2]. " +
    "Extensions may be granted in writing by the finance lead [3], though the " +
    "policy notes they are rarely approved after the first occurrence.",
  sources: [
    {
      id: "s1",
      file_name: "finance-policy-2024.pdf",
      page_number: 14,
      preview:
        "All monthly reports must be filed no later than the 15th calendar " +
        "day of the following month via the shared submission portal.",
      score: 0.17,
    },
    {
      id: "s2",
      file_name: "finance-policy-2024.pdf",
      page_number: 15,
      preview:
        "A penalty of two percent (2%) of the outstanding balance is applied " +
        "automatically to any submission received after the stated deadline.",
      score: 0.41,
    },
    {
      id: "s3",
      file_name: "ops-handbook.pdf",
      page_number: 7,
      preview:
        "Deadline extensions require written approval from the finance lead " +
        "and are recorded in the quarterly exceptions log.",
      score: 0.58,
    },
  ],
};
