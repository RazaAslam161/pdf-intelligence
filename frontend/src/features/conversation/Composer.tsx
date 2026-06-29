import { useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { Button } from "../../components/Button/Button";
import styles from "./Composer.module.css";

export interface ComposerProps {
  /** Submit a question. The parent owns the request lifecycle. */
  onSubmit: (question: string) => void;
  /** Disable input + send while a request is in flight. */
  disabled: boolean;
}

/**
 * Multiline question composer. Enter sends; Shift+Enter inserts a newline. The
 * textarea auto-grows up to a cap. Empty / whitespace-only input cannot be
 * sent. Disabled while a request is in flight.
 */
export function Composer({ onSubmit, disabled }: ComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSend = value.trim().length > 0 && !disabled;

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
    // Reset auto-grow height after clearing.
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function handleInput() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  return (
    <form
      className={styles.composer}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={textareaRef}
        className={styles.input}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        placeholder="Ask a question about your documents…"
        rows={1}
        disabled={disabled}
        aria-label="Ask a question"
      />
      <Button
        type="submit"
        variant="default"
        disabled={!canSend}
        aria-label="Send question"
      >
        Ask
      </Button>
    </form>
  );
}
