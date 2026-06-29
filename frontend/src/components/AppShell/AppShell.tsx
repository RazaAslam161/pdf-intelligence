import type { ReactNode } from "react";
import { ThemeToggle } from "../ThemeToggle/ThemeToggle";
import styles from "./AppShell.module.css";

export interface AppShellProps {
  /** Primary conversation column content. */
  children: ReactNode;
  /** Right-hand source rail content. */
  rail: ReactNode;
  /** Optional quiet status indicator rendered in the header. */
  status?: ReactNode;
  /** Optional header action(s) (e.g. an "Add documents" toggle), rendered
   *  before the theme toggle. */
  headerActions?: ReactNode;
}

/**
 * Top-level layout: an instrument header over a split body — a primary
 * conversation column (flex:1) and a right "source rail" <aside>. Below 900px
 * the rail stacks under the main column. Uses semantic landmarks throughout.
 */
export function AppShell({
  children,
  rail,
  status,
  headerActions,
}: AppShellProps) {
  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.wordmark}>PDF Intelligence</div>
        <div className={styles.headerControls}>
          {status ? <div className={styles.status}>{status}</div> : null}
          {headerActions}
          <ThemeToggle />
        </div>
      </header>

      <div className={styles.body}>
        <main className={styles.main}>{children}</main>
        <aside
          className={styles.rail}
          role="complementary"
          aria-label="Sources"
        >
          {rail}
        </aside>
      </div>
    </div>
  );
}
