import type { ReactNode } from "react";
import { ThemeToggle } from "../ThemeToggle/ThemeToggle";
import styles from "./AppShell.module.css";

export interface AppShellProps {
  children: ReactNode;
  rail: ReactNode;
  status?: ReactNode;
  /** Header action(s) rendered before the theme toggle. */
  headerActions?: ReactNode;
}

// Header over a split body: conversation column plus a source rail that stacks
// underneath below 900px.
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
