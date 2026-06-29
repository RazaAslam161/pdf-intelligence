import type { ButtonHTMLAttributes, ReactNode } from "react";
import styles from "./IconButton.module.css";

export interface IconButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Required: icon-only buttons must have an accessible label. */
  "aria-label": string;
  children: ReactNode;
}

export function IconButton({
  className,
  type = "button",
  children,
  ...rest
}: IconButtonProps) {
  const classes = [styles.iconButton, className].filter(Boolean).join(" ");

  return (
    // eslint-disable-next-line react/button-has-type
    <button type={type} className={classes} {...rest}>
      {children}
    </button>
  );
}
