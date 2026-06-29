import type { ButtonHTMLAttributes, ReactNode } from "react";
import styles from "./Button.module.css";

type ButtonVariant = "default" | "primary";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /**
   * "default" is an outline/ghost button (transparent surface, hairline
   * border). "primary" uses the accent color — accent is RESERVED for grounding
   * / evidence actions, so only use "primary" for those (e.g. "Ask",
   * "View source"), never for generic CTAs.
   */
  variant?: ButtonVariant;
  children: ReactNode;
}

export function Button({
  variant = "default",
  className,
  type = "button",
  children,
  ...rest
}: ButtonProps) {
  const classes = [styles.button, styles[variant], className]
    .filter(Boolean)
    .join(" ");

  return (
    // eslint-disable-next-line react/button-has-type
    <button type={type} className={classes} {...rest}>
      {children}
    </button>
  );
}
