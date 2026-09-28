import clsx from "clsx";
import type { ReactNode } from "react";

type Tone = "default" | "info" | "success" | "warning" | "danger";

export function Badge({ children, tone = "default" }: { children: ReactNode; tone?: Tone }) {
  return <span className={clsx("badge", `badge-${tone}`)}>{children}</span>;
}