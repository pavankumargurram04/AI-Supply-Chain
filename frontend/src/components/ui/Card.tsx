import clsx from "clsx";
import type { ReactNode } from "react";

interface Props {
  title?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Card({ title, actions, className, children }: Props) {
  return (
    <div className={clsx("card", className)}>
      {(title || actions) && (
        <div className="card-header">
          <div className="card-title">{title}</div>
          <div>{actions}</div>
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  );
}