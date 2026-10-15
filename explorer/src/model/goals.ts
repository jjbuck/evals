// The words for goals, used everywhere a goal is named. Metric names are shown exactly as the study declares them.

import { num } from "../format";
import type { Constraint, Objective } from "../record/schema";
import type { Goal, StudyModel } from "./study";

export function op(c: Constraint): string {
  return c.kind === "AtMost" ? "≤" : "≥";
}

export function arrow(o: Objective): string {
  return o.kind === "Maximize" ? "↑" : "↓";
}

/** "↑ answer" for an objective, "tokens ≤ 30,000" for a constraint. */
export function goalTitle(g: Goal): string {
  return g.role === "objective" ? `${arrow(g.objective)} ${g.metric}` : `${g.metric} ${op(g.constraint)} ${num(g.constraint.limit)}`;
}

/** "maximize answer" for an objective. */
export function objectiveText(o: Objective): string {
  return `${o.kind === "Maximize" ? "maximize" : "minimize"} ${o.metric}`;
}

export function limitText(c: Constraint): string {
  return `${op(c)} ${num(c.limit)}`;
}

/** Whether a value is better than another for this objective. */
export function better(o: Objective, a: number, b: number): boolean {
  return o.kind === "Maximize" ? a > b : a < b;
}

/** The objective a round served (with Independent), as its metric name. */
export function objectiveOf(study: StudyModel, k: number | null): string | null {
  return k === null || study.h.combine.copies <= 1 ? null : study.h.objectives[k]?.metric ?? null;
}
