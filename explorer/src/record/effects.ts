// effects.json, written next to a study's record by strands_evals/optimization/effects.py: the effect of each choice
// on each goal metric, from an effects model fitted to the record.

import { useEffect, useState } from "react";
import type { StudyRef } from "../source";

/** ``rewrites``: a text variable by how many rewrites deep it is (default, 1, 2, 3+), not by its words. */
export type EffectsFactor = { key: string; kind: "choice" | "range" | "rewrites"; reference: string; levels: string[] };
export type Effect = { key: string; level: string; mean: number; ci90: [number, number]; p_positive: number };
export type EffectsDiagnostics = { n_holdout: number; z_mean: number; z_sq_sum: number; flagged: boolean; messages: string[] };
export type MetricEffects = {
  scale: { kind: "range"; low: number; high: number } | { kind: "baseline"; unit: number };
  observations: number;
  effects: Effect[];
  diagnostics: EffectsDiagnostics;
};
export type EffectsGoal = { role: "objective" | "constraint"; index: number; metric: string; better: "higher" | "lower" };
export type Effects = {
  schema: 1;
  computed_at: string;
  record_events: number;
  method: string;
  factors: EffectsFactor[];
  omitted: { key: string; why: string }[];
  goals: EffectsGoal[];
  metrics: Record<string, MetricEffects>;
  messages: string[];
};

/** The study's effects.json, or null when it has none (or one of another schema). */
export function useEffects(ref: StudyRef | null, events: number): Effects | null {
  const [state, setState] = useState<Effects | null>(null);
  useEffect(() => {
    setState(null);
    if (!ref) return;
    ref.read("effects.json").then((text) => {
      if (text === null) return;
      try {
        const e = JSON.parse(text) as Effects;
        if (e.schema === 1 && e.metrics) setState(e);
      } catch {
        // not an effects file (a dev server's fallback page): no effects
      }
    });
  }, [ref, events]);
  return state;
}

/** The probability that a level is better than the reference for a goal. */
export function pBetter(e: Effect, better: "higher" | "lower"): number {
  return better === "higher" ? e.p_positive : 1 - e.p_positive;
}
