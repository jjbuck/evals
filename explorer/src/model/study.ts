// A study as the views read it: the record's facts, indexed, plus the few things that are pure functions of them
// (the schedule tree from the scope events, Pareto history from best_after, text versions). Nothing here is guessed,
// colored or worded: encodings live in encoding/, words in model/goals.ts.

import type { Lifecycle, StudyRecord } from "../record/parse";
import type {
  Constraint,
  Estimate,
  Objective,
  ScheduleNode,
  ScopeEndEvent,
  ScopeEvent,
  SpaceRow,
  StudyEvent,
  ToldEvent,
  CandidateEvent,
} from "../record/schema";

export const SELECTORS = new Set(["one_of", "switch"]);

/** One goal of the study: an objective or a constraint, by its position in the header. */
export type Goal =
  | { role: "objective"; index: number; metric: string; objective: Objective }
  | { role: "constraint"; index: number; metric: string; constraint: Constraint };

/** How a candidate is drawn: comparable and feasible, comparable but breaking a constraint, not comparable (some case
 * lacks a goal metric), or nothing measured (invalid or failed). */
export type MarkKind = "ok" | "infeasible" | "ineligible" | "unmeasured";

/** One scope of the schedule: a run of a schedule (the root of an optimize() call) or a round inside one. */
export type Scope = {
  id: number;
  event: ScopeEvent;
  end: ScopeEndEvent | null; // null while it is open
  parent: Scope | null;
  depth: number; // 0 for a run's root scope, 1 for its rounds, 2 for rounds inside those, ...
  children: Scope[];
  candidates: number[]; // every candidate in it, its children's included, in order
  told: ToldEvent | null; // what its outer proposal was told (a round of Nested), when it has been
};

/** The rounds of one level of the schedule: every round a given schedule node opened (one Nested level), by depth. */
export type Level = { depth: number; node: string; outer: string | null; scopes: Scope[] };

export type Choice = { key: string; option: string };

export class StudyModel {
  readonly h: StudyEvent;
  readonly candidates: CandidateEvent[];
  readonly byNumber: Map<number, CandidateEvent>;
  readonly space: SpaceRow[];
  readonly spaceByKey: Map<string, SpaceRow>;
  readonly blockOf: Map<string, string>;
  readonly blockOrder: string[];
  readonly goals: Goal[];
  readonly objectives: Extract<Goal, { role: "objective" }>[];
  readonly constraints: Extract<Goal, { role: "constraint" }>[];
  readonly scopes: Map<number, Scope>;
  readonly roots: Scope[];
  readonly levels: Level[];
  readonly scheduleNodes: Map<string, ScheduleNode>;
  /** The best candidates now: after the last candidate, as the study's combine setting defines best. */
  readonly best: Set<number>;
  /** For each candidate in the best set at any point: the candidate after which it entered, and after which it left
   * (null: still in). */
  readonly bestSpans: Map<number, { from: number; to: number | null }[]>;

  constructor(readonly record: StudyRecord, readonly id: string, readonly name: string) {
    this.h = record.header;
    this.candidates = record.candidates;
    this.byNumber = new Map(this.candidates.map((t) => [t.number, t]));
    this.space = this.h.space;
    this.spaceByKey = new Map(this.space.map((s) => [s.key, s]));
    this.blockOf = new Map();
    for (const [name, b] of Object.entries(this.h.blocks)) for (const k of b.keys) this.blockOf.set(k, name);
    this.blockOrder = this.h.block_order;
    this.objectives = this.h.objectives.map((objective, index) => ({ role: "objective" as const, index, metric: objective.metric, objective }));
    this.constraints = this.h.constraints.map((constraint, index) => ({ role: "constraint" as const, index, metric: constraint.metric, constraint }));
    this.goals = [...this.objectives, ...this.constraints];

    // the schedule tree
    this.scopes = new Map();
    this.roots = [];
    for (const event of record.scopes) {
      const parent = event.parent === null ? null : this.scopes.get(event.parent) ?? null;
      const scope: Scope = { id: event.id, event, end: record.scopeEnds.get(event.id) ?? null, parent, depth: parent ? parent.depth + 1 : 0, children: [], candidates: [], told: null };
      this.scopes.set(event.id, scope);
      if (parent) parent.children.push(scope);
      else this.roots.push(scope);
    }
    for (const t of this.candidates) for (const id of t.path) this.scopes.get(id)?.candidates.push(t.number);
    for (const told of record.told) {
      const scope = this.scopes.get(told.scope);
      if (scope) scope.told = told;
    }
    const levels = new Map<string, Level>();
    for (const s of this.scopes.values()) {
      if (s.depth === 0) continue;
      const key = `${s.depth}|${s.event.node}|${s.event.outer}`;
      if (!levels.has(key)) levels.set(key, { depth: s.depth, node: s.event.node, outer: s.event.outer, scopes: [] });
      levels.get(key)!.scopes.push(s);
    }
    this.levels = [...levels.values()].sort((a, b) => a.depth - b.depth);

    this.scheduleNodes = new Map();
    const walk = (n: ScheduleNode) => {
      this.scheduleNodes.set(n.node, n);
      Object.values(n.children).forEach(walk);
    };
    walk(this.h.schedule);

    // best now, and when each candidate was in the best set
    const last = this.candidates.at(-1);
    this.best = new Set(record.end?.type === "finished" ? record.end.best : last?.best_after ?? []);
    this.bestSpans = new Map();
    let prev = new Set<number>();
    for (const t of this.candidates) {
      const now = new Set(t.best_after);
      for (const n of now) if (!prev.has(n)) this.bestSpans.set(n, [...(this.bestSpans.get(n) ?? []), { from: t.number, to: null }]);
      for (const n of prev)
        if (!now.has(n)) {
          const spans = this.bestSpans.get(n)!;
          spans[spans.length - 1].to = t.number;
        }
      prev = now;
    }
  }

  // ---- the study --------------------------------------------------------------------------------------------------
  get state(): Lifecycle {
    return this.record.state;
  }
  get baseline(): CandidateEvent | undefined {
    return this.candidates.find((t) => t.role === "baseline");
  }
  get multi(): boolean {
    return this.objectives.length > 1;
  }
  get bestIsSet(): boolean {
    return this.h.combine.best_is_set;
  }
  /** The name of the one number the methods were told, when it is not an objective's metric (WeightedSum,
   * Chebyshev): what they optimized, worth its own panel. */
  get toldScalar(): string | null {
    const names = this.h.combine.told?.[0];
    if (!names || names.length !== 1 || this.h.combine.copies !== 1) return null;
    return this.h.objectives.some((o) => o.metric === names[0]) ? null : names[0];
  }
  /** Every metric any candidate has, goals first, then evaluators, then usage, in the header's order. */
  get metrics(): string[] {
    const seen = new Set(this.candidates.flatMap((t) => Object.keys(t.estimates)));
    const goals = [...new Set(this.goals.map((g) => g.metric))];
    const rest = Object.keys(this.h.metrics).filter((m) => seen.has(m) && !goals.includes(m));
    const role = (m: string) => (this.h.metrics[m].role === "evaluator" ? 0 : 1);
    return [...goals, ...rest.sort((a, b) => role(a) - role(b))];
  }
  range(metric: string): [number | null, number | null] {
    return this.h.metrics[metric]?.range ?? [null, null];
  }
  scheduleAt(scope: Scope): ScheduleNode | undefined {
    return this.scheduleNodes.get(scope.event.node);
  }

  // ---- one candidate ------------------------------------------------------------------------------------------------
  est(t: CandidateEvent, metric: string): Estimate | null {
    return t.estimates[metric] ?? null;
  }
  mean(t: CandidateEvent, metric: string): number | null {
    return t.estimates[metric]?.mean ?? null;
  }
  violation(t: CandidateEvent, constraint: number): number | null {
    return t.violations[constraint] ?? null;
  }
  markKind(t: CandidateEvent): MarkKind {
    if (t.status === "invalid" || t.status === "failed") return "unmeasured";
    if (!t.eligible) return "ineligible";
    return t.feasible === false ? "infeasible" : "ok";
  }
  /** The blocks the candidate's methods proposed, in block order. */
  changed(t: CandidateEvent): string[] {
    return this.blockOrder.filter((b) => t.proposed.includes(b));
  }
  /** Every choice the candidate made (at any depth), outermost first: planner=plan_execute, memory=vector. */
  choices(t: CandidateEvent): Choice[] {
    return this.space.filter((s) => SELECTORS.has(s.type) && s.key in t.params).map((s) => ({ key: s.key, option: String(t.params[s.key]) }));
  }
  /** The scopes the candidate ran in, outermost first. */
  path(t: CandidateEvent): Scope[] {
    return t.path.map((id) => this.scopes.get(id)).filter((s): s is Scope => !!s);
  }
  /** A round's number, counted within its enclosing round: "3", or "3.2" for the second round inside round 3. */
  roundNumber(scope: Scope): string {
    const parts: number[] = [];
    for (let s: Scope | null = scope; s && s.depth > 0; s = s.parent) parts.unshift((s.event.round ?? 0) + 1);
    return parts.join(".");
  }
  /** The model usage the methods spent proposing this candidate (reflection's own calls). */
  searchTokens(t: CandidateEvent): number {
    return Object.values(t.origin).reduce((sum, o) => sum + (o.usage?.tokens ?? 0), 0);
  }

  // ---- the space ----------------------------------------------------------------------------------------------------
  /** The choices a variable exists under, outermost first. */
  conditions(key: string): Choice[] {
    const out: Choice[] = [];
    let at = this.spaceByKey.get(key)?.active_when ?? null;
    while (at) {
      out.unshift({ key: at.key, option: at.option });
      at = this.spaceByKey.get(at.key)?.active_when ?? null;
    }
    return out;
  }

  private versions = new Map<string, { text: string; first: number | null; candidates: number[] }[]>();

  /** Every distinct value of a text variable, in order of first use (v0 is the default). */
  textVersions(key: string): { text: string; first: number | null; candidates: number[] }[] {
    const cached = this.versions.get(key);
    if (cached) return cached;
    const row = this.spaceByKey.get(key);
    const out: { text: string; first: number | null; candidates: number[] }[] = [{ text: String(row?.default ?? ""), first: null, candidates: [] }];
    for (const t of this.candidates) {
      const raw = t.params[key];
      const v = raw === null && row?.default === null ? "" : raw; // an empty skill library is the default
      if (typeof v !== "string") continue;
      let found = out.find((o) => o.text === v);
      if (!found) out.push((found = { text: v, first: t.number, candidates: [] }));
      if (found.first === null) found.first = t.number;
      found.candidates.push(t.number);
    }
    this.versions.set(key, out);
    return out;
  }
}
