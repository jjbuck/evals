// The study record (format strands_evals.optimization, schema 1), exactly as strands_evals/optimization/record.py
// writes it.

export const FORMAT = "strands_evals.optimization";
export const SCHEMA = 2;
/** Older schemas this explorer still reads, normalized to the current one as they are parsed. */
export const OLDER_SCHEMAS = [1];

export type Objective = { kind: "Maximize" | "Minimize"; metric: string; weight: number; scale: number | null; tolerance: number };
export type Constraint = { kind: "AtMost" | "AtLeast"; metric: string; limit: number; name: string };
/** ``told``: per method copy, what each number it is told is (an objective's metric, or the combination's name). */
export type Combine = { kind: string; best_is_set: boolean; copies: number; told: string[][]; [setting: string]: unknown };
export type ScheduleNode = { node: string; kind: string; children: Record<string, ScheduleNode>; [setting: string]: unknown };
export type MethodDesc = { kind: string; [setting: string]: unknown };
export type BlockDesc = { method: MethodDesc; keys: string[]; structural: boolean };
export type Range = [number | null, number | null];
export type MetricDesc = { role: "evaluator" | "usage" | "derived"; range: Range };

export type SpaceRow = {
  key: string;
  type: "float" | "int" | "choice" | "text" | "skills" | "one_of" | "switch";
  default: unknown;
  low?: number;
  high?: number;
  log?: boolean;
  step?: number | null;
  choices?: unknown[];
  active_when: { key: string; option: string } | null;
  /** text and skills: what the value is, for the model that rewrites it. */
  about?: string;
  /** text: the longest value that is measured. */
  max_chars?: number;
};

export type StudyEvent = {
  type: "study";
  format: string;
  schema: number;
  started_at: string;
  agent: string;
  objectives: Objective[];
  constraints: Constraint[];
  combine: Combine;
  schedule: ScheduleNode;
  blocks: Record<string, BlockDesc>;
  block_order: string[];
  space: SpaceRow[];
  metrics: Record<string, MetricDesc>;
  cases: string[];
  repetitions: number;
  estimate_unit: "case" | "trial";
  /** The evaluator score a trial whose agent raised gets, per metric (null: left out). */
  agent_error_score: Record<string, number | null>;
};

/** ``iterations``: candidates asked for, not counting the baseline the first run measures. ``patience``: stop once the
 * best has not changed for this many iterations (null: run every iteration). */
export type RunEvent = { type: "run"; started_at: string; iterations: number; patience: number | null };

export type ScopeEvent = {
  type: "scope";
  id: number;
  parent: number | null;
  node: string;
  kind: string;
  blocks: string[];
  objective: number | null;
  round: number | null;
  outer: string | null;
  limit: number;
  opened_at: string;
};

export type ScopeEndEvent = { type: "scope_end"; id: number; ran: number; blocks: string[]; closed_at: string };

export type Estimate = { mean: number | null; ci: [number, number] | null; n: number };

export type TrialStatus = "completed" | "agent_failure" | "infrastructure_invalid" | "verifier_invalid" | "cancelled";

export type Trial = {
  case: string;
  repetition: number;
  status: TrialStatus;
  scored: boolean;
  metrics: Record<string, number>;
  unavailable: string[];
  error: string | null;
  output: string | null;
  reasons: Record<string, string>;
  transcript: { hash: string; format: "strands-messages" } | null;
};

export type Usage = { input_tokens: number; output_tokens: number; cache_read_tokens: number; cache_write_tokens: number; tokens: number };
export type Origin = { method: string; status?: string; error?: string; usage?: Usage; seconds?: number; [detail: string]: unknown };

export type CandidateStatus = "complete" | "partial" | "failed" | "invalid";
export type Role = "baseline" | "outer" | "step";

export type CandidateEvent = {
  type: "candidate";
  finished_at: string;
  number: number;
  params: Record<string, unknown>;
  status: CandidateStatus;
  reason: string | null;
  eligible: boolean;
  feasible: boolean | null;
  violations: (number | null)[];
  scope: number | null;
  path: number[];
  role: Role;
  step: number | null;
  proposed: string[];
  drawn: Record<string, string[]>;
  inherited: Record<string, number>;
  defaulted: string[];
  base: number | null;
  objective: number | null;
  origin: Record<string, Origin>;
  shared: string[];
  deferred: string[];
  told: (number[] | null)[];
  best_after: number[];
  estimates: Record<string, Estimate>;
  trials: Trial[];
};

export type ToldEvent = {
  type: "told";
  scope: number;
  block: string;
  objective: number | null;
  draw_candidate: number;
  candidate: number | null;
  values: number[] | null;
};

/** A proposal repeated ``candidate``'s values, which were not measured again; ``told``: the blocks whose draws learned
 * that result now (a round that opens on ``candidate`` tells its outer draw the round's best later). */
export type RepeatEvent = {
  type: "repeat";
  repeated_at: string;
  scope: number | null;
  path: number[];
  role: Role;
  step: number | null;
  candidate: number;
  proposed: string[];
  drawn: Record<string, string[]>;
  origin: Record<string, Origin>;
  objective: number | null;
  told: string[];
  values: number[] | null;
};

/** ``reason``: every candidate asked for ran (``candidates``), the best stopped changing (``patience``), or the schedule had
 * nothing new to propose (``exhausted``). */
export type FinishedEvent = { type: "finished"; finished_at: string; best: number[]; reason: "candidates" | "patience" | "exhausted" };
export type StoppedEvent = { type: "stopped"; stopped_at: string; error: string };

export type Event = StudyEvent | RunEvent | ScopeEvent | ScopeEndEvent | CandidateEvent | RepeatEvent | ToldEvent | FinishedEvent | StoppedEvent;

/** A transcript: the trial's Strands messages. */
export type Transcript = unknown[];
