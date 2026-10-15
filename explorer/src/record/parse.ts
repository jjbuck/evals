// Reading a study.jsonl strictly: a strands_evals.optimization record of schema 1, every line an event; only a final
// line still being written may be incomplete. Anything else is an error that names the line, never a silently skipped
// or defaulted field.

import {
  FORMAT,
  OLDER_SCHEMAS,
  SCHEMA,
  type Event,
  type FinishedEvent,
  type RunEvent,
  type ScopeEndEvent,
  type ScopeEvent,
  type StoppedEvent,
  type StudyEvent,
  type RepeatEvent,
  type ToldEvent,
  type CandidateEvent,
} from "./schema";

export class RecordError extends Error {
  constructor(message: string, readonly line?: number) {
    super(line === undefined ? message : `line ${line}: ${message}`);
  }
}

export type Lifecycle = "running" | "finished" | "stopped";

export type StudyRecord = {
  header: StudyEvent;
  runs: RunEvent[];
  scopes: ScopeEvent[];
  scopeEnds: Map<number, ScopeEndEvent>;
  candidates: CandidateEvent[];
  repeats: RepeatEvent[];
  told: ToldEvent[];
  /** How the last run ended, or null while it runs. */
  end: FinishedEvent | StoppedEvent | null;
  state: Lifecycle;
  /** How many events the record holds (effects.json says how many it was fitted on). */
  events: number;
};

const REQUIRED: Record<Event["type"], string[]> = {
  study: ["format", "schema", "started_at", "agent", "objectives", "constraints", "combine", "schedule", "blocks", "block_order", "space", "metrics", "cases", "repetitions", "estimate_unit", "agent_error_score"],
  run: ["started_at", "iterations", "patience"],
  scope: ["id", "parent", "node", "kind", "blocks", "objective", "round", "outer", "limit", "opened_at"],
  scope_end: ["id", "ran", "blocks", "closed_at"],
  candidate: ["finished_at", "number", "params", "status", "reason", "eligible", "feasible", "violations", "scope", "path", "role", "step", "proposed", "drawn", "inherited", "defaulted", "base", "objective", "origin", "shared", "deferred", "told", "best_after", "estimates", "trials"],
  repeat: ["repeated_at", "scope", "path", "role", "step", "candidate", "proposed", "drawn", "origin", "objective", "told", "values"],
  told: ["scope", "block", "objective", "draw_candidate", "candidate", "values"],
  finished: ["finished_at", "best", "reason"],
  stopped: ["stopped_at", "error"],
};

/** The first event must be this format's header, of this schema. */
function header(e: unknown, line: number): void {
  const h = (typeof e === "object" && e !== null ? e : {}) as { type?: unknown; format?: unknown; schema?: unknown };
  if (h.type !== "study") throw new RecordError("the record does not start with a study event", line);
  if (h.format !== FORMAT)
    throw new RecordError(`not a ${FORMAT} record (format ${h.format === undefined ? "missing" : JSON.stringify(h.format)})`, line);
  if (h.schema !== SCHEMA && !OLDER_SCHEMAS.includes(h.schema as number))
    throw new RecordError(`${FORMAT} schema ${JSON.stringify(h.schema)}; this explorer reads schemas ${[...OLDER_SCHEMAS, SCHEMA].join(", ")}`, line);
}

/** Schema 1's names, as schema 2 states the same facts: ``agent_failure`` is ``agent_error_score``, ``policy`` is
 * ``schedule`` (and a node's ``inner_candidates`` and ``steps`` are ``inner_iterations`` and ``iterations``); a run's
 * ``n_candidates`` counted the baseline its first run measured, which ``iterations`` does not. */
function renamed(node: Record<string, unknown>): Record<string, unknown> {
  const { inner_candidates, steps, children, ...rest } = node;
  return {
    ...rest,
    ...(inner_candidates !== undefined ? { inner_iterations: inner_candidates } : {}),
    ...(steps !== undefined ? { iterations: steps } : {}),
    children: Object.fromEntries(Object.entries((children ?? {}) as Record<string, Record<string, unknown>>).map(([k, c]) => [k, renamed(c)])),
  };
}

function fromSchema1(e: Record<string, unknown>, firstRun: boolean): void {
  if (e.type === "study" && "agent_failure" in e) e.agent_error_score = e.agent_failure;
  if (e.type === "study" && "policy" in e) e.schedule = renamed(e.policy as Record<string, unknown>);
  if (e.type === "run" && "n_candidates" in e) e.iterations = (e.n_candidates as number) - (firstRun ? 1 : 0);
}

export function parseRecord(text: string): StudyRecord {
  const lines = text.split("\n");
  const events: { e: Event; line: number }[] = [];
  lines.forEach((raw, i) => {
    if (!raw.trim()) return;
    let e: unknown;
    try {
      e = JSON.parse(raw);
    } catch {
      if (i === lines.length - 1) return; // the line being written
      throw new RecordError("not valid JSON", i + 1);
    }
    const type = (e as { type?: unknown }).type;
    // the header first, so that a record of another format or schema says so rather than naming its first odd event
    if (!events.length) header(e, i + 1);
    if ((events.length ? (events[0].e as StudyEvent).schema : (e as { schema?: unknown }).schema) === 1)
      fromSchema1(e as Record<string, unknown>, type === "run" && !events.some((x) => x.e.type === "run"));
    if (typeof type !== "string" || !(type in REQUIRED)) throw new RecordError(`unknown event type ${JSON.stringify(type)}`, i + 1);
    const missing = REQUIRED[type as Event["type"]].filter((k) => !(k in (e as object)));
    if (missing.length) throw new RecordError(`${type} event without ${missing.join(", ")}`, i + 1);
    events.push({ e: e as Event, line: i + 1 });
  });
  if (!events.length) throw new RecordError("empty record");
  const first = events[0].e as StudyEvent;

  const record: StudyRecord = { header: first, runs: [], scopes: [], scopeEnds: new Map(), candidates: [], repeats: [], told: [], end: null, state: "running", events: events.length };
  for (const { e, line } of events.slice(1)) {
    switch (e.type) {
      case "study":
        throw new RecordError("a second study event: one record holds one study", line);
      case "run":
        record.runs.push(e);
        record.end = null;
        break;
      case "scope":
        record.scopes.push(e);
        break;
      case "scope_end":
        record.scopeEnds.set(e.id, e);
        break;
      case "candidate":
        if (e.number !== record.candidates.length) throw new RecordError(`candidate ${e.number} out of order`, line);
        record.candidates.push(e);
        break;
      case "repeat":
        record.repeats.push(e);
        break;
      case "told":
        record.told.push(e);
        break;
      case "finished":
      case "stopped":
        record.end = e;
        break;
    }
  }
  record.state = record.end === null ? "running" : record.end.type === "finished" ? "finished" : "stopped";
  return record;
}
