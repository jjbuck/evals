// One candidate: what it was, how it was proposed, and how each case went. Everything is the record's: the schedule
// path (with what each round's outer proposal was told), each proposed block's origin, every variable's provenance, the
// change from its base candidate, each goal with its interval, and every trial by case and repetition.

import { Fragment, type ReactNode } from "react";
import { BlockChip, Changed, Choices, EstimateText, SchedulePath, CandidateLink, valueText } from "../components/common";
import { TextDiff } from "../components/Diff";
import { MUTED, MARK } from "../encoding/theme";
import { ci, num, secs, when, words } from "../format";
import { goalTitle, objectiveOf } from "../model/goals";
import type { Goal, Scope, StudyModel } from "../model/study";
import type { Trial, Origin, CandidateEvent } from "../record/schema";
import { href } from "../router";
import { breakable, provenance, roleText } from "./Candidates";
import "./Candidate.css";

// ---- small shared pieces (also used by the trial page) --------------------------------------------------------------

export function trialHref(study: StudyModel, t: CandidateEvent, a: Trial): string {
  return href(study.id, { page: "trial", n: t.number, case: a.case, rep: a.repetition });
}

/** The first line of an error, for grouping and previews. */
export function firstLine(s: string | null): string {
  if (!s) return "";
  const line = s.split("\n").find((l) => l.trim()) ?? "";
  return line.length > 220 ? `${line.slice(0, 220)}…` : line;
}

/** The evaluator scores a trial whose agent raised is given, as the header records them ({metric: value or null};
 * null leaves the failure out of that metric). */
export function agentFailureValues(study: StudyModel): [string, number | null][] {
  return Object.entries(study.h.agent_error_score);
}

/** A trial's value of a metric as the study counts it (projection.value): nothing unless scored; for an agent
 * failure, an evaluator metric takes the header's agent_error_score value (null: left out); otherwise the recorded value.
 * ``failure`` says the value is the agent_failure one, not a measurement. */
export function trialValue(study: StudyModel, a: Trial, metric: string): { v: number | null; failure: boolean } {
  if (!a.scored) return { v: null, failure: false };
  if (a.status === "agent_failure" && study.h.metrics[metric]?.role === "evaluator") return { v: study.h.agent_error_score[metric] ?? null, failure: true };
  const v = a.metrics[metric];
  return { v: v === undefined ? null : v, failure: false };
}

export function agentFailureText(study: StudyModel): string | null {
  const vs = agentFailureValues(study);
  if (!vs.length) return null;
  return vs.map(([m, v]) => `${m} = ${v === null ? "null" : num(v)}`).join(", ");
}

export function Count({ n }: { n: number }) {
  return <span className="count">{n}</span>;
}

/** A recorded value of any shape, verbatim: strings and numbers as written, objects as key = value. */
export function Verbatim({ v }: { v: unknown }) {
  if (v === null) return <span className="muted">null</span>;
  if (v === undefined) return <span className="muted">–</span>;
  if (typeof v === "string") return <span className="text">{v}</span>;
  if (typeof v === "number" || typeof v === "boolean") return <span className="tnum">{String(v)}</span>;
  if (Array.isArray(v)) return <span className="mono small">{JSON.stringify(v)}</span>;
  const entries = Object.entries(v as Record<string, unknown>);
  if (!entries.length) return <span className="muted">{"{}"}</span>;
  return (
    <span className="kv-inline">
      {entries.map(([k, x]) => (
        <span key={k}>
          <span className="mono">{k}</span> = <Verbatim v={x} />
        </span>
      ))}
    </span>
  );
}

function costText(o: Origin): string | null {
  const parts: string[] = [];
  if (o.usage) {
    const u = o.usage;
    const io = [`${num(u.input_tokens)} in`, `${num(u.output_tokens)} out`];
    if (u.cache_read_tokens) io.push(`${num(u.cache_read_tokens)} cache read`);
    if (u.cache_write_tokens) io.push(`${num(u.cache_write_tokens)} cache write`);
    parts.push(`${num(u.tokens)} tokens (${io.join(" · ")})`);
  }
  if (o.seconds !== undefined) parts.push(secs(o.seconds));
  return parts.length ? parts.join(" · ") : null;
}

function feasibleWords(t: CandidateEvent): string {
  return t.feasible === null ? "cannot be judged" : t.feasible ? "yes" : "no";
}

function Check({ v }: { v: number | null }) {
  if (v === null) return <span className="muted" title="not measured">–</span>;
  return v <= 0 ? <span className="good" title={`violation ${num(v)}`}>✓</span> : <span className="bad" title={`violation ${num(v)}`}>✗</span>;
}

// ---- header ---------------------------------------------------------------------------------------------------------

function Head({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  const prev = study.byNumber.get(t.number - 1);
  const next = study.byNumber.get(t.number + 1);
  const bestWord = study.bestIsSet ? "Pareto set" : "best";
  const spans = study.bestSpans.get(t.number) ?? [];
  const past = spans.filter((s) => s.to !== null);
  return (
    <div className="candidate-head">
      <div className="candidate-nav">
        {prev ? <a href={href(study.id, { page: "candidate", n: prev.number })}>‹ candidate {prev.number}</a> : <span />}
        {next ? <a href={href(study.id, { page: "candidate", n: next.number })}>candidate {next.number} ›</a> : <span />}
      </div>
      <div className="candidate-title">
        <h2>Candidate {t.number}</h2>
        {study.best.has(t.number) && <span className="tag">{bestWord}</span>}
        <span className={`tag${t.status === "complete" ? "" : " warn"}`}>{words(t.status)}</span>
        {past.length > 0 && (
          <span className="muted small">
            {bestWord} after candidate{past.length > 1 || past[0].to! - 1 > past[0].from ? "s" : ""} {past.map((s) => (s.to! - 1 > s.from ? `${s.from}–${s.to! - 1}` : `${s.from}`)).join(", ")}
          </span>
        )}
      </div>
      <div className="candidate-sub">
        <span>
          <SchedulePath study={study} t={t} />
          {t.role !== "baseline" && (
            <>
              {study.path(t).some((s) => s.depth > 0) ? " · " : ""}
              {roleText(t)}
            </>
          )}
        </span>
        {t.role !== "baseline" && <Changed study={study} t={t} />}
        <Choices study={study} t={t} />
        <span className="muted">finished {when(t.finished_at)}</span>
      </div>
      {t.reason && (
        <div className="candidate-reason">
          <span className="k">reason</span> <span className="text">{t.reason}</span>
        </div>
      )}
    </div>
  );
}

// ---- goals ----------------------------------------------------------------------------------------------------------

function GoalRows({ study, t, goals, base }: { study: StudyModel; t: CandidateEvent; goals: Goal[]; base: CandidateEvent | undefined }) {
  const unit = study.h.estimate_unit;
  return (
    <>
      {goals.map((g) => {
        const e = study.est(t, g.metric);
        return (
          <tr key={`${g.role}${g.index}`}>
            <td className="goal-name">{breakable(goalTitle(g))}</td>
            <td className="num">
              <EstimateText e={e} />
            </td>
            <td className="num muted small">{e ? `${e.n} ${unit}${e.n === 1 ? "" : "s"}` : ""}</td>
            <td className="check">{g.role === "constraint" ? <Check v={study.violation(t, g.index)} /> : null}</td>
            {base && <td className="num muted">{num(study.mean(base, g.metric))}</td>}
          </tr>
        );
      })}
    </>
  );
}

function Goals({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  const base = t.base === null ? undefined : study.byNumber.get(t.base);
  const head = (
    <thead>
      <tr>
        <th />
        <th className="num">candidate {t.number}</th>
        <th className="num">n</th>
        <th />
        {base && (
          <th className="num">
            <CandidateLink study={study} n={base.number}>candidate {base.number}</CandidateLink> <span className="muted">(base)</span>
          </th>
        )}
      </tr>
    </thead>
  );
  return (
    <section className="card candidate-goals">
      <h2>Goals</h2>
      <div className="group-head">
        Objectives <Count n={study.objectives.length} />
      </div>
      {study.objectives.length ? (
        <table className="grid-table goal-table">
          {head}
          <tbody>
            <GoalRows study={study} t={t} goals={study.objectives} base={base} />
          </tbody>
        </table>
      ) : (
        <div className="none-box">No objectives</div>
      )}
      <div className="group-head">
        Constraints <Count n={study.constraints.length} />
      </div>
      {study.constraints.length ? (
        <table className="grid-table goal-table">
          {head}
          <tbody>
            <GoalRows study={study} t={t} goals={study.constraints} base={base} />
          </tbody>
        </table>
      ) : (
        <div className="none-box">No constraints</div>
      )}
      <div className="facts">
        <span>
          <span className="k">eligible</span> {t.eligible ? "yes" : "no"}
        </span>
        <span>
          <span className="k">feasible</span> {feasibleWords(t)}
        </span>
      </div>
    </section>
  );
}

// ---- the schedule ---------------------------------------------------------------------------------------------------

function candidateRange(study: StudyModel, s: Scope): ReactNode {
  if (!s.candidates.length) return <span className="muted">none</span>;
  const first = s.candidates[0];
  const last = s.candidates[s.candidates.length - 1];
  return (
    <>
      <CandidateLink study={study} n={first}>{first}</CandidateLink>
      {last !== first && (
        <>
          –<CandidateLink study={study} n={last}>{last}</CandidateLink>
        </>
      )}
      <span className="muted">
        {" "}
        · {s.end ? s.end.ran : s.candidates.length} of {s.event.limit}
        {s.end ? "" : " · open"}
      </span>
    </>
  );
}

function ToldText({ study, s }: { study: StudyModel; s: Scope }) {
  const told = s.told;
  if (!told) return <span className="muted">–</span>;
  return (
    <span>
      <BlockChip study={study} block={told.block} /> <span className="k">draw</span>
      <CandidateLink study={study} n={told.draw_candidate}>{told.draw_candidate}</CandidateLink>
      {told.candidate === null ? (
        <>
          <span className="k"> · candidate</span>
          <span className="muted">none</span>
        </>
      ) : (
        <>
          <span className="k"> · candidate</span>
          <CandidateLink study={study} n={told.candidate}>{told.candidate}</CandidateLink>
          {told.values && <span className="tnum"> · {told.values.map((v) => num(v)).join(", ")}</span>}
        </>
      )}
      {told.objective !== null && objectiveOf(study, told.objective) && <span className="muted"> · for {objectiveOf(study, told.objective)}</span>}
    </span>
  );
}

function Schedule({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  const path = study.path(t);
  const served = objectiveOf(study, t.objective);
  const role = roleText(t);
  return (
    <section className="card candidate-schedule">
      <h2>Schedule</h2>
      {path.length > 0 ? (
        <div className="path-list">
          {path.map((s) => {
            const obj = objectiveOf(study, s.event.objective);
            return (
              <div key={s.id} className={`scope-row${s.id === t.scope ? " here" : ""}`} style={{ marginLeft: s.depth * 16 }}>
                <div className="scope-line">
                  <span className="scope-name">
                    {s.depth === 0 ? (
                      <span className="muted">run</span>
                    ) : (
                      <>
                        {s.event.outer ? <BlockChip study={study} block={s.event.outer} /> : null} <b>round {study.roundNumber(s)}</b>
                        {obj && <span className="muted"> · for {obj}</span>}
                      </>
                    )}
                    <span className="mono small muted"> · {s.event.kind}</span>
                  </span>
                  <span className="scope-candidates small">
                    <span className="k">candidates</span>
                    {candidateRange(study, s)}
                  </span>
                </div>
                <div className="scope-line small">
                  <span className="k">blocks</span>
                  <span className="chips">
                    {s.event.blocks.map((b) => (
                      <BlockChip key={b} study={study} block={b} />
                    ))}
                  </span>
                </div>
                {s.depth > 0 && (
                  <div className="scope-line small">
                    <span className="k">told</span>
                    <ToldText study={study} s={s} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="muted small">no scope</div>
      )}
      <dl className="facts-dl">
        <dt>role</dt>
        <dd>
          {role}
          {served && <span className="muted"> · for {served}</span>}
        </dd>
        <dt>shared</dt>
        <dd>
          {t.shared.length ? (
            <span className="chips">
              {t.shared.map((b) => (
                <BlockChip key={b} study={study} block={b} />
              ))}
            </span>
          ) : (
            <span className="muted">none</span>
          )}
        </dd>
        {t.deferred.length > 0 && (
          <>
            <dt>deferred</dt>
            <dd>
              <span className="chips">
                {t.deferred.map((b) => (
                  <BlockChip key={b} study={study} block={b} />
                ))}
              </span>
            </dd>
          </>
        )}
        {t.told.map((v, k) => (
          <Fragment key={k}>
            <dt>
              told
              {t.told.length > 1 ? ` · ${objectiveOf(study, k) ?? k + 1}` : ""}
            </dt>
            <dd className="tnum">
              {v === null ? <span className="muted">none</span> : v.map((x) => num(x)).join(", ")}
              <span className="muted"> · {study.h.combine.kind}</span>
            </dd>
          </Fragment>
        ))}
        <dt>best_after</dt>
        <dd>
          {t.best_after.length ? (
            t.best_after.map((b, i) => (
              <Fragment key={b}>
                {i > 0 && ", "}
                <CandidateLink study={study} n={b}>{b}</CandidateLink>
              </Fragment>
            ))
          ) : (
            <span className="muted">none</span>
          )}
        </dd>
      </dl>
    </section>
  );
}

// ---- how it was proposed --------------------------------------------------------------------------------------------

const ORIGIN_SHOWN = new Set(["method", "status", "error", "usage", "seconds"]);

function VarTable({ study, t, keys, from }: { study: StudyModel; t: CandidateEvent; keys: string[]; from?: (k: string) => ReactNode }) {
  return (
    <table className="grid-table var-table">
      <tbody>
        {keys.map((k) => (
          <tr key={k}>
            <td className="mono small">{k}</td>
            <td className="tnum">
              {!(k in t.params) ? (
                <span className="muted">–</span>
              ) : study.spaceByKey.get(k)?.type === "text" && typeof t.params[k] === "string" ? (
                <details className="text-value">
                  <summary>{valueText(study, study.spaceByKey.get(k), t.params[k])}</summary>
                  <div className="textbox">{String(t.params[k])}</div>
                </details>
              ) : (
                valueText(study, study.spaceByKey.get(k), t.params[k])
              )}
            </td>
            {from && <td className="small">{from(k)}</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Proposal({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  const blocks = study.changed(t);
  const extra = t.proposed.filter((b) => !blocks.includes(b)); // proposed blocks the header does not order (never hidden)
  const inherited = Object.keys(t.inherited);
  const held = Object.keys(t.params).filter((k) => provenance(study, t, k)?.how === "held");
  return (
    <section className="card candidate-proposal">
      <h2>Proposal</h2>
      {[...blocks, ...extra].map((b) => {
        const o = t.origin[b];
        const cost = o ? costText(o) : null;
        const rest = o ? Object.entries(o).filter(([k]) => !ORIGIN_SHOWN.has(k)) : [];
        const drawn = t.drawn[b] ?? [];
        return (
          <div key={b} className="pblock">
            <div className="pblock-head">
              <BlockChip study={study} block={b} />
              {o ? <b className="mono">{o.method}</b> : <span className="muted">no origin</span>}
              {o?.status !== undefined && <span className="tag">{o.status}</span>}
              {cost && <span className="muted small tnum">{cost}</span>}
            </div>
            {o?.error !== undefined && (
              <div className="pblock-error">
                <span className="k">error</span> <span className="text">{o.error}</span>
              </div>
            )}
            <div className="pblock-body">
              {rest.length > 0 && (
                <table className="grid-table kv-table">
                  <tbody>
                    {rest.map(([k, v]) => (
                      <tr key={k}>
                        <td className="mono small">{k}</td>
                        <td>
                          <Verbatim v={v} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <div className="pblock-vars">
                <div className="sub-head">
                  drawn <Count n={drawn.length} />
                </div>
                {drawn.length ? <VarTable study={study} t={t} keys={drawn} /> : <div className="muted small">none</div>}
              </div>
            </div>
          </div>
        );
      })}
      {(inherited.length > 0 || t.defaulted.length > 0 || held.length > 0) && (
        <div className="pvars">
          {inherited.length > 0 && (
            <div>
              <div className="sub-head">
                inherited <Count n={inherited.length} />
              </div>
              <VarTable study={study} t={t} keys={inherited} from={(k) => <>from <CandidateLink study={study} n={t.inherited[k]} /></>} />
            </div>
          )}
          {t.defaulted.length > 0 && (
            <div>
              <div className="sub-head">
                defaulted <Count n={t.defaulted.length} />
              </div>
              <VarTable study={study} t={t} keys={t.defaulted} />
            </div>
          )}
          {held.length > 0 && (
            <details>
              <summary className="sub-head">
                held from {t.base === null ? "–" : `candidate ${t.base}`} <Count n={held.length} />
              </summary>
              <VarTable study={study} t={t} keys={held} />
            </details>
          )}
        </div>
      )}
    </section>
  );
}

// ---- the change from its base candidate -----------------------------------------------------------------------------

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function BaseChanges({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  if (t.base === null) return null;
  const base = study.byNumber.get(t.base);
  if (!base) return null;
  const order = [...study.space.map((s) => s.key)];
  for (const k of [...Object.keys(base.params), ...Object.keys(t.params)]) if (!order.includes(k)) order.push(k);
  const changed = order.filter((k) => (k in t.params || k in base.params) && !same(t.params[k], base.params[k]));
  const texts = changed.filter((k) => study.spaceByKey.get(k)?.type === "text" && (typeof t.params[k] === "string" || typeof base.params[k] === "string"));
  return (
    <section className="card candidate-changes">
      <h2>
        Changes from <CandidateLink study={study} n={base.number} /> <span className="muted small">base</span>
      </h2>
      {changed.length === 0 ? (
        <div className="muted">no changes</div>
      ) : (
        <>
          <table className="grid-table change-table">
            <thead>
              <tr>
                <th />
                <th>variable</th>
                <th>candidate {base.number}</th>
                <th>candidate {t.number}</th>
              </tr>
            </thead>
            <tbody>
              {changed.map((k) => {
                const s = study.spaceByKey.get(k);
                const b = study.blockOf.get(k);
                return (
                  <tr key={k}>
                    <td>{b && <BlockChip study={study} block={b} />}</td>
                    <td className="mono small">{k}</td>
                    <td className="tnum muted">{k in base.params ? valueText(study, s, base.params[k]) : "–"}</td>
                    <td className="tnum">{k in t.params ? valueText(study, s, t.params[k]) : "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {texts.map((k) => (
            <div key={k} className="text-change">
              <div className="sub-head mono">{k}</div>
              {typeof base.params[k] === "string" && typeof t.params[k] === "string" ? (
                <TextDiff before={base.params[k] as string} after={t.params[k] as string} />
              ) : (
                <div className="textbox">{String(t.params[k] ?? base.params[k])}</div>
              )}
            </div>
          ))}
        </>
      )}
    </section>
  );
}

// ---- cases ----------------------------------------------------------------------------------------------------------

type FailGroup = { status: string; error: string | null; scored: boolean; trials: Trial[] };

function failureGroups(t: CandidateEvent): FailGroup[] {
  const groups = new Map<string, FailGroup>();
  for (const a of t.trials) {
    if (a.status === "completed" && !a.unavailable.length) continue;
    const key = JSON.stringify([a.status, a.scored, a.error, a.unavailable]);
    if (!groups.has(key)) groups.set(key, { status: a.status, error: a.error, scored: a.scored, trials: [] });
    groups.get(key)!.trials.push(a);
  }
  return [...groups.values()].sort((x, y) => y.trials.length - x.trials.length);
}

function groupWords(g: FailGroup): string {
  const parts = [words(g.status)];
  if (g.trials[0].unavailable.length) parts.push(`unavailable ${g.trials[0].unavailable.join(", ")}`);
  return parts.join(" · ");
}

function Failures({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  const groups = failureGroups(t);
  if (!groups.length) return null;
  return (
    <div className="failures">
      {groups.map((g, i) => {
        const byCase = new Map<string, Trial[]>();
        for (const a of g.trials) byCase.set(a.case, [...(byCase.get(a.case) ?? []), a]);
        return (
          <div key={i} className="failure">
            <span className="fail-count tnum">
              {g.trials.length}/{t.trials.length}
            </span>
            <span className="fail-what">
              <b>{groupWords(g)}</b>
              {g.scored && <span className="tag">scored</span>}
              {g.status === "agent_failure" && agentFailureText(study) && <span className="muted small"> agent_failure: {agentFailureText(study)}</span>}
              {g.error && (
                <div className="mono small fail-error" title={g.error}>
                  {firstLine(g.error)}
                </div>
              )}
            </span>
            <span className="fail-where small">
              {[...byCase.entries()].map(([c, as], j) => (
                <span key={c}>
                  {j > 0 && " · "}
                  {c}{" "}
                  {as.map((a, k) => (
                    <Fragment key={a.repetition}>
                      {k > 0 && ", "}
                      <a href={trialHref(study, t, a)}>{a.repetition}</a>
                    </Fragment>
                  ))}
                </span>
              ))}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** One trial's value of a metric, as a link to the trial: the number, × when it was not scored, n/a when the
 * metric is unavailable. */
function RepValue({ study, t, a, metric }: { study: StudyModel; t: CandidateEvent; a: Trial; metric: string }) {
  const title = `${a.case} · repetition ${a.repetition} · ${words(a.status)}${a.error ? `\n${firstLine(a.error)}` : ""}`;
  const { v, failure } = trialValue(study, a, metric);
  if (!a.scored) return <a className="rep x" href={trialHref(study, t, a)} title={title}>×</a>;
  if (v === null)
    return (
      <a className="rep na" href={trialHref(study, t, a)} title={`${title}\n${metric} ${failure ? "agent_failure: null" : "unavailable"}`}>
        n/a
      </a>
    );
  return (
    <a className={`rep${a.status === "completed" ? "" : " flagged"}`} href={trialHref(study, t, a)} title={`${title}\n${metric} = ${v}${failure ? " (agent_failure)" : ""}`}>
      {num(v)}
    </a>
  );
}

const STRIP_W = 70;
const STRIP_H = 14;

/** Many repetitions: one dot per scored repetition on a small strip (the metric's range, else this candidate's values),
 * the case mean as a tick; each dot opens its trial. */
function RepStrip({ study, t, as, metric, domain }: { study: StudyModel; t: CandidateEvent; as: Trial[]; metric: string; domain: [number, number] }) {
  const [lo, hi] = domain;
  const x = (v: number) => (hi === lo ? STRIP_W / 2 : 4 + ((v - lo) / (hi - lo)) * (STRIP_W - 8));
  const vals = as.map((a) => ({ a, ...trialValue(study, a, metric) })).filter((x): x is { a: Trial; v: number; failure: boolean } => x.v !== null);
  const mean = vals.length ? vals.reduce((s, x) => s + x.v, 0) / vals.length : null;
  return (
    <svg width={STRIP_W} height={STRIP_H} className="strip">
      <line x1={2} x2={STRIP_W - 2} y1={STRIP_H / 2} y2={STRIP_H / 2} stroke={MUTED} strokeOpacity={0.4} />
      {mean !== null && <line x1={x(mean)} x2={x(mean)} y1={1} y2={STRIP_H - 1} stroke={MARK} strokeWidth={1.5} />}
      {vals.map(({ a, v, failure }) => (
        <a key={a.repetition} href={trialHref(study, t, a)}>
          <title>{`${a.case} · repetition ${a.repetition} · ${words(a.status)} · ${metric} = ${v}${failure ? " (agent_failure)" : ""}`}</title>
          <circle cx={x(v)} cy={STRIP_H / 2} r={2.6} fill={a.status === "completed" ? MARK : "#fff"} stroke={MARK} strokeWidth={1.2} fillOpacity={0.75} />
        </a>
      ))}
    </svg>
  );
}

/** A case's mean of a metric over its scored repetitions, each valued as the study counts it (projection.case_means). */
function caseMean(study: StudyModel, as: Trial[], metric: string): number | null {
  const vs = as.map((a) => trialValue(study, a, metric).v).filter((v): v is number => v !== null);
  return vs.length ? vs.reduce((s, v) => s + v, 0) / vs.length : null;
}

function Cases({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  const cases = [...study.h.cases];
  for (const a of t.trials) if (!cases.includes(a.case)) cases.push(a.case);
  const byCase = new Map(cases.map((c) => [c, t.trials.filter((a) => a.case === c).sort((x, y) => x.repetition - y.repetition)]));
  const reps = Math.max(0, ...[...byCase.values()].map((as) => as.length));
  const dots = reps > 3;
  const domain = (metric: string): [number, number] => {
    const vs = t.trials.map((a) => trialValue(study, a, metric).v).filter((v): v is number => v !== null);
    const [rlo, rhi] = study.range(metric);
    return [rlo ?? Math.min(...vs), rhi ?? Math.max(...vs)];
  };
  const goalCols: (Goal | null)[] = [...(study.objectives.length ? study.objectives : [null]), ...(study.constraints.length ? study.constraints : [null])];
  const ncols = goalCols.length;
  return (
    <section className="card candidate-cases">
      <h2>
        Cases <Count n={cases.length} />
        <span className="muted small"> · {t.trials.length} trials · {study.h.repetitions} repetition{study.h.repetitions === 1 ? "" : "s"}</span>
      </h2>
      {t.trials.length === 0 ? (
        <div className="muted">no trials</div>
      ) : (
        <>
          <Failures study={study} t={t} />
          <div className="cases-scroll">
            <table className="grid-table case-table">
              <thead>
                <tr className="case-groups">
                  <th colSpan={2} />
                  <th colSpan={Math.max(1, study.objectives.length)} className="gstart">
                    Objectives <Count n={study.objectives.length} />
                  </th>
                  <th colSpan={Math.max(1, study.constraints.length)} className="gstart">
                    Constraints <Count n={study.constraints.length} />
                  </th>
                </tr>
                <tr>
                  <th>case</th>
                  <th>trials</th>
                  {goalCols.map((g, i) => {
                    const start = i === 0 || (g?.role === "constraint" && goalCols[i - 1]?.role !== "constraint") || (g === null && i > 0);
                    return (
                      <th key={g ? `${g.role}${g.index}` : `none${i}`} className={`num${start ? " gstart" : ""}`}>
                        {g ? <div className="case-head">{breakable(goalTitle(g))}</div> : <span className="muted">none</span>}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {cases.map((c) => {
                  const as = byCase.get(c)!;
                  const scored = as.filter((a) => a.scored);
                  return (
                    <tr key={c}>
                      <td className="mono small case-name">{c}</td>
                      <td className="nowrap">
                        {as.length === 0 && <span className="muted">none</span>}
                        {as.map((a) => (
                          <a
                            key={a.repetition}
                            href={trialHref(study, t, a)}
                            className={`rep-link${a.status === "completed" ? "" : a.scored ? " flagged" : " x"}`}
                            title={`repetition ${a.repetition} · ${words(a.status)}${a.error ? `\n${firstLine(a.error)}` : ""}`}
                          >
                            {a.repetition}
                            {a.scored ? "" : "×"}
                          </a>
                        ))}
                      </td>
                      {scored.length === 0 && as.length > 0 ? (
                        <td colSpan={ncols} className="gstart muted small">
                          {as.length}/{as.length} {[...new Set(as.map((a) => words(a.status)))].join(", ")}
                        </td>
                      ) : (
                        goalCols.map((g, i) => {
                          const start = i === 0 || (g?.role === "constraint" && goalCols[i - 1]?.role !== "constraint") || (g === null && i > 0);
                          const key = g ? `${g.role}${g.index}` : `none${i}`;
                          if (!g) return <td key={key} className={start ? "gstart" : ""} />;
                          const mean = caseMean(study, as, g.metric);
                          return (
                            <td key={key} className={`num${start ? " gstart" : ""}`}>
                              <span className="cell">
                                {dots ? (
                                  <RepStrip study={study} t={t} as={as} metric={g.metric} domain={domain(g.metric)} />
                                ) : (
                                  as.length > 1 && (
                                    <span className="reps">
                                      {as.map((a) => (
                                        <RepValue key={a.repetition} study={study} t={t} a={a} metric={g.metric} />
                                      ))}
                                    </span>
                                  )
                                )}
                                {as.length === 1 && !dots ? (
                                  <RepValue study={study} t={t} a={as[0]} metric={g.metric} />
                                ) : (
                                  <b className="mean" title="case mean">
                                    {num(mean)}
                                  </b>
                                )}
                              </span>
                            </td>
                          );
                        })
                      )}
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={2} className="muted small">
                    estimate
                  </td>
                  {goalCols.map((g, i) => {
                    const start = i === 0 || (g?.role === "constraint" && goalCols[i - 1]?.role !== "constraint") || (g === null && i > 0);
                    const key = g ? `${g.role}${g.index}` : `none${i}`;
                    if (!g) return <td key={key} className={start ? "gstart" : ""} />;
                    const e = study.est(t, g.metric);
                    return (
                      <td key={key} className={`num${start ? " gstart" : ""}`}>
                        <EstimateText e={e} />
                        {g.role === "constraint" && (
                          <>
                            {" "}
                            <Check v={study.violation(t, g.index)} />
                          </>
                        )}
                      </td>
                    );
                  })}
                </tr>
              </tfoot>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

// ---- every metric -----------------------------------------------------------------------------------------------------

function AllMetrics({ study, t }: { study: StudyModel; t: CandidateEvent }) {
  const metrics = [...study.metrics];
  for (const m of Object.keys(t.estimates)) if (!metrics.includes(m)) metrics.push(m);
  const cases = study.h.cases.filter((c) => t.trials.some((a) => a.case === c));
  if (!metrics.length || !Object.keys(t.estimates).length) return null;
  return (
    <details className="card candidate-metrics">
      <summary>
        <h2>
          All metrics <Count n={metrics.length} />
        </h2>
      </summary>
      <div className="cases-scroll">
        <table className="grid-table">
          <thead>
            <tr>
              <th>metric</th>
              <th>role</th>
              <th className="num">estimate</th>
              <th className="num">n</th>
              {cases.map((c) => (
                <th key={c} className="num mono">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => {
              const e = study.est(t, m);
              return (
                <tr key={m}>
                  <td className="mono small">{m}</td>
                  <td className="muted small">{study.h.metrics[m]?.role ?? "–"}</td>
                  <td className="num">
                    {e && e.mean !== null ? (
                      <span>
                        {num(e.mean)} <span className="ci">{ci(e.ci)}</span>
                      </span>
                    ) : (
                      <span className="muted">–</span>
                    )}
                  </td>
                  <td className="num muted small">{e?.n ?? ""}</td>
                  {cases.map((c) => (
                    <td key={c} className="num">
                      {num(caseMean(study, t.trials.filter((a) => a.case === c), m))}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function Candidate({ study, n }: { study: StudyModel; n: number }) {
  const t = study.byNumber.get(n);
  if (!t) return <div className="empty">No candidate {n}</div>;
  return (
    <div className="candidate">
      <Head study={study} t={t} />
      <div className="candidate-top">
        <Goals study={study} t={t} />
        <Schedule study={study} t={t} />
      </div>
      <Proposal study={study} t={t} />
      <BaseChanges study={study} t={t} />
      <Cases study={study} t={t} />
      <AllMetrics study={study} t={t} />
    </div>
  );
}
