// What the study was configured to do, exactly as its header records it: the goals and how they combine, the schedule
// tree (with the scopes each node ran in), the blocks, the search space as a tree, and the experiment.

import type { ReactNode } from "react";
import { BlockChip, Long, valueText } from "../components/common";
import { num, when } from "../format";
import { arrow, goalTitle, objectiveOf, op } from "../model/goals";
import { SELECTORS, type StudyModel } from "../model/study";
import type { ScheduleNode, SpaceRow } from "../record/schema";
import "./Setup.css";

// ---- local helpers ------------------------------------------------------------------------------------------------

/** A recorded setting's value, as written: numbers formatted, null as "–", lists and objects as JSON-ish text. */
function settingText(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return num(v);
  if (typeof v === "string") return v;
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return `[${v.map(settingText).join(", ")}]`;
  return JSON.stringify(v);
}

/** A metric's range, as an interval: [0, 1], [0, ∞). */
function rangeText([lo, hi]: [number | null, number | null]): string {
  return `${lo === null ? "(−∞" : `[${num(lo)}`}, ${hi === null ? "∞)" : `${num(hi)}]`}`;
}

/** The settings of a recorded description (a schedule node, a method, combine): every field but the named ones. */
function settings(desc: Record<string, unknown>, skip: string[]): [string, unknown][] {
  return Object.entries(desc).filter(([k]) => !skip.includes(k));
}

function Settings({ study, items, blocks = [] }: { study: StudyModel; items: [string, unknown][]; blocks?: string[] }) {
  if (!items.length) return null;
  return (
    <span className="su-settings">
      {items.map(([k, v]) => (
        <span key={k} className="su-setting">
          <span className="su-sk">{k}</span>
          <span className="su-eq">=</span>
          {blocks.includes(k) && (typeof v === "string" || Array.isArray(v)) ? (
            <span className="chips">
              {(Array.isArray(v) ? v : [v]).map((b) => (study.h.blocks[String(b)] ? <BlockChip key={String(b)} study={study} block={String(b)} /> : <span key={String(b)}>{settingText(b)}</span>))}
              {Array.isArray(v) && !v.length && <span className="su-sv">[]</span>}
            </span>
          ) : (
            <span className="su-sv">{settingText(v)}</span>
          )}
        </span>
      ))}
    </span>
  );
}

/** Text that may wrap only after _ / - or a dot that ends a name (keys and metric names), never inside a word or a
 * number. */
function Breakable({ text }: { text: string }) {
  const parts = text.split(/(?<=[_/-])|(?<=\.)(?=[^\d])/);
  return (
    <>
      {parts.map((p, i) => (
        <span key={i}>
          {p}
          {i < parts.length - 1 && <wbr />}
        </span>
      ))}
    </>
  );
}

function Section({ title, count, children }: { title: string; count?: number; children: ReactNode }) {
  return (
    <section className="su-section">
      <div className="su-head">
        {title}
        {count !== undefined && <span className="count">{count}</span>}
      </div>
      {children}
    </section>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <div className="su-empty">{children}</div>;
}

// ---- 1. goals ------------------------------------------------------------------------------------------------------

/** The objective settings shown: each only when some objective differs from its default (weight 1, scale None,
 * tolerance 0, as Maximize/Minimize declare them). */
const OBJECTIVE_DEFAULTS: { key: "weight" | "scale" | "tolerance"; value: number | null }[] = [
  { key: "weight", value: 1 },
  { key: "scale", value: null },
  { key: "tolerance", value: 0 },
];

function Goals({ study }: { study: StudyModel }) {
  const shown = OBJECTIVE_DEFAULTS.filter((d) => study.h.objectives.some((o) => o[d.key] !== d.value));
  const hidden = OBJECTIVE_DEFAULTS.filter((d) => !shown.includes(d));
  const combine = study.h.combine;
  return (
    <>
      <div className="su-goals-grid">
      <Section title="Objectives" count={study.objectives.length}>
        {study.objectives.length ? (
          <div className="su-card">
            <table className="grid-table su-table">
              <thead>
                <tr>
                  <th className="num su-n">#</th>
                  <th>Kind</th>
                  <th>Metric</th>
                  {shown.map((d) => (
                    <th key={d.key} className="num">
                      {d.key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {study.objectives.map((g) => (
                  <tr key={g.index}>
                    <td className="num muted su-n">{g.index + 1}</td>
                    <td>
                      {arrow(g.objective)} {g.objective.kind}
                    </td>
                    <td className="mono su-wrap">
                      <Breakable text={g.metric} />
                    </td>
                    {shown.map((d) => (
                      <td key={d.key} className={`num${g.objective[d.key] === d.value ? " muted" : ""}`}>
                        {settingText(g.objective[d.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {hidden.length > 0 && (
              <div className="su-foot">
                {hidden.map((d) => `${d.key} = ${settingText(d.value)}`).join(" · ")}
              </div>
            )}
          </div>
        ) : (
          <Empty>No objectives</Empty>
        )}
      </Section>
      <Section title="Constraints" count={study.constraints.length}>
        {study.constraints.length ? (
          <div className="su-card">
            <table className="grid-table su-table">
              <thead>
                <tr>
                  <th className="num su-n">#</th>
                  <th>Metric</th>
                  <th>Operator</th>
                  <th className="num">Limit</th>
                  <th>Name</th>
                </tr>
              </thead>
              <tbody>
                {study.constraints.map((g) => (
                  <tr key={g.index}>
                    <td className="num muted su-n">{g.index + 1}</td>
                    <td className="mono su-wrap">
                      <Breakable text={g.metric} />
                    </td>
                    <td>
                      {op(g.constraint)} <span className="muted">{g.constraint.kind}</span>
                    </td>
                    <td className="num">{num(g.constraint.limit)}</td>
                    <td className="mono su-wrap">{g.constraint.name}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No constraints</Empty>
        )}
      </Section>
      </div>
      <Section title="Combine">
        <div className="su-card su-pad">
          <div className="su-node-row">
            <b className="su-kind">{combine.kind}</b>
            <Settings study={study} items={settings(combine, ["kind"])} />
          </div>
        </div>
      </Section>
    </>
  );
}

// ---- 2. the schedule tree --------------------------------------------------------------------------------------------

/** Which settings of a schedule kind name blocks (the framework's Nested and Alternate). */
const BLOCK_SETTINGS: Record<string, string[]> = { Nested: ["outer"], Alternate: ["blocks"] };

function plural(n: number, one: string, many: string): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}

function ScheduleView({ study, node, name }: { study: StudyModel; node: ScheduleNode; name: string | null }) {
  const scopes = [...study.scopes.values()].filter((s) => s.event.node === node.node);
  const runs = scopes.filter((s) => s.parent === null).length;
  const rounds = scopes.length - runs;
  const open = scopes.filter((s) => s.end === null).length;
  const candidates = scopes.reduce((sum, s) => sum + s.candidates.length, 0);
  const opened = [...study.scopes.values()].filter((s) => s.parent?.event.node === node.node).length;
  const children = Object.entries(node.children);
  // with one method copy per objective (Independent), each round records the objective it served
  const served = new Map<string, number>();
  for (const s of scopes) {
    const m = objectiveOf(study, s.event.objective);
    if (m !== null) served.set(m, (served.get(m) ?? 0) + 1);
  }
  const stats = [
    runs ? plural(runs, "run", "runs") : "",
    rounds ? plural(rounds, "round", "rounds") : "",
    scopes.length ? plural(candidates, "candidate", "candidates") : "",
    served.size ? `for ${[...served].map(([m, n]) => `${m}: ${n.toLocaleString()}`).join(", ")}` : "",
    opened ? `opened ${plural(opened, "round", "rounds")}` : "",
    open ? `${open.toLocaleString()} open` : "",
  ].filter(Boolean);
  return (
    <div className="su-node">
      <div className="su-node-row">
        {name !== null && <span className="su-child">{name}</span>}
        <b className="su-kind">{node.kind}</b>
        <Settings study={study} items={settings(node, ["node", "kind", "children"])} blocks={BLOCK_SETTINGS[node.kind]} />
        <span className="su-stats">{stats.length ? stats.join(" · ") : "no scopes"}</span>
      </div>
      {children.length > 0 && (
        <div className="su-children">
          {children.map(([k, child]) => (
            <ScheduleView key={k} study={study} node={child} name={k} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---- 3. blocks -----------------------------------------------------------------------------------------------------

function Blocks({ study }: { study: StudyModel }) {
  const names = [...study.blockOrder, ...Object.keys(study.h.blocks).filter((b) => !study.blockOrder.includes(b))];
  return (
    <div className="su-card">
      <table className="grid-table su-table">
        <thead>
          <tr>
            <th>Block</th>
            <th>Method</th>
            <th>Structural</th>
            <th>Variables</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name) => {
            const b = study.h.blocks[name];
            if (!b) return null;
            return (
              <tr key={name}>
                <td>
                  <BlockChip study={study} block={name} />
                </td>
                <td>
                  <div className="su-method">
                    <b className="su-kind">{b.method.kind}</b>
                    <Settings study={study} items={settings(b.method, ["kind"])} />
                  </div>
                </td>
                <td>{b.structural ? "yes" : <span className="muted">no</span>}</td>
                <td className="su-vars">
                  {b.keys.map((k) => (
                    <div key={k} className="mono su-wrap">
                      <Breakable text={k} />
                    </div>
                  ))}
                  {!b.keys.length && <span className="muted">none</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---- 4. the search space, as a tree --------------------------------------------------------------------------------

function optionText(v: unknown): string {
  return typeof v === "string" ? v : settingText(v);
}

function Domain({ s }: { s: SpaceRow }) {
  if (s.type === "float" || s.type === "int")
    return (
      <span className="tnum">
        [{num(s.low ?? null)}, {num(s.high ?? null)}]{s.log && <span className="su-flag">log</span>}
        {s.step !== null && s.step !== undefined && <span className="su-flag">step {num(s.step)}</span>}
      </span>
    );
  if (s.type === "text") return <span className="tnum">{String(s.default ?? "").length.toLocaleString()} chars</span>;
  return (
    <span className="su-options">
      {(s.choices ?? []).map((c, i) => (
        <span key={i} className="su-option">
          {optionText(c)}
        </span>
      ))}
    </span>
  );
}

type Line = { kind: "row"; row: SpaceRow; depth: number } | { kind: "option"; key: string; option: string; depth: number };

/** The space in tree order: every row whose active_when is null, then under each selector, each of its options
 * that has variables, and those variables (recursively); a row whose condition names no selector in the space stays
 * at the top, so every row appears once. */
function spaceLines(study: StudyModel): Line[] {
  const under = new Map<string, SpaceRow[]>();
  const top: SpaceRow[] = [];
  for (const s of study.space) {
    const at = s.active_when;
    if (at && study.spaceByKey.has(at.key)) {
      const k = JSON.stringify([at.key, at.option]);
      under.set(k, [...(under.get(k) ?? []), s]);
    } else top.push(s);
  }
  const out: Line[] = [];
  const seen = new Set<string>();
  const visit = (s: SpaceRow, depth: number) => {
    if (seen.has(s.key)) return;
    seen.add(s.key);
    out.push({ kind: "row", row: s, depth });
    if (!SELECTORS.has(s.type)) return;
    const options = (s.choices ?? []).map(String);
    const named = new Set(options);
    // options in the selector's order, then any condition naming an option the selector does not list
    const extra = [...under.keys()].map((k) => JSON.parse(k) as [string, string]).filter(([key, o]) => key === s.key && !named.has(o)).map(([, o]) => o);
    for (const option of [...options, ...extra]) {
      const rows = under.get(JSON.stringify([s.key, option]));
      if (!rows?.length) continue;
      out.push({ kind: "option", key: s.key, option, depth: depth + 1 });
      for (const r of rows) visit(r, depth + 2);
    }
  };
  for (const s of top) visit(s, 0);
  for (const s of study.space) visit(s, 0); // anything under a condition that never resolved (a cycle)
  return out;
}

function Space({ study }: { study: StudyModel }) {
  const lines = spaceLines(study);
  const indent = (depth: number) => ({ paddingLeft: 16 + depth * 14 });
  return (
    <div className="su-card">
      <table className="grid-table su-table su-space">
        <thead>
          <tr>
            <th>Key</th>
            <th>Type</th>
            <th>Domain</th>
            <th>Default</th>
            <th>Block</th>
          </tr>
        </thead>
        <tbody>
          {lines.map((l) => {
            if (l.kind === "option")
              return (
                <tr key={`${l.key}=${l.option}`} className="su-opt-row">
                  <td colSpan={5} style={indent(l.depth)}>
                    <span className="su-branch" />
                    <span className="mono">
                      <Breakable text={l.key} />
                    </span>{" "}
                    = <b>{l.option}</b>
                  </td>
                </tr>
              );
            const s = l.row;
            const block = study.blockOf.get(s.key);
            return (
              <tr key={s.key} className={SELECTORS.has(s.type) ? "su-selector" : ""}>
                <td className="mono su-wrap su-key" style={indent(l.depth)}>
                  <Breakable text={s.key} />
                </td>
                <td className="su-type">{s.type}</td>
                <td>
                  <Domain s={s} />
                </td>
                <td className="su-default">
                  {s.type === "text" ? <Long text={String(s.default ?? "")} limit={90} /> : <span className="tnum">{valueText(study, s, s.default)}</span>}
                </td>
                <td>{block ? <BlockChip study={study} block={block} /> : <span className="muted">–</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!lines.length && <div className="su-foot">No variables</div>}
    </div>
  );
}

// ---- 5. the experiment ---------------------------------------------------------------------------------------------

function KV({ k, children }: { k: string; children: ReactNode }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{children}</dd>
    </>
  );
}

function Experiment({ study }: { study: StudyModel }) {
  const h = study.h;
  const goalsOf = (m: string) => study.goals.filter((g) => g.metric === m);
  // the evaluator score a trial whose agent raised gets (null: left out)
  const failure = h.agent_error_score;
  return (
    <div className="su-card su-pad">
      <dl className="su-kv">
        <KV k="agent">
          <span className="mono su-wrap">{h.agent}</span>
        </KV>
        <KV k="cases">
          <span className="tnum">{h.cases.length.toLocaleString()}</span>
          <span className="su-cases">
            {h.cases.map((c) => (
              <span key={c} className="su-option mono">
                {c}
              </span>
            ))}
          </span>
        </KV>
        <KV k="repetitions">
          <span className="tnum">{num(h.repetitions)}</span>
        </KV>
        <KV k="estimate unit">{h.estimate_unit}</KV>
        <KV k="started">{when(h.started_at)}</KV>
        <KV k="runs">
          {study.record.runs.length ? (
            study.record.runs.map((r, i) => (
              <div key={i} className="tnum">
                {when(r.started_at)} · iterations {r.iterations.toLocaleString()}
              </div>
            ))
          ) : (
            <span className="muted">none</span>
          )}
        </KV>
      </dl>
      <table className="grid-table su-table su-metrics">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Role</th>
            <th>Range</th>
            {failure && <th>agent_error_score</th>}
            <th>Goals</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(h.metrics).map(([m, d]) => (
            <tr key={m}>
              <td className="mono su-wrap">
                <Breakable text={m} />
              </td>
              <td>{d.role}</td>
              <td className="tnum">{rangeText(d.range)}</td>
              {failure && (
                <td className="tnum">{m in failure ? failure[m] === null ? <span className="muted">left out</span> : num(failure[m]) : ""}</td>
              )}
              <td>
                <span className="su-goals">
                  {goalsOf(m).map((g) => (
                    <span key={`${g.role}${g.index}`} className={`su-goal ${g.role}`}>
                      {goalTitle(g)}
                    </span>
                  ))}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- the page ------------------------------------------------------------------------------------------------------

export function Setup({ study }: { study: StudyModel }) {
  return (
    <div className="setup">
      <Goals study={study} />
      <Section title="Schedule">
        <div className="su-card su-pad su-tree">
          <ScheduleView study={study} node={study.h.schedule} name={null} />
        </div>
      </Section>
      <Section title="Blocks" count={Object.keys(study.h.blocks).length}>
        <Blocks study={study} />
      </Section>
      <Section title="Search space" count={study.space.length}>
        <Space study={study} />
      </Section>
      <Section title="Experiment">
        <Experiment study={study} />
      </Section>
    </div>
  );
}
