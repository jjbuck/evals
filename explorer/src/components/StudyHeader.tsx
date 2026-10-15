// The study's name, state, goals (objectives, then constraints, always both) and the tabs.

import { ago, num } from "../format";
import { arrow, op } from "../model/goals";
import type { StudyModel } from "../model/study";
import { href, type Route } from "../router";
import { Pill } from "./common";

export function StudyHeader({ study, route }: { study: StudyModel; route: Route }) {
  const tabs: { page: "overview" | "progress" | "tradeoff" | "space" | "candidates" | "setup"; label: string }[] = [
    { page: "overview", label: "Overview" },
    { page: "progress", label: "Progress" },
    ...(study.multi ? [{ page: "tradeoff" as const, label: "Trade-off" }] : []),
    { page: "space", label: "Search space" },
    { page: "candidates", label: "Candidates" },
    { page: "setup", label: "Setup" },
  ];
  const active = route.page === "candidate" || route.page === "trial" ? "candidates" : route.page;
  const last = study.candidates.at(-1)?.finished_at;
  const end = study.record.end;
  return (
    <header className="study-head">
      <div className="title-row">
        <h1>{study.name}</h1>
        {study.state === "running" && <Pill kind="running">running{last ? ` · last candidate ${ago(last)}` : ""}</Pill>}
        {study.state === "finished" && (
          <Pill kind="finished">
            finished
            {end?.type === "finished" && end.reason === "patience" ? ` · patience ${study.record.runs.at(-1)?.patience ?? ""}` : ""}
          </Pill>
        )}
        {study.state === "stopped" && end?.type === "stopped" && (
          <Pill kind="stopped" title={end.error}>
            stopped: {end.error.length > 80 ? `${end.error.slice(0, 80)}…` : end.error}
          </Pill>
        )}
      </div>
      <div className="goal-row">
        <div className="goal-group">
          <span className="role objective">objectives</span>
          {study.h.objectives.map((o, k) => (
            <span key={k} className="goal-chip">
              {arrow(o)} <b>{o.metric}</b>
            </span>
          ))}
          {study.multi && <span className="qualifier">{study.h.combine.kind}</span>}
        </div>
        <div className="goal-group">
          <span className="role constraint">constraints</span>
          {study.h.constraints.map((c, k) => (
            <span key={k} className="goal-chip">
              <b>{c.metric}</b> {op(c)} {num(c.limit)}
            </span>
          ))}
          {study.h.constraints.length === 0 && <span className="qualifier">none</span>}
        </div>
      </div>
      <nav className="tabs">
        {tabs.map((t) => (
          <a key={t.page} href={href(study.id, { page: t.page })} className={active === t.page ? "on" : ""}>
            {t.label}
          </a>
        ))}
      </nav>
    </header>
  );
}
