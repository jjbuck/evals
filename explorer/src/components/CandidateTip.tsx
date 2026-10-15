// The one tooltip for a candidate, everywhere: what it is, where it ran, what it changed, and how it did on each goal.

import { num, words } from "../format";
import { goalTitle } from "../model/goals";
import type { StudyModel } from "../model/study";
import { Changed, Choices, EstimateText, SchedulePath } from "./common";

export function CandidateTip({ study, n }: { study: StudyModel; n: number }) {
  const t = study.byNumber.get(n);
  if (!t) return null;
  const measured = t.trials.filter((a) => a.scored).length;
  const lost = t.trials.find((a) => !a.scored);
  return (
    <div className="tip">
      <div className="tip-head">
        <b>Candidate {t.number}</b>
        {study.best.has(t.number) && <span className="tag">{study.bestIsSet ? "Pareto set" : "best"}</span>}
        {t.status !== "complete" && <span className="tag warn">{words(t.status)}</span>}
      </div>
      <div className="tip-sub">
        <SchedulePath study={study} t={t} />
      </div>
      <div className="tip-sub">
        <Changed study={study} t={t} />
      </div>
      <div className="tip-sub">
        <Choices study={study} t={t} />
      </div>
      {t.status === "invalid" && <div className="tip-sub bad">{t.reason}</div>}
      {t.status !== "invalid" && t.status !== "complete" && (
        <div className="tip-sub muted">
          {measured}/{t.trials.length} measured{lost ? ` · ${words(lost.status)}` : ""}
        </div>
      )}
      {study.objectives.length > 0 && t.status !== "invalid" && (
        <div className="tip-group">
          {study.objectives.map((g) => (
            <div key={`o${g.index}`} className="tip-row">
              <span>{goalTitle(g)}</span>
              <EstimateText e={study.est(t, g.metric)} />
            </div>
          ))}
        </div>
      )}
      {study.constraints.length > 0 && t.status !== "invalid" && (
        <div className="tip-group">
          {study.constraints.map((g) => {
            const v = study.violation(t, g.index);
            return (
              <div key={`c${g.index}`} className="tip-row">
                <span>{goalTitle(g)}</span>
                <span className="tnum">
                  <b>{num(study.mean(t, g.metric))}</b> {v === null ? "" : v <= 0 ? <span className="good">✓</span> : <span className="bad">✗</span>}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
