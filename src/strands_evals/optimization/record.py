"""A study's record on disk: what was searched, how the search was scheduled, and every candidate as it finishes.

    <directory>/
      study.jsonl              one JSON object per line (an event), in the order they happened
      transcripts/<hash>.json  each trial's messages, stored once by content hash

Events (``format: strands_evals.optimization``, schema 2):

- ``study``: the header. The agent factory and the ``task`` that runs each trial (with ``retries`` and ``redraws``);
  objectives; constraints (in order, each with its exact ``name``); ``combine``; the ``schedule`` tree (each node with
  its ``node`` id and ``children``); ``blocks`` (method, variables, structural) and ``block_order``; the ``space``;
  every metric with its role and range; cases and repetitions.
- ``run``: one ``optimize()`` call began (``iterations``, not counting the baseline the first call measures, and
  ``patience`` or null).
- ``scope``: a schedule began running: the root of a run, or a round (``round``, ``outer``) inside its ``parent``, with
  the blocks it may change, the objective it serves (Independent) and its candidate ``limit``.
- ``candidate``: a candidate finished: its values, status and estimates; ``scope`` and ``path`` (the scopes it ran in,
  outermost first), ``role`` (baseline | outer | step) and ``step``; what was ``proposed`` (blocks), ``drawn``
  (variables per block), ``inherited`` and ``defaulted``; ``violations`` (one per constraint, in order) and
  ``feasible``; which blocks it was given to (``shared``) and which wait for a round's result (``deferred``); what each
  method copy was ``told``; and ``best_after``, the best candidates once it finished.
- ``repeat``: a proposal (in ``scope``, with ``role`` and ``step``) repeated the values of ``candidate``, which was
  not measured again. The blocks in ``told`` learned that their draws scored what ``candidate`` measured (``values``;
  with ``Independent``, as ``objective``'s methods are told it). A step that repeats proposes again; a round whose
  opening proposal repeats opens on ``candidate``, and its outer draw is told the round's best later (``told``).
- ``told``: a deferred draw (a round's outer proposal, made in ``draw_candidate``) was told ``candidate``, the round's
  best
  (null: the round had no comparable candidate, and the draw failed).
- ``scope_end``: a scope finished, having run ``ran`` candidates over ``blocks`` (the blocks that existed in it); a
  round that opened on an earlier candidate does not count it.
- ``finished`` (the run ended, with ``best`` and its ``reason``: ``candidates``, ``patience``, or ``exhausted``: its
  schedule had nothing new to propose) or ``stopped`` (it raised, with ``error``).

Values are plain JSON; transcripts are referenced by hash, never inlined, so the record stays small and readable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .projection import constraint_name, describe_goal
from .trials import SCORED, USAGE

if TYPE_CHECKING:
    from .study import Candidate, Scope, Study

FORMAT = "strands_evals.optimization"
SCHEMA = 2


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _import_path(obj: Any) -> str:
    """Where a function, or a callable object's class, is defined."""
    named = obj if hasattr(obj, "__qualname__") else type(obj)
    return f"{named.__module__}.{named.__qualname__}"


def describe_schedule(schedule: Any, study: "Study", node: str = "") -> dict[str, Any]:
    children = schedule.children() if hasattr(schedule, "children") else {}
    describe = schedule.describe(study) if hasattr(schedule, "describe") else {"kind": type(schedule).__name__}
    return {
        "node": node,
        **describe,
        "children": {
            name: describe_schedule(child, study, f"{node}.{name}" if node else name)
            for name, child in children.items()
        },
    }


class Record:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "study.jsonl"
        if self.path.exists() and self.path.stat().st_size > 0:
            raise FileExistsError(f"{self.path} already holds a study; give each study its own record directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "transcripts").mkdir(exist_ok=True)

    def _write(self, event: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _transcript(self, messages: Any) -> dict[str, str] | None:
        if not messages:
            return None
        text = json.dumps(_plain(list(messages)), indent=1)
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        path = self.directory / "transcripts" / f"{digest}.json"
        if not path.exists():
            path.write_text(text)
        return {"hash": digest, "format": "strands-messages"}

    def study(self, study: "Study") -> None:
        p = study.projection
        metrics = sorted(p.evaluators | set(USAGE) | set(p.derived) | set(p.goal_metrics))
        self._write(
            {
                "type": "study",
                "format": FORMAT,
                "schema": SCHEMA,
                "started_at": _now(),
                "agent": study.agent.import_path,
                "task": _import_path(study.task),
                "retries": study.retries,
                "redraws": study.search.redraws,
                "objectives": [describe_goal(o) for o in p.objectives],
                "constraints": [{**describe_goal(c), "name": constraint_name(c)} for c in p.constraints],
                "combine": {
                    **study.combine.describe(),
                    "best_is_set": study.combine.best_is_set(len(p.objectives)),
                    "copies": len(study.methods),
                    "told": [study.combine.told_names(p, k) for k in range(len(study.methods))],
                },
                "schedule": describe_schedule(study.schedule, study),
                "blocks": {
                    name: {"method": b.method.describe(), "keys": list(b.keys), "structural": b.structural}
                    for name, b in study.blocks.items()
                },
                "block_order": study.block_order(),
                "space": [info.to_dict() for info in study.params.values()],
                "metrics": {m: {"role": p.role(m), "range": list(p.range(m))} for m in metrics},
                "cases": [c.name for c in study.experiment.cases],
                "repetitions": study.repetitions,
                "estimate_unit": "case" if len(study.experiment.cases) >= 2 else "trial",
                "agent_error_score": {m: p.failure_value(m) for m in sorted(p.evaluators)},
            }
        )

    def run(self, iterations: int, patience: int | None = None) -> None:
        self._write({"type": "run", "started_at": _now(), "iterations": iterations, "patience": patience})

    def scope(self, scope: "Scope", id: int) -> None:
        self._write(
            {
                "type": "scope",
                "id": id,
                "parent": scope.parent.id if scope.parent else None,
                "node": scope.node,
                "kind": scope.kind,
                "blocks": list(scope.blocks),
                "objective": scope.objective if len(scope.study.methods) > 1 else None,
                "round": scope.round,
                "outer": scope.outer,
                "limit": scope.limit,
                "opened_at": _now(),
            }
        )

    def scope_end(self, scope: "Scope") -> None:
        self._write(
            {
                "type": "scope_end",
                "id": scope.id,
                "ran": sum(scope.id in c.path for c in scope.candidates),  # not a candidate a round opened on again
                "blocks": list(scope.blocks),
                "closed_at": _now(),
            }
        )

    def _told(self, study: "Study", candidate: "Candidate", k: int) -> list[float] | None:
        if not candidate.eligible:
            return None
        return list(study.combine.told(study.projection, candidate, k, study.baseline))

    def candidate(
        self, study: "Study", candidate: "Candidate", *, shared: Sequence[str], deferred: Sequence[str]
    ) -> None:
        self._write(
            {
                "type": "candidate",
                "finished_at": _now(),
                "number": candidate.number,
                "params": candidate.params,
                "status": candidate.status,
                "reason": candidate.reason,
                "eligible": candidate.eligible,
                "feasible": candidate.feasible,
                "violations": list(candidate.violations),
                "scope": candidate.scope,
                "path": list(candidate.path),
                "role": candidate.role,
                "step": candidate.step,
                "proposed": list(candidate.proposed),
                "drawn": {b: list(ks) for b, ks in candidate.drawn.items()},
                "inherited": dict(candidate.inherited),
                "defaulted": list(candidate.defaulted),
                "base": candidate.base,
                "objective": candidate.objective,
                "origin": _plain(candidate.origin),
                "shared": list(shared),
                "deferred": list(deferred),
                "told": [self._told(study, candidate, k) for k in range(len(study.methods))],
                "best_after": [c.number for c in study.best_candidates],
                "estimates": {
                    m: {"mean": e.mean, "ci": list(e.ci) if e.ci else None, "n": e.n}
                    for m, e in candidate.estimates.items()
                },
                "trials": [
                    {
                        "case": t.case,
                        "repetition": t.repetition,
                        "status": t.status.value,
                        "scored": t.status in SCORED,
                        "metrics": dict(t.metrics),
                        "unavailable": sorted(t.unavailable),
                        "error": t.error,
                        "output": t.output,
                        "reasons": dict(t.reasons),
                        "transcript": self._transcript(t.messages),
                    }
                    for t in candidate.trials
                ],
            }
        )

    def told(
        self, study: "Study", scope: "Scope", block: str, draw_candidate: int, candidate: "Candidate | None", k: int
    ) -> None:
        self._write(
            {
                "type": "told",
                "scope": scope.id,
                "block": block,
                "objective": k if len(study.methods) > 1 else None,
                "draw_candidate": draw_candidate,
                "candidate": candidate.number if candidate is not None else None,
                "values": self._told(study, candidate, k) if candidate is not None else None,
            }
        )

    def repeat(
        self,
        study: "Study",
        candidate: "Candidate",
        draws: dict[str, Any],
        drawn: dict[str, Sequence[str]],
        *,
        scope: "Scope | None",
        role: str,
        step: int | None,
        objective: int | None,
        told: Sequence[str],
    ) -> None:
        k = objective or 0
        self._write(
            {
                "type": "repeat",
                "repeated_at": _now(),
                "scope": scope.id if scope else None,
                "path": list(scope.path) if scope else [],
                "role": role,
                "step": step,
                "candidate": candidate.number,
                "proposed": list(draws),
                "drawn": {b: list(ks) for b, ks in drawn.items()},
                "origin": _plain({name: draw.origin for name, draw in draws.items()}),
                "objective": objective if len(study.methods) > 1 else None,
                "told": list(told),
                "values": self._told(study, candidate, k) if told else None,
            }
        )

    def finish(self, study: "Study", reason: str = "candidates") -> None:
        """``reason``: ``candidates`` (every candidate asked for ran), ``patience`` (the best stopped changing) or
        ``exhausted`` (the policy had nothing new to propose)."""
        self._write(
            {
                "type": "finished",
                "finished_at": _now(),
                "best": [c.number for c in study.best_candidates],
                "reason": reason,
            }
        )

    def stopped(self, error: str) -> None:
        self._write({"type": "stopped", "stopped_at": _now(), "error": error[:2000]})
