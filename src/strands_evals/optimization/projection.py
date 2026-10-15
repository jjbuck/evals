"""From trials to what the optimizer sees: metric values, case-level estimates, objectives and constraints, and how
several objectives combine.

The case is the unit: repetitions are averaged within a case first (with a single case, the trial is the unit).
Every estimate is a mean with a 95% t-interval, clipped to the metric's range.

A metric is an evaluator's score (in 0 to 1 unless the evaluator says otherwise, with ``score_range``), the agent's
usage (``tokens``, ``seconds``, ...: 0 or more), or ``Derived`` from a trial's other metrics.

Several objectives are given wrapped in how they combine, which decides what each method is told, which candidate the
policy holds, and what "best" means:

- ``Pareto(...)`` (the objectives optimized jointly): methods that take a vector get every objective; methods that need
  one number get the weighted mean of normalized scores. The held candidate is the Pareto member best on that mean;
  best is the Pareto set. A single objective is a Pareto of one.
- ``Independent(...)``: one copy of every method per objective, each told only its own objective and holding its own
  best candidate; the copies take turns proposing and all see every result. Best is the Pareto set.
- ``WeightedSum(...)`` / ``Chebyshev(...)``: one number for every method, the weighted sum (or the weakest) of each
  objective's improvement over the baseline in units of ``scale``; best is the best number.
- ``Lexicographic(...)``: the first objective, ties broken by the next within each objective's ``tolerance``.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from .trials import SCORED, USAGE, Trial, TrialStatus

if TYPE_CHECKING:
    from .study import Candidate

Range = tuple[float | None, float | None]

# ---- interval helpers (from feat/study:optimization/estimate.py) ----------------------------------------------------
_T975 = [
    (1, 12.706),
    (2, 4.303),
    (3, 3.182),
    (4, 2.776),
    (5, 2.571),
    (6, 2.447),
    (7, 2.365),
    (8, 2.306),
    (9, 2.262),
    (10, 2.228),
    (12, 2.179),
    (15, 2.131),
    (20, 2.086),
    (25, 2.060),
    (30, 2.042),
    (40, 2.021),
    (60, 2.000),
    (120, 1.980),
]


def t975(df: int) -> float:
    for d, t in _T975:
        if df <= d:
            return t
    return 1.960


def summary(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(xs)
    if n == 0:
        return {"mean": None, "se": None, "sd": None, "n": 0, "ci": None}
    mean = statistics.fmean(xs)
    if n < 2:
        return {"mean": mean, "se": None, "sd": None, "n": 1, "ci": None}
    sd = statistics.stdev(xs)
    se = sd / math.sqrt(n)
    h = t975(n - 1) * se
    return {"mean": mean, "se": se, "sd": sd, "n": n, "ci": [mean - h, mean + h]}


def clip(stat: dict[str, Any], lo: float | None, hi: float | None) -> dict[str, Any]:
    if stat.get("ci") is None:
        return stat
    a, b = stat["ci"]
    return {**stat, "ci": [a if lo is None else max(lo, a), b if hi is None else min(hi, b)]}


def bounds(range_: Range, compare: str) -> Range:
    """Where an estimate can lie, given where the metric's values lie."""
    lo, hi = range_
    if compare == "absolute":
        return range_
    if compare == "difference":
        return (lo - hi, hi - lo) if lo is not None and hi is not None else (None, None)
    return (0.0, None) if lo is not None and lo >= 0 else (None, None)


# ---- metrics, objectives and constraints ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Derived:
    """A metric computed from each trial's other metrics (its evaluators' scores and its usage, by name), then estimated
    like any other: ``Derived("cost", lambda m: m["input_tokens"] * 3e-6 + m["output_tokens"] * 15e-6)``. Use it in
    objectives and constraints in place of a metric's name. ``range``: where its values lie (None for unbounded)."""

    name: str
    fn: Callable[[Mapping[str, float]], float] = field(compare=False, repr=False)
    range: Range = (None, None)


def _named(goal: Any) -> None:
    """A goal given a ``Derived`` metric keeps its name as ``metric`` and the metric itself as ``derived``."""
    if isinstance(goal.metric, Derived):
        object.__setattr__(goal, "derived", goal.metric)
        object.__setattr__(goal, "metric", goal.metric.name)


@dataclass(frozen=True)
class Maximize:
    metric: str
    weight: float = 1.0  # WeightedSum, Chebyshev, and Pareto's single number
    scale: float | None = None  # a meaningful change, so objectives in different units combine (default: the range)
    tolerance: float = 0.0  # Lexicographic: differences up to this are ties
    derived: Derived | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _named(self)


@dataclass(frozen=True)
class Minimize:
    metric: str
    weight: float = 1.0
    scale: float | None = None
    tolerance: float = 0.0
    derived: Derived | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _named(self)


@dataclass(frozen=True)
class AtMost:
    metric: str
    limit: float
    derived: Derived | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _named(self)


@dataclass(frozen=True)
class AtLeast:
    metric: str
    limit: float
    derived: Derived | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _named(self)


def describe_goal(goal: Any) -> dict[str, Any]:
    """A goal as the record states it (a derived metric is described with the metrics, not here)."""
    fields = {f.name: getattr(goal, f.name) for f in dataclasses.fields(goal) if f.name != "derived"}
    return {"kind": type(goal).__name__, **fields}


Objective = Maximize | Minimize
Constraint = AtMost | AtLeast


def constraint_name(c: Constraint) -> str:
    """The constraint as written, exactly: ``tokens <= 30000``."""
    return f"{c.metric} {'<=' if isinstance(c, AtMost) else '>='} {c.limit!r}"


def violation(c: Constraint, value: float) -> float:
    """<= 0 when the constraint is met."""
    return value - c.limit if isinstance(c, AtMost) else c.limit - value


@dataclass(frozen=True)
class Estimate:
    mean: float | None
    ci: tuple[float, float] | None
    n: int  # the number of units (cases, or trials with a single case)


class Projection:
    """How trials become values, estimates, objective values and constraint violations."""

    def __init__(
        self,
        objectives: Sequence[Objective],
        constraints: Sequence[Constraint],
        *,
        evaluators: Mapping[str, Range],
        agent_error_score: float | str | None = "worst",
    ) -> None:
        """``evaluators``: each evaluator's metric and where its scores lie."""
        self.objectives, self.constraints = list(objectives), list(constraints)
        self.evaluators, self.agent_error_score = set(evaluators), agent_error_score
        self.derived = {g.derived.name: g.derived for g in [*self.objectives, *self.constraints] if g.derived}
        taken = sorted(set(self.derived) & (self.evaluators | set(USAGE)))
        if taken:
            raise ValueError(f"a derived metric has the name of a measured one: {taken}")
        self.ranges = {**dict(evaluators), **{name: d.range for name, d in self.derived.items()}}
        self.goal_metrics = list(
            dict.fromkeys([o.metric for o in self.objectives] + [c.metric for c in self.constraints])
        )

    @property
    def directions(self) -> list[str]:
        return ["maximize" if isinstance(o, Maximize) else "minimize" for o in self.objectives]

    def role(self, metric: str) -> str:
        return "derived" if metric in self.derived else "evaluator" if metric in self.evaluators else "usage"

    def range(self, metric: str) -> Range:
        if metric in self.ranges:
            return self.ranges[metric]
        return (0.0, None) if metric in USAGE else (0.0, 1.0)

    def failure_value(self, metric: str) -> float | None:
        """The evaluator score a trial whose agent raised gets: ``agent_error_score`` when it is a number; with
        ``"worst"``, the end of the metric's range that is worst for the goals that use it (None when that end is
        unbounded, the goals disagree, or no goal uses the metric); None leaves failures out."""
        if self.agent_error_score is None or isinstance(self.agent_error_score, (int, float)):
            return self.agent_error_score
        wants = {
            ("max" if isinstance(g, (Maximize, AtLeast)) else "min")
            for g in [*self.objectives, *self.constraints]
            if g.metric == metric
        }
        lo, hi = self.range(metric)
        if wants == {"max"}:
            return lo
        if wants == {"min"}:
            return hi
        return None

    def value(self, trial: Trial, metric: str) -> float | None:
        """A trial's value of a metric: measured, the failure policy's value, or none."""
        if trial.status not in SCORED:
            return None
        if trial.status == TrialStatus.AGENT_FAILURE and metric in self.evaluators:
            return self.failure_value(metric)
        v = trial.metrics.get(metric)
        return None if v is None else float(v)

    def prepare(self, trial: Trial) -> Trial:
        """The trial as the study counts it: its derived metrics added, and every value outside its metric's range taken
        out (unavailable, with the reason); a trial left with no evaluator score measured nothing
        (``VERIFIER_INVALID``)."""
        trial = self._derive(trial)
        outside = {}
        for metric, v in trial.metrics.items():
            lo, hi = self.range(metric)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                where = (
                    "give the evaluator a score_range = (low, high)"
                    if metric in self.evaluators
                    else "give the Derived metric a range=(low, high)"
                )
                outside[metric] = f"{metric}={v!r} is outside its range [{lo}, {hi}]; {where} (None for unbounded)"
        if not outside:
            return trial
        metrics = {m: v for m, v in trial.metrics.items() if m not in outside}
        unavailable = trial.unavailable | set(outside)
        status = trial.status
        if status == TrialStatus.COMPLETED and self.evaluators and self.evaluators <= unavailable:
            status = TrialStatus.VERIFIER_INVALID
        error = "; ".join(filter(None, [trial.error, *outside.values()]))
        return dataclasses.replace(
            trial, status=status, metrics=metrics, unavailable=frozenset(unavailable), error=error
        )

    def _derive(self, trial: Trial) -> Trial:
        """Each derived metric from the trial's metrics. For an agent that raised, its scores are the failure values,
        and a derived metric that needs a score with none counts as the failure value of its own; otherwise a derived
        metric that needs a metric the trial lacks is unavailable."""
        if not self.derived or trial.status not in SCORED:
            return trial
        inputs = dict(trial.metrics)
        if trial.status == TrialStatus.AGENT_FAILURE:
            for e in self.evaluators:
                if (v := self.failure_value(e)) is not None:
                    inputs.setdefault(e, v)
        metrics, unavailable, missing = dict(trial.metrics), set(trial.unavailable), []
        for name, d in self.derived.items():
            try:
                metrics[name] = inputs[name] = float(d.fn(MappingProxyType(inputs)))
            except KeyError as exc:
                if trial.status == TrialStatus.AGENT_FAILURE and (v := self.failure_value(name)) is not None:
                    metrics[name] = inputs[name] = v  # what it needs was never scored: it counts as its own worst
                    continue
                unavailable.add(name)
                missing.append(f"{name} needs {exc.args[0]!r}, which this trial lacks (it has {sorted(inputs)})")
        error = "; ".join(filter(None, [trial.error, *missing])) or None
        return dataclasses.replace(trial, metrics=metrics, unavailable=frozenset(unavailable), error=error)

    def case_means(self, trials: Sequence[Trial], metric: str) -> dict[str, float]:
        groups: dict[str, list[float]] = defaultdict(list)
        for a in trials:
            v = self.value(a, metric)
            if v is not None:
                groups[a.case].append(v)
        return {case: statistics.fmean(vs) for case, vs in groups.items()}

    def estimates(self, trials: Sequence[Trial]) -> dict[str, Estimate]:
        metrics = sorted(
            {m for a in trials if a.status in SCORED for m in a.metrics}
            | ({m for m in self.evaluators} if any(a.status == TrialStatus.AGENT_FAILURE for a in trials) else set())
        )
        by_case = len({a.case for a in trials}) >= 2
        out = {}
        for metric in metrics:
            if by_case:
                units = list(self.case_means(trials, metric).values())
            else:
                units = [v for a in trials if (v := self.value(a, metric)) is not None]
            s = clip(summary(units), *self.range(metric))
            out[metric] = Estimate(mean=s["mean"], ci=tuple(s["ci"]) if s["ci"] else None, n=s["n"])
        return out

    def eligible(self, trials: Sequence[Trial]) -> bool:
        """Every case has a value for every objective and constraint metric."""
        cases = {a.case for a in trials}
        return bool(cases) and all(set(self.case_means(trials, m)) == cases for m in self.goal_metrics)

    def values(self, candidate: "Candidate") -> list[float]:
        return [candidate.metrics[o.metric] for o in self.objectives]

    def violations(self, metrics: dict[str, float | None]) -> tuple[float | None, ...]:
        """One per constraint, in order: <= 0 when met, None when its metric was not measured."""
        return tuple(
            violation(c, metrics[c.metric]) if metrics.get(c.metric) is not None else None for c in self.constraints
        )

    def oriented(self, candidate: "Candidate") -> list[float]:
        """Objective values, every one oriented so that larger is better."""
        return [v if d == "maximize" else -v for v, d in zip(self.values(candidate), self.directions, strict=True)]

    def scale(self, objective: Objective) -> float:
        if objective.scale is not None:
            return objective.scale
        lo, hi = self.range(objective.metric)
        if lo is None or hi is None or hi <= lo:
            raise ValueError(
                f"{objective.metric} has no bounded range: give the objective a scale= (the size of a "
                "meaningful change) so it can be combined with the others"
            )
        return hi - lo

    # ---- per-trial rewards for methods that learn from cases (reflection) -------------------------------------
    def scores(self, candidate: "Candidate") -> list[tuple[Trial, list[float | None]]]:
        """Each scored trial's objective values normalized to [0, 1], oriented so higher is better: by the metric's
        range when it is bounded, else by its mid-rank among the candidate's trials."""
        scored = [a for a in candidate.trials if a.status in SCORED]
        columns = []
        for o, direction in zip(self.objectives, self.directions, strict=True):
            values = [self.value(a, o.metric) for a in scored]
            lo, hi = self.range(o.metric)
            known = [v for v in values if v is not None]
            column: list[float | None] = []
            for v in values:
                if v is None:
                    column.append(None)
                elif lo is not None and hi is not None and hi > lo:
                    s = (v - lo) / (hi - lo)
                    column.append(s if direction == "maximize" else 1.0 - s)
                elif len(known) < 2 or len(set(known)) == 1:
                    column.append(0.5)
                else:
                    better = sum(1 for w in known if (w < v if direction == "maximize" else w > v))
                    equal = sum(1 for w in known if w == v) - 1
                    column.append((better + 0.5 * equal) / (len(known) - 1))
            columns.append(column)
        return [(a, [col[i] for col in columns]) for i, a in enumerate(scored)]

    def rewards(self, candidate: "Candidate", combine: "Combine", k: int | None) -> list[tuple[Trial, float, dict]]:
        """One reward per scored trial, in [0, 1]: the combined normalized objectives, halved when the trial
        breaks a constraint, 0 when the agent failed."""
        out = []
        for trial, scores in self.scores(candidate):
            broken = {
                constraint_name(c): violation(c, v)
                for c in self.constraints
                if (v := self.value(trial, c.metric)) is not None and violation(c, v) > 0
            }
            if trial.status == TrialStatus.AGENT_FAILURE:
                reward = 0.0
            else:
                reward = combine.scalar([s if s is not None else 0.0 for s in scores], self.objectives, k)
                if broken:
                    reward *= 0.5
            out.append(
                (
                    trial,
                    reward,
                    {
                        "status": trial.status.value,
                        "error": trial.error,
                        "metrics": dict(trial.metrics),
                        "reasons": dict(trial.reasons),
                        "objectives": {o.metric: self.value(trial, o.metric) for o in self.objectives},
                        "constraints_broken": broken,
                    },
                )
            )
        return out


# ---- combining several objectives ----------------------------------------------------------------------------------
def _excess(candidate: "Candidate") -> float:
    return sum(max(v, 0.0) for v in candidate.violations if v is not None)


def _feasible_first(candidates: Sequence["Candidate"]) -> list["Candidate"]:
    eligible = [t for t in candidates if t.eligible]
    feasible = [t for t in eligible if t.feasible]
    if feasible:
        return feasible
    if not eligible:
        return []
    least = min(_excess(t) for t in eligible)
    return [t for t in eligible if _excess(t) == least]


def pareto(projection: Projection, candidates: Sequence["Candidate"]) -> list["Candidate"]:
    def dominates(a: "Candidate", b: "Candidate") -> bool:
        x, y = projection.oriented(a), projection.oriented(b)
        return all(p >= q for p, q in zip(x, y, strict=True)) and x != y

    return [t for t in candidates if not any(dominates(o, t) for o in candidates)]


class Combine:
    """How several objectives combine; built with the objectives: ``Pareto(Maximize("answer"), Minimize("tokens"))``."""

    def __init__(self, *objectives: Objective) -> None:
        if not objectives:
            raise ValueError(f"{type(self).__name__}() needs at least one objective")
        wrong = [o for o in objectives if not isinstance(o, (Maximize, Minimize))]
        if wrong:
            raise TypeError(f"{type(self).__name__} takes objectives (Maximize, Minimize), not {wrong}")
        self.objectives = tuple(objectives)

    def settings(self) -> dict[str, Any]:
        """The combination's own settings, for the record."""
        return {}

    def instances(self, n_objectives: int) -> int:
        """How many copies of each method the study keeps (one per objective for Independent)."""
        return 1

    def best_is_set(self, n_objectives: int) -> bool:
        """Whether "best" is a set of candidates (a Pareto set) rather than one candidate."""
        return False

    def check(self, projection: Projection) -> None:
        """Raise if this combination cannot be computed for these objectives (before anything runs)."""

    def describe(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, **self.settings()}

    def told_names(self, projection: Projection, k: int | None) -> list[str]:
        """What each number ``told`` gives is: an objective's metric, or this combination's own name."""
        return [o.metric for o in projection.objectives]

    def directions(self, projection: Projection, k: int | None) -> list[str]:
        raise NotImplementedError

    def told(self, projection: Projection, candidate: "Candidate", k: int | None, baseline: "Candidate") -> list[float]:
        raise NotImplementedError

    def held(
        self, projection: Projection, pool: Sequence["Candidate"], k: int | None, baseline: "Candidate"
    ) -> "Candidate | None":
        raise NotImplementedError

    def best(
        self, projection: Projection, candidates: Sequence["Candidate"], baseline: "Candidate"
    ) -> list["Candidate"]:
        raise NotImplementedError

    def scalar(self, scores: list[float], objectives: Sequence[Objective], k: int | None) -> float:
        """One number from per-objective scores in [0, 1] (for methods that need one)."""
        raise NotImplementedError


def _unit(projection: Projection, objective: Objective, baseline: "Candidate") -> float:
    """The size of a meaningful change in an objective: its ``scale``, else its bounded range, else the baseline's
    magnitude (so the same candidate is always measured the same way, whatever it is compared with)."""
    if objective.scale is not None:
        return objective.scale
    lo, hi = projection.range(objective.metric)
    if lo is not None and hi is not None and hi > lo:
        return hi - lo
    return abs(baseline.metrics.get(objective.metric, 0.0)) or 1.0


class Pareto(Combine):
    def __init__(self, *objectives: Objective, weights: Sequence[float] | None = None) -> None:
        """``weights``: for methods that need one number (default: each objective's own ``weight``)."""
        super().__init__(*objectives)
        if weights is not None and len(weights) != len(objectives):
            raise ValueError(f"Pareto has {len(objectives)} objectives but {len(weights)} weights")
        self.weights = tuple(weights) if weights is not None else None

    def settings(self) -> dict[str, Any]:
        return {"weights": list(self.weights) if self.weights else None}

    def _weights(self, objectives: Sequence[Objective]) -> list[float]:
        return list(self.weights) if self.weights else [o.weight for o in objectives]

    def best_is_set(self, n_objectives):
        return n_objectives > 1

    def directions(self, projection, k):
        return projection.directions

    def told(self, projection, candidate, k, baseline):
        return projection.values(candidate)

    def held(self, projection, pool, k, baseline):
        """The Pareto member with the largest weighted gain over the baseline, each objective in its own unit."""
        front = pareto(projection, _feasible_first(pool))
        if not front:
            return None
        weights = self._weights(projection.objectives)
        units = [_unit(projection, o, baseline) for o in projection.objectives]
        base = projection.oriented(baseline)

        def gain(t: "Candidate") -> float:
            return sum(w * (x - b) / u for w, x, b, u in zip(weights, projection.oriented(t), base, units, strict=True))

        return max(front, key=lambda t: (gain(t), projection.oriented(t)))

    def best(self, projection, candidates, baseline):
        feasible = [t for t in candidates if t.eligible and t.feasible]
        return pareto(projection, feasible)

    def scalar(self, scores, objectives, k):
        weights = self._weights(objectives)
        return sum(w * s for w, s in zip(weights, scores, strict=True)) / sum(weights)


class Independent(Combine):
    def instances(self, n_objectives):
        return n_objectives

    def best_is_set(self, n_objectives):
        return n_objectives > 1

    def told_names(self, projection, k):
        return [projection.objectives[k or 0].metric]

    def directions(self, projection, k):
        return [projection.directions[k or 0]]

    def told(self, projection, candidate, k, baseline):
        return [projection.values(candidate)[k or 0]]

    def held(self, projection, pool, k, baseline):
        candidates = _feasible_first(pool)
        i = k or 0
        return max(candidates, key=lambda t: projection.oriented(t)[i]) if candidates else None

    def best(self, projection, candidates, baseline):
        return pareto(projection, [t for t in candidates if t.eligible and t.feasible])

    def scalar(self, scores, objectives, k):
        return scores[k or 0]


class WeightedSum(Combine):
    def check(self, projection):
        for o in projection.objectives:
            projection.scale(o)

    def told_names(self, projection, k):
        return [type(self).__name__]

    def directions(self, projection, k):
        return ["maximize"]

    def _gains(self, projection, candidate, baseline):
        return [
            o.weight * (x - b) / projection.scale(o)
            for o, x, b in zip(
                projection.objectives, projection.oriented(candidate), projection.oriented(baseline), strict=True
            )
        ]

    def told(self, projection, candidate, k, baseline):
        return [sum(self._gains(projection, candidate, baseline))]

    def held(self, projection, pool, k, baseline):
        candidates = _feasible_first(pool)
        return max(candidates, key=lambda t: self.told(projection, t, k, baseline)[0]) if candidates else None

    def best(self, projection, candidates, baseline):
        feasible = [t for t in candidates if t.eligible and t.feasible]
        return [self.held(projection, feasible, None, baseline)] if feasible else []

    def scalar(self, scores, objectives, k):
        return sum(o.weight * s for o, s in zip(objectives, scores, strict=True)) / sum(o.weight for o in objectives)


class Chebyshev(WeightedSum):
    def told(self, projection, candidate, k, baseline):
        return [min(self._gains(projection, candidate, baseline))]

    def scalar(self, scores, objectives, k):
        return min(scores)


class Lexicographic(Combine):
    def told_names(self, projection, k):
        return [projection.objectives[0].metric]

    def directions(self, projection, k):
        return [projection.directions[0]]

    def told(self, projection, candidate, k, baseline):
        return [projection.values(candidate)[0]]

    def held(self, projection, pool, k, baseline):
        candidates = _feasible_first(pool)
        for i, o in enumerate(projection.objectives):
            if len(candidates) <= 1:
                break
            top = max(projection.oriented(t)[i] for t in candidates)
            candidates = [t for t in candidates if projection.oriented(t)[i] >= top - o.tolerance]
        return candidates[0] if candidates else None

    def best(self, projection, candidates, baseline):
        feasible = [t for t in candidates if t.eligible and t.feasible]
        return [self.held(projection, feasible, None, baseline)] if feasible else []

    def scalar(self, scores, objectives, k):
        return scores[0]
