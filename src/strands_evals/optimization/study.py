"""The study: candidates, the incumbent and the best candidates, and building, validating and measuring one
candidate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .methods import SELECTORS, Block, Draw, resolve_blocks
from .projection import (
    AtLeast,
    AtMost,
    Combine,
    Estimate,
    Maximize,
    Minimize,
    Pareto,
    Projection,
    Range,
    constraint_name,
)
from .schedules import Schedule, default_schedule
from .search import Search
from .space import TunableFactory, Values, tunable
from .trials import SCORED, USAGE, Task, Trial, invoke, run_trials


@dataclass(frozen=True)
class Candidate:
    number: int
    params: dict[str, Any]  # the complete candidate: exactly the variables that exist under its choices
    status: str  # complete | partial (some trial or value missing) | failed (nothing measured) | invalid (not run)
    trials: tuple[Trial, ...] = field(repr=False)
    estimates: dict[str, Estimate] = field(repr=False)
    eligible: bool  # every case has every objective and constraint metric: the candidate can be held and told
    violations: tuple[float | None, ...]  # one per constraint, in order: <= 0 when met (None: not measured)
    proposed: tuple[str, ...]  # the blocks whose methods proposed values in this candidate (none for the baseline)
    base: int | None  # the candidate whose values were held for everything not proposed
    objective: int | None  # with Independent, the objective whose methods proposed it
    origin: dict[str, dict[str, Any]]  # per proposed block: which method proposed its values, and how
    drawn: dict[str, tuple[str, ...]] = field(default_factory=dict)  # per proposed block, the variables it gave
    inherited: dict[str, int] = field(
        default_factory=dict
    )  # variables ``base`` lacks, from the best candidate with them
    defaulted: tuple[str, ...] = ()  # variables no candidate had yet: at their defaults
    scope: int | None = None  # the scope (schedule run or round) it ran in; None for the baseline
    path: tuple[int, ...] = ()  # the scopes it ran in, outermost first
    role: str = "step"  # baseline | outer (opens a round) | step
    step: int | None = None  # its index in its schedule's run, where the schedule counts them
    reason: str | None = None  # why the candidate is invalid
    _factory: TunableFactory | None = field(default=None, repr=False, compare=False)

    @property
    def metrics(self) -> dict[str, float]:
        """Each metric's mean over the cases."""
        return {m: e.mean for m, e in self.estimates.items() if e.mean is not None}

    @property
    def cases(self) -> dict[str, dict[str, float]]:
        """Each case's metrics, averaged over its scored repetitions."""
        out: dict[str, dict[str, list[float]]] = {}
        for a in self.trials:
            if a.status in SCORED:
                for m, v in a.metrics.items():
                    out.setdefault(a.case, {}).setdefault(m, []).append(v)
        return {c: {m: sum(vs) / len(vs) for m, vs in ms.items()} for c, ms in out.items()}

    @property
    def feasible(self) -> bool | None:
        """Every constraint is met; None when the candidate is not eligible (it cannot be judged)."""
        if not self.eligible:
            return None
        return all(v is not None and v <= 0 for v in self.violations)

    def agent(self, **resources: Any) -> Any:
        """Build the agent this candidate measured."""
        assert self._factory is not None
        return self._factory(**resources, **self._factory.bind(Values(self.params)))


class Scope:
    """One run of one schedule: the blocks it may change, how many candidates it may still run, and the candidates it
    holds values from. A round of ``Nested`` is a child scope of the run it belongs to; ``optimize()`` opens the root.

    Policies create candidates only through a scope (``candidate``) and rounds only through ``child``, so every
    candidate's place in the schedule is recorded, the budget is kept in one place, and every result reaches the right
    methods."""

    def __init__(
        self,
        study: "Study",
        *,
        schedule: Schedule,
        node: str,
        blocks: Sequence[str],
        limit: int,
        parent: "Scope | None" = None,
        round: int | None = None,
        outer: str | None = None,
        objective: int | None = None,
    ) -> None:
        self.study, self.schedule, self.node, self.parent = study, schedule, node, parent
        self.blocks, self.limit = tuple(blocks), limit
        self.round, self.outer, self.objective = round, outer, objective
        self.candidates: list[Candidate] = []  # the candidates run in this scope and its children
        self._deferred: dict[str, tuple[Draw, int]] = {}
        self.id = study._open(self)

    @property
    def kind(self) -> str:
        return type(self.schedule).__name__

    @property
    def path(self) -> tuple[int, ...]:
        return (*self.parent.path, self.id) if self.parent else (self.id,)

    @property
    def budget(self) -> int:
        """The candidates this scope may still run (never more than the scopes around it; none once the run has lost
        patience)."""
        own = self.limit - len(self.candidates)
        if self.parent is None:
            return 0 if self.study._out_of_patience() else own
        return min(own, self.parent.budget)

    def turn(self) -> int:
        """The objective whose methods propose next: this round's, or the next in turn (with ``Independent``)."""
        return self.objective if self.objective is not None else self.study.turn()

    def incumbent(self, objective: int | None = None) -> Candidate:
        """The candidate other values are held at: the best of this scope's candidates (the root holds from every
        candidate in the study), else the enclosing scope's."""
        k = objective if objective is not None else self.objective
        pool = self.study.candidates if self.parent is None else self.candidates
        held = self.study.combine.held(self.study.projection, pool, k, self.study.baseline)
        if held is not None:
            return held
        return self.parent.incumbent(objective) if self.parent else self.study.baseline

    def best(self, objective: int | None = None) -> Candidate | None:
        """The best of this scope's own candidates, or None when none of them can be compared."""
        k = objective if objective is not None else self.objective
        return self.study.combine.held(self.study.projection, self.candidates, k, self.study.baseline)

    def candidate(
        self,
        update: Sequence[str],
        *,
        role: str = "step",
        step: int | None = None,
        defer: Sequence[str] = (),
        base: Candidate | None = None,
        objective: int | None = None,
    ) -> Candidate | None:
        """Propose and measure one candidate that changes ``update`` (the blocks among them that exist under the
        structure being built), holding everything else at ``base`` (default: this scope's incumbent). Returns
        None, without running anything, when none of them has anything to propose. ``defer`` blocks are not told
        the result now: ``tell`` gives them the result that stands for their draw."""
        if self.budget <= 0:
            raise RuntimeError(f"scope {self.id} ({self.kind}) has no candidates left")
        allowed = set(self.blocks) | ({self.outer} if role == "outer" and self.outer else set())
        outside = [b for b in update if b not in allowed]
        if outside:
            raise ValueError(f"blocks {outside} are outside this scope's blocks {list(self.blocks)}")
        k = objective if objective is not None else self.turn()
        made = self.study._candidate(
            update,
            base if base is not None else self.incumbent(k),
            objective=k,
            scope=self,
            role=role,
            step=step,
            defer=defer,
        )
        if made is None:
            return None
        candidate, draws = made
        for block in defer:
            if block in draws:
                self._deferred[block] = (draws[block], candidate.number)
        return candidate

    def child(
        self,
        name: str,
        schedule: Schedule,
        *,
        blocks: Sequence[str],
        limit: int,
        round: int | None = None,
        outer: str | None = None,
        objective: int | None = None,
    ) -> "Scope":
        """Open a scope inside this one (a round), where ``schedule`` (this schedule's child ``name``) runs."""
        return Scope(
            self.study,
            schedule=schedule,
            node=f"{self.node}.{name}" if self.node else name,
            blocks=blocks,
            limit=limit,
            parent=self,
            round=round,
            outer=outer,
            objective=objective,
        )

    def narrow(self, candidate: Candidate) -> None:
        """Keep only the blocks with variables under ``candidate``'s choices."""
        self.blocks = tuple(b for b in self.blocks if self.study.active(b, candidate.params))

    def tell(self, block: str, candidate: Candidate | None) -> None:
        """Give ``block``'s deferred draw the result that stands for it (None: it failed)."""
        draw, number = self._deferred.pop(block)
        self.study._tell(block, draw, number, candidate, objective=self.objective, scope=self)

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, kind: Any, *exc: Any) -> bool:
        if kind is not None:  # interrupted: the scope stays open in the record, and its draws untold
            return False
        for block in list(self._deferred):  # a draw never told (the round ended early): it failed
            self.tell(block, None)
        self.study._close(self)
        return False


def _score_range(evaluator: Any) -> Range:
    """Where an evaluator's scores lie: its ``score_range``, else 0 to 1."""
    lo, hi = getattr(evaluator, "score_range", (0.0, 1.0))
    return (None if lo is None else float(lo), None if hi is None else float(hi))


class Study:
    def __init__(
        self,
        *,
        agent: TunableFactory | Callable[..., Any],
        experiment: Any,
        objectives: Maximize | Minimize | Combine,
        constraints: Sequence[AtMost | AtLeast] = (),
        search: Search | None = None,
        repetitions: int = 1,
        agent_error_score: float | str | None = "worst",
        max_concurrency: int = 4,
        task: Task = invoke,
        retries: int = 2,
        record: Any = None,
        log: Callable[[str], None] | None = print,
    ) -> None:
        """``objectives``: one objective, or several wrapped in how they combine: ``Pareto(...)`` (jointly),
        ``WeightedSum(...)``, ``Chebyshev(...)``, ``Lexicographic(...)`` or ``Independent(...)``. An objective or
        constraint names a metric: an evaluator's (its name), the agent's usage (``tokens``, ``seconds``, ...), or a
        ``Derived`` one. An evaluator's scores lie in 0 to 1 unless it says otherwise (``score_range = (0, None)``).
        ``repetitions``: how many times each candidate runs every case (each run is a trial; a candidate's estimates
        are case-level means with 95% intervals).
        ``agent_error_score`` is the evaluator score a trial gets when its agent raised: ``"worst"`` (the end of the
        metric's range that is worst for the goals using it, e.g. 0 for a maximized score in [0, 1]), a number, or
        ``None`` (left out).
        ``task(agent, case)`` runs one trial with the agent built for it and returns its output, as a Strands Evals task
        does (or a dict with ``output``, ``trajectory``, ...); the default calls ``agent.invoke_async(case.input)``. It
        may be async. ``retries``: how many more times a trial runs when it measured nothing because something other
        than the agent failed (the task raised ``TrialInvalid``, or a connection failed; Strands Evals retries
        throttling).
        ``search``: how to search (``Search``: blocks, each a method and the variables it proposes, and the schedule
        that says when each moves); the default puts every variable in a block for its type and builds the schedule
        from the space. Values already measured are never measured again (``Search(redraws=...)``).
        ``log`` receives one line per finished candidate (``None`` for silence); ``record`` is a directory where the
        study and every candidate are written as they finish (``record.py``), from the first ``optimize()`` on.

        ``experiment`` is a Strands Evals experiment: its cases, scored by its evaluators, run in this process."""
        self.agent = agent if isinstance(agent, TunableFactory) else tunable()(agent)
        self.experiment = experiment
        if isinstance(objectives, (list, tuple)):
            raise TypeError(
                "give several objectives wrapped in how they combine: objectives=Pareto(...) (jointly), "
                "WeightedSum(...), Chebyshev(...), Lexicographic(...) or Independent(...)"
            )
        self.combine = objectives if isinstance(objectives, Combine) else Pareto(objectives)
        evaluators = {e.get_name(): _score_range(e) for e in experiment.evaluators}
        self.projection = Projection(
            list(self.combine.objectives), constraints, evaluators=evaluators, agent_error_score=agent_error_score
        )
        self.combine.check(self.projection)
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        self.repetitions, self.max_concurrency, self.log = repetitions, max_concurrency, log
        self.task, self.retries = task, retries
        self.search = search if search is not None else Search()
        self._check(set(evaluators))
        self.params = {p.key: p for p in self.agent.params()}
        blocks, self._names = resolve_blocks(self.params, self.search.blocks, self.agent.references())
        self.blocks = {b.name: b for b in blocks}
        self.block_of = {key: b.name for b in self.blocks.values() for key in b.keys}
        # Every block's method as given, plus, with Independent, a fresh copy of it for each further objective.
        self.methods = [
            {name: b.method if k == 0 else b.method.copy(k) for name, b in self.blocks.items()}
            for k in range(self.combine.instances(len(self.combine.objectives)))
        ]
        self.schedule = self.search.schedule or default_schedule(self)
        self.schedule.check(list(self.blocks), self)
        self.candidates: list[Candidate] = []
        self._turn = 0
        self._scopes = 0
        self._patience: int | None = None
        self._unchanged = 0  # candidates since the best set last changed
        self._record_dir = record  # opened by the first optimize(): building a Study touches nothing on disk
        self.record: Any = None

    def name_of(self, block: Block) -> str:
        """The name of one of this study's blocks: a ``Block`` given in its ``Search``, or one it resolved."""
        if id(block) in self._names:
            return self._names[id(block)]
        if isinstance(block, Block) and self.blocks.get(block.name) is block:
            return block.name
        raise ValueError(f"{block!r} is not one of this study's blocks: give it in Search(blocks=...)")

    def _check(self, evaluators: set[str]) -> None:
        if not self.projection.objectives:
            raise ValueError("give at least one objective")
        names = [case.name for case in self.experiment.cases]
        if not names:
            raise ValueError("the experiment has no cases")
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("give every Case a unique name: each case's metrics are recorded by name")
        available = evaluators | set(USAGE) | set(self.projection.derived)
        for goal in [*self.projection.objectives, *self.projection.constraints]:
            if goal.metric not in available:
                raise ValueError(f"unknown metric {goal.metric!r}; the metrics are {sorted(available)}")

    @property
    def baseline(self) -> Candidate:
        return self.candidates[0]

    # ---- the space ------------------------------------------------------------------------------------------------
    def ancestors(self, key: str) -> list[str]:
        """The choices ``key`` exists under, innermost first."""
        out, at = [], self.params[key].active_when
        while at is not None:
            out.append(at[0])
            at = self.params[at[0]].active_when
        return out

    def active(self, block: str, params: dict[str, Any]) -> bool:
        """Whether any of the block's variables exists in ``params`` (a candidate holds exactly the variables its
        choices make exist)."""
        return any(k in params for k in self.blocks[block].keys)

    def structural_order(self) -> list[str]:
        """The structural blocks, a block whose variables exist only under another's choices after that block."""
        structural = [name for name, b in self.blocks.items() if b.structural]
        after = {
            name: {self.block_of[s] for k in self.blocks[name].keys for s in self.ancestors(k)} - {name}
            for name in structural
        }
        order: list[str] = []
        while len(order) < len(structural):
            ready = [n for n in structural if n not in order and after[n] <= set(order)]
            if not ready:
                raise ValueError(f"the structural blocks depend on each other in a cycle: {after}")
            order.append(ready[0])
        return order

    @property
    def artifacts(self) -> Path:
        """Where the study keeps what its methods write (skill-library versions, curators' decisions): ``artifacts/``
        in the record directory, or a temporary directory for a study without a record."""
        if getattr(self, "_artifacts", None) is None:
            import tempfile

            self._artifacts = (
                Path(self._record_dir) / "artifacts"
                if self._record_dir is not None
                else Path(tempfile.mkdtemp(prefix="study-artifacts-"))
            )
        return self._artifacts

    def block_order(self) -> list[str]:
        """Every block: the structural ones outermost first, then the rest in the order the agent declares them."""
        position = {key: i for i, key in enumerate(self.params)}
        rest = sorted(
            (b for b in self.blocks if not self.blocks[b].structural),
            key=lambda b: min(position[k] for k in self.blocks[b].keys),
        )
        return [*self.structural_order(), *rest]

    # ---- running ----------------------------------------------------------------------------------------------------
    def optimize(self, iterations: int, *, patience: int | None = None) -> None:
        """Run ``iterations`` more iterations, each proposing and measuring one candidate. A study's first call measures
        the agent at its defaults (the baseline) first.

        ``patience``: stop early once the best candidates (as the objectives' combination defines best) have not
        changed for this many iterations in a row; omitted, the run uses every iteration. A round in progress ends
        there, and its outer proposal is told the round's best so far. A run also ends early when its schedule has
        nothing new to propose (every proposal repeats a measured candidate)."""
        if iterations < 0:
            raise ValueError("iterations must be at least 0")
        if patience is not None and patience < 1:
            raise ValueError("patience must be at least 1 (or None, to run every iteration)")
        if self._record_dir is not None and self.record is None:
            from .record import Record

            self.record = Record(self._record_dir)
            self.record.study(self)
        if self.record is not None:
            self.record.run(iterations, patience)
        self._patience, self._unchanged = patience, 0
        try:
            if not self.candidates:
                made = self._candidate([], None, objective=None, scope=None, role="baseline", step=None, defer=())
                assert made is not None
                baseline = made[0]
                if not baseline.eligible:
                    raise RuntimeError(
                        f"the baseline (the agent at its defaults) measured nothing usable: "
                        f"{baseline.status} ({baseline.reason or self._first_error(baseline)})"
                    )
            left = 0
            if iterations > 0:
                with Scope(self, schedule=self.schedule, node="", blocks=list(self.blocks), limit=iterations) as root:
                    self.schedule.run(root)
                    left = root.budget
        except BaseException as exc:
            if self.record is not None:
                self.record.stopped(f"{type(exc).__name__}: {exc}")
            raise
        if self.record is not None:
            reason = "patience" if self._out_of_patience() else "exhausted" if left > 0 else "candidates"
            self.record.finish(self, reason)
            self._effects()
        self._patience = None

    def _out_of_patience(self) -> bool:
        return self._patience is not None and self._unchanged >= self._patience

    def _effects(self) -> None:
        """Fit the effects model to the record (``effects.json``); a study without one still finishes."""
        from . import effects

        try:
            effects.write(self.record.directory)
        except Exception as exc:  # noqa: BLE001 - analysis is never a reason to fail a study
            if self.log:
                self.log(f"effects: not written ({type(exc).__name__}: {exc})")

    @staticmethod
    def _first_error(candidate: Candidate) -> str:
        return next((a.error for a in candidate.trials if a.error), "no trial had a value for every goal metric")

    def turn(self) -> int:
        """The objective whose methods propose next (they take turns with Independent)."""
        k = self._turn % len(self.methods)
        self._turn += 1
        return k

    def method(self, block: str, objective: int | None = None) -> Any:
        return self.methods[objective or 0][block]

    def _open(self, scope: Scope) -> int:
        self._scopes += 1
        if self.record is not None:
            self.record.scope(scope, self._scopes)
        return self._scopes

    def _close(self, scope: Scope) -> None:
        if self.record is not None:
            self.record.scope_end(scope)

    def _propose(
        self,
        update: Sequence[str],
        base: Candidate | None,
        objective: int | None,
    ) -> tuple[Values, dict[str, Draw], _Source] | None:
        """The candidate's values: the ``update`` blocks that exist under the structure being built proposed by their
        methods, structural blocks first (outermost first, each given the structure so far), everything else held."""
        held = dict(base.params) if base else {}
        draws: dict[str, Draw] = {}
        values = self.agent.suggest(_Source(self, held, draws, objective)).values
        choices = lambda: {k: v for k, v in values.items() if self.params[k].type in SELECTORS}  # noqa: E731
        for name in self.structural_order():
            if name in update and self.active(name, values):
                draws[name] = self.method(name, objective).begin(
                    self.blocks[name], base, self.candidates, choices(), self, objective
                )
                values = self.agent.suggest(_Source(self, held, draws, objective)).values
        structure = choices()
        for name in update:
            if not self.blocks[name].structural and self.active(name, values):
                draws[name] = self.method(name, objective).begin(
                    self.blocks[name], base, self.candidates, structure, self, objective
                )
        if update and not draws:
            return None
        source = _Source(self, held, draws, objective)
        return self.agent.suggest(source), draws, source

    def _candidate(
        self,
        update: Sequence[str],
        base: Candidate | None,
        *,
        objective: int | None,
        scope: Scope | None,
        role: str,
        step: int | None,
        defer: Sequence[str],
    ) -> tuple[Candidate, dict[str, Draw]] | None:
        """Build, check and measure one candidate; give its result to the methods of the scope's blocks."""
        unknown = [name for name in update if name not in self.blocks]
        if unknown:
            raise ValueError(f"unknown blocks {unknown}; the blocks are {sorted(self.blocks)}")
        for _ in range(self.search.redraws + 1):
            proposal = self._propose(update, base, objective)
            if proposal is None:
                return None
            values, draws, source = proposal
            earlier = self._measured(values.to_dict())
            if earlier is None:
                break
            opens = role == "outer"
            keep = defer if opens else ()
            self._repeat(earlier, draws, source, scope=scope, role=role, step=step, objective=objective, keep=keep)
            if opens:  # the round opens on the earlier candidate: not measured again, nor counted against the budget
                scope.candidates.append(earlier)
                return earlier, draws
        else:
            return None  # every proposal repeated a measured candidate: nothing new to measure here
        common = dict(
            number=len(self.candidates),
            params=values.to_dict(),
            proposed=tuple(draws),
            base=base.number if base else None,
            objective=objective if len(self.methods) > 1 else None,
            origin={name: draw.origin for name, draw in draws.items()},
            drawn={name: tuple(k for k in source.drawn if self.block_of[k] == name) for name in draws},
            inherited=dict(source.inherited),
            defaulted=tuple(source.defaulted),
            scope=scope.id if scope else None,
            path=scope.path if scope else (),
            role=role,
            step=step,
            _factory=self.agent,
        )
        problem = self._validate(values)
        if problem:
            candidate = Candidate(
                status="invalid",
                trials=(),
                estimates={},
                eligible=False,
                violations=(None,) * len(self.projection.constraints),
                reason=problem,
                **common,
            )
        else:
            run = run_trials(
                self.agent,
                values,
                self.experiment,
                repetitions=self.repetitions,
                max_concurrency=self.max_concurrency,
                task=self.task,
                retries=self.retries,
            )
            trials = tuple(self.projection.prepare(t) for t in self._loop().run_until_complete(run))
            estimates = self.projection.estimates(trials)
            eligible = self.projection.eligible(trials)
            scored = [a for a in trials if a.status in SCORED]
            status = "failed" if not scored else ("complete" if eligible and len(scored) == len(trials) else "partial")
            violations = self.projection.violations({m: e.mean for m, e in estimates.items()})
            candidate = Candidate(
                status=status, trials=trials, estimates=estimates, eligible=eligible, violations=violations, **common
            )
        before = {t.number for t in self.best_candidates} if self.candidates else None
        self.candidates.append(candidate)
        self._unchanged = self._unchanged + 1 if before == {t.number for t in self.best_candidates} else 0
        at = scope
        while at is not None:
            at.candidates.append(candidate)
            at = at.parent
        shared = [b for b in (scope.blocks if scope else self.blocks) if b not in defer]
        self._share(candidate, draws, shared)
        if self.record is not None:
            self.record.candidate(self, candidate, shared=shared, deferred=[b for b in defer if b in draws])
        if self.log:
            self.log(self._line(candidate))
        return candidate, draws

    def _measured(self, values: dict[str, Any]) -> Candidate | None:
        """The candidate that already measured these values, if one did (a candidate that measured nothing usable can
        be measured again)."""
        return next((c for c in self.candidates if c.eligible and c.params == values), None)

    def _repeat(
        self,
        earlier: Candidate,
        draws: dict[str, Draw],
        source: "_Source",
        *,
        scope: Scope | None,
        role: str,
        step: int | None,
        objective: int | None,
        keep: Sequence[str],
    ) -> None:
        """A proposal repeated ``earlier``: each of its draws (but those in ``keep``, told later) learns that its values
        scored what ``earlier`` measured."""
        k = objective or 0
        told = [name for name in draws if name not in keep]
        for name in told:
            self.methods[k][name].observe(self.blocks[name], earlier, draws[name], self, k)
        if self.record is not None:
            drawn = {name: tuple(key for key in source.drawn if self.block_of[key] == name) for name in draws}
            self.record.repeat(
                self, earlier, draws, drawn, scope=scope, role=role, step=step, objective=objective, told=told
            )
        if self.log:
            self.log(f"{'repeat':<13}{'/'.join(draws):<18} = candidate {earlier.number}: not measured again")

    def _loop(self) -> asyncio.AbstractEventLoop:
        """One event loop for the whole study: libraries keep asyncio objects between candidates (Harbor's per-image
        build locks), and those are bound to the loop that made them."""
        if getattr(self, "_event_loop", None) is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()
        return self._event_loop

    def _validate(self, candidate: Candidate) -> str | None:
        """Why the candidate cannot be measured, if it cannot: it is outside the space, or the agent cannot be built."""
        try:
            self.agent.validate(candidate)
            self.agent(**self.agent.bind(candidate))
        except Exception as exc:  # noqa: BLE001 - any failure to build makes the candidate invalid
            return f"{type(exc).__name__}: {exc}"
        return None

    def _share(self, candidate: Candidate, draws: dict[str, Draw], blocks: Sequence[str]) -> None:
        """Give a candidate's result to the methods of ``blocks``: to the copy that proposed it as the outcome of its
        own draw, to every other copy as evidence."""
        for k, methods in enumerate(self.methods):
            mine = (candidate.objective or 0) == k
            for name in blocks:
                methods[name].observe(self.blocks[name], candidate, draws.get(name) if mine else None, self, k)

    def _tell(
        self, block: str, draw: Draw, number: int, candidate: Candidate | None, *, objective: int | None, scope: Scope
    ) -> None:
        """Tell the copy that drew ``draw`` (in candidate ``number``) the result that stands for it, and give that
        result to the block's other copies as evidence."""
        k = objective or 0
        for i, methods in enumerate(self.methods):
            if i == k:
                methods[block].observe(self.blocks[block], candidate, draw, self, i)
            elif candidate is not None:
                methods[block].observe(self.blocks[block], candidate, None, self, i)
        if self.record is not None:
            self.record.told(self, scope, block, number, candidate, k)

    # ---- results ----------------------------------------------------------------------------------------------------
    def incumbent(self, among: Sequence[Candidate] | None = None, objective: int | None = None) -> Candidate:
        """The candidate other blocks are held at, chosen by ``combine`` from ``among`` (default: every candidate)."""
        pool = list(among) if among is not None else self.candidates
        held = self.combine.held(self.projection, pool, objective, self.baseline)
        return held if held is not None else self.baseline

    @property
    def best_candidates(self) -> list[Candidate]:
        """The best feasible candidates as ``combine`` defines best: the Pareto set, or the single best."""
        return self.combine.best(self.projection, self.candidates, self.baseline)

    @property
    def best_candidate(self) -> Candidate:
        best = self.best_candidates
        if self.combine.best_is_set(len(self.projection.objectives)):
            raise ValueError("with several objectives there is no single best candidate: see best_candidates")
        if not best:
            raise ValueError("no candidate meets the constraints")
        return best[0]

    def report(self) -> str:
        """One line per candidate: what changed, each goal metric's mean and interval, status."""
        return "\n".join(self._line(t) for t in self.candidates)

    def _line(self, t: Candidate) -> str:
        parts = []
        for m in self.projection.goal_metrics:
            e = t.estimates.get(m)
            if e and e.mean is not None:
                ci = f" [{e.ci[0]:.3g}, {e.ci[1]:.3g}]" if e.ci else ""
                parts.append(f"{m} {e.mean:.3g}{ci}")
        n = len({a.case for a in t.trials})
        return (
            f"candidate {t.number:2} {'/'.join(t.proposed) or t.role:18} {t.status:8} n={n} "
            + "  ".join(parts)
            + ("  (infeasible)" if t.feasible is False else "")
            + (f"  {t.reason}" if t.reason else "")
        )


class _Source:
    """What the factory's search space asks for while a candidate is built: values of the proposed blocks from their
    methods' draws; every other value as held; a value the held candidate lacks (it exists only under a choice the held
    candidate did not make) from the best candidate that has it, else its default."""

    def __init__(self, study: Study, held: dict[str, Any], draws: dict[str, Draw], objective: int | None) -> None:
        self.study, self.held, self.draws, self.objective = study, held, draws, objective
        self.drawn: list[str] = []
        self.inherited: dict[str, int] = {}
        self.defaulted: list[str] = []

    def _value(self, name: str, ask: Callable[[Draw], Any]) -> Any:
        block = self.study.block_of[name]
        if block in self.draws:
            self.drawn.append(name)
            return ask(self.draws[block])
        if name in self.held:
            return self.held[name]
        having = [t for t in self.study.candidates if name in t.params]
        if having:
            best = self.study.incumbent(having, self.objective)
            if name in best.params:
                self.inherited[name] = best.number
                return best.params[name]
        self.defaulted.append(name)
        return self.study.params[name].default

    def suggest_float(self, name: str, *args: Any, **kwargs: Any) -> float:
        return self._value(name, lambda d: d.suggest_float(name, *args, **kwargs))

    def suggest_int(self, name: str, *args: Any, **kwargs: Any) -> int:
        return self._value(name, lambda d: d.suggest_int(name, *args, **kwargs))

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        return self._value(name, lambda d: d.suggest_categorical(name, choices))

    def suggest_text(self, name: str, default: str) -> str:
        return self._value(name, lambda d: d.suggest_text(name, default))

    def suggest_skills(self, name: str, default: str | None) -> str | None:
        return self._value(name, lambda d: d.suggest_skills(name, default))


__all__ = ["AtLeast", "AtMost", "Maximize", "Minimize", "Scope", "Study", "Candidate", "constraint_name"]
