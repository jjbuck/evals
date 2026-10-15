"""Blocks of variables and the methods that update them.

A block is a method and the variables it proposes. The variables are the objects declared in ``@tunable``:

    window_size = Int(4, 40, default=6)
    manager = OneOf(component(SlidingWindowConversationManager, window_size=window_size), ...)

    @tunable(conversation_manager=manager, system_prompt=system_prompt, tools=tool_descriptions(...))
    def create_agent(...): ...

    Block(TPE(), manager)                       # the choice of manager
    Block(TPE(), window_size)                   # that one setting
    Block(Reflection(), system_prompt, tools)   # a component (here, tools) stands for every variable declared in it

A value (``Int``, ``TextValue``, ...) and a choice (``OneOf``, ``Switch``) are each one variable; a component stands for
the variables declared in it, unless another block names one of them itself. Every variable no block names goes to a
default block for its type: ``structure`` (the choices) and ``numbers`` (``Float``, ``Int``, ``Choice``) by TPE,
``text`` (``TextValue``) by contrastive reflection, ``skills`` (``SkillLibrary``) by skill-library curation. A block
that holds a choice is *structural*: it is proposed before the blocks whose variables exist only under it.

A method proposes values for its block given the values held for everything else (``begin`` returns a draw the
candidate is built from) and learns from results (``observe``): the outcome of its own proposal, or evidence from
candidates it did not propose.
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .projection import constraint_name

if TYPE_CHECKING:
    from .study import Candidate, Study

SELECTORS = ("one_of", "switch")  # variables that choose which other variables exist


class Draw:
    """One proposal in progress: the values a method gives while the candidate is built."""

    origin: dict[str, Any]

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float:
        raise TypeError(f"{name}: this method does not propose numbers")

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int:
        raise TypeError(f"{name}: this method does not propose numbers")

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        raise TypeError(f"{name}: this method does not propose choices")

    def suggest_text(self, name: str, default: str) -> str:
        raise TypeError(f"{name}: this method does not write text")

    def suggest_skills(self, name: str, default: str | None) -> str | None:
        raise TypeError(f"{name}: this method does not curate skills")


class Method(ABC):
    kinds: tuple[str, ...] = ()  # the variable types this method can propose

    @abstractmethod
    def begin(
        self,
        block: "Block",
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        """Start a proposal for ``block``. ``base`` is the candidate whose values are held for every other block, and
        ``structure`` the choices this candidate has so far (structural blocks are proposed first, outermost first,
        so a structural block sees the choices of the blocks around it). ``objective`` is the objective this copy of
        the method serves (with ``Independent``)."""

    def observe(  # noqa: B027 - optional: a method that learns nothing from results need not implement it
        self, block: "Block", candidate: "Candidate | None", draw: Draw | None, study: "Study", objective: int
    ) -> None:
        """Learn from a candidate: the outcome of ``draw`` when this method proposed it, else evidence. A ``candidate``
        of ``None`` with a draw means the draw failed (nothing could be told for it)."""

    def copy(self, k: int) -> "Method":
        """A fresh copy of this method for the k-th further objective (with ``Independent``)."""
        import copy

        return copy.deepcopy(self)

    def describe(self) -> dict[str, Any]:
        """The method's kind and settings, for the record."""
        return {"kind": type(self).__name__}


class Block:
    """A method and the variables it proposes: ``Block(TPE(), window_size, pin_first)``. ``name`` labels the block in
    the record and the explorer (default: the kind of variable it holds)."""

    def __init__(self, method: Method, *variables: Any, name: str | None = None) -> None:
        if not isinstance(method, Method):
            raise TypeError("Block(method, *variables): the method comes first, as in Block(TPE(), window_size)")
        if not variables:
            raise ValueError("a block needs at least one variable: the objects declared in @tunable")
        self.method, self.variables, self.name = method, tuple(variables), name
        self.keys: tuple[str, ...] = ()  # the variables' keys, once the study has resolved them
        self.structural = False  # whether it holds a choice, once resolved

    @classmethod
    def resolved(cls, name: str, method: Method, keys: Sequence[str], structural: bool) -> "Block":
        """A block as the study runs it: named, with the keys of its variables."""
        block = cls.__new__(cls)
        block.method, block.variables, block.name = method, (), name
        block.keys, block.structural = tuple(keys), structural
        return block

    def __repr__(self) -> str:
        return f"Block({self.name!r}, {type(self.method).__name__}, keys={list(self.keys)})"


def context(block: Block, params: dict[str, Any], study: "Study") -> tuple:
    """Which landscape a block's values belong to: a structural block has one; any other block one per combination of
    the choices it is built under. Proposals and evidence are both filed by this."""
    if block.structural:
        return ()
    return tuple(sorted((k, v) for k, v in params.items() if study.params[k].type in SELECTORS and k not in block.keys))


# ---- Optuna ---------------------------------------------------------------------------------------------------------
class _OptunaDraw(Draw):
    def __init__(self, ask: Any, study: Any, origin: dict[str, Any]) -> None:
        self.ask, self.study, self.origin = ask, study, origin  # the Optuna trial this draw asked for

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float:
        return self.ask.suggest_float(name, low, high, step=step, log=log)

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int:
        return self.ask.suggest_int(name, low, high, step=step, log=log)

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        return self.ask.suggest_categorical(name, choices)


class Optuna(Method):
    """An Optuna sampler over the block's numbers and choices: ``Optuna(lambda: optuna.samplers.GPSampler())``.

    It keeps one Optuna study per landscape (``context``: for a non-structural block, each combination of the choices
    outside it), and every candidate in that landscape is evidence for it, whoever proposed it."""

    kinds = ("float", "int", "choice", "one_of", "switch")

    def __init__(self, sampler: Callable[[], Any] | None = None) -> None:
        self.sampler = sampler
        self._studies: dict[tuple, Any] = {}

    def _study(self, context: tuple, study: "Study", objective: int | None) -> Any:
        import optuna

        if context not in self._studies:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                sampler = self._sampler(len(self._studies))
            directions = study.combine.directions(study.projection, objective)
            optuna_study = optuna.create_study(directions=directions, sampler=sampler)
            optuna_study.stop = lambda: None  # the study's budget ends the search, not the sampler (grid, brute force)
            self._studies[context] = optuna_study
        return self._studies[context]

    def _sampler(self, landscape: int) -> Any:
        """The sampler for the ``landscape``-th Optuna study this method opens."""
        return self.sampler() if self.sampler else None

    def begin(
        self,
        block: Block,
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        given = context(block, structure, study)
        optuna_study = self._study(given, study, objective)
        ask = optuna_study.ask()
        return _OptunaDraw(
            ask, optuna_study, {"method": type(self).__name__, "optuna_trial": ask.number, "given": dict(given)}
        )

    def observe(
        self, block: Block, candidate: "Candidate | None", draw: Draw | None, study: "Study", objective: int
    ) -> None:
        """Told what ``combine`` tells this objective's methods, with the constraint violations; a candidate that cannot
        be told (none, invalid, or some case lacks a goal metric) fails its own draw and is no evidence."""
        import optuna

        if isinstance(draw, _OptunaDraw):
            if candidate is None or not candidate.eligible:
                draw.study.tell(draw.ask, state=optuna.trial.TrialState.FAIL)
                return
            for c, violation in zip(study.projection.constraints, candidate.violations, strict=True):
                draw.ask.set_constraint(constraint_name(c), violation)
            draw.study.tell(draw.ask, study.combine.told(study.projection, candidate, objective, study.baseline))
            return
        if candidate is None or not candidate.eligible:
            return
        distributions = {k: d for k, d in study.agent.distributions(candidate.params).items() if k in block.keys}
        if not distributions:
            return
        self._study(context(block, candidate.params, study), study, objective).add_trial(
            optuna.trial.create_trial(
                params={k: candidate.params[k] for k in distributions},
                distributions=distributions,
                values=study.combine.told(study.projection, candidate, objective, study.baseline),
                constraints={
                    constraint_name(c): v
                    for c, v in zip(study.projection.constraints, candidate.violations, strict=True)
                }
                or None,
            )
        )


class TPE(Optuna):
    """Optuna's tree-structured Parzen estimator: ``TPE(seed=7, n_startup_trials=4)``.

    ``n_startup_trials`` is how many candidates a landscape holds (proposed or shared as evidence) before TPE's model is
    used; until then proposals are random. Optuna's default of 10 would make a short study random search, so the
    default here is 4."""

    def __init__(self, seed: int | None = None, n_startup_trials: int = 4, **options: Any) -> None:
        self.seed, self.n_startup_trials, self.options = seed, n_startup_trials, options
        super().__init__()

    def _sampler(self, landscape: int) -> Any:
        import optuna

        seed = None if self.seed is None else self.seed + landscape  # each landscape draws its own startup points
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=self.n_startup_trials, **{"multivariate": True, "group": True, **self.options}
        )

    def copy(self, k: int) -> "TPE":
        return TPE(
            seed=None if self.seed is None else self.seed + 1000 * k,
            n_startup_trials=self.n_startup_trials,
            **self.options,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "seed": self.seed,
            "n_startup_trials": self.n_startup_trials,
            **self.options,
        }


# ---- contrastive reflection -----------------------------------------------------------------------------------------
class _TextValues:
    """Text values in the shape the reflection optimizer reads and writes."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = dict(values)

    def get_tunable_params(self) -> dict:
        return dict(self.values)

    def update_params(self, params: dict) -> None:
        self.values.update({k: v for k, v in params.items() if k in self.values})


class _TextDraw(Draw):
    def __init__(self, values: dict[str, str], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_text(self, name: str, default: str) -> str:
        return self.values.get(name, default)


class _TaskMessage:
    """The reflection task message: the template, given what each variable is and how it may change."""

    def __init__(self, template: Any, variables: list[dict[str, Any]]) -> None:
        self.template, self.variables = template, variables

    def render(self, **values: Any) -> str:
        return self.template.render(**values, variables=self.variables)


def _feed(optimizer: Any, rewards: list, study: "Study") -> None:
    """Give a reflection optimizer the held candidate's trials: messages, rewards and their reasons."""
    from ._reflection import Reward, Rollout

    cases = {case.name: case for case in study.experiment.cases}
    for trial, reward, metadata in rewards:
        case = cases[trial.case]
        optimizer.add_rollouts(
            [
                Rollout(
                    data_sample={
                        "name": trial.case,
                        "repetition": trial.repetition,
                        "input": case.input,
                        "expected_output": case.expected_output,
                    },
                    messages=list(trial.messages),
                    metadata={"output": trial.output},
                )
            ]
        )
        optimizer.add_rewards([Reward(reward=reward, metadata=metadata)])


class Reflection(Method):
    """Contrastive reflection (harness-optimizer's ``ContrastiveReflectionOptimizer``, ported unchanged) over the
    block's text variables: a model compares the held candidate's trials that scored well with those that did not (their
    messages, rewards from the study's objectives and constraints, the evaluators' reasons) and changes the texts the
    traces give it a reason to change. Each variable says what it is (``TextValue(about=...)``) and how it may change: a
    bounded one (``max_chars``, such as a tool's description) is rewritten within its limit, any other has what was
    learned appended. A value over its limit is dropped and the held one kept. Options go to the optimizer
    (``model_config``, ``n_sample_traces``, templates, ...).

    The optimizer gives its model a shell over the traces: run studies with text blocks where model-written shell
    commands are acceptable (a devbox or a container), with ``BYPASS_TOOL_CONSENT=true``."""

    kinds = ("text",)

    def __init__(self, **options: Any) -> None:
        self.options = options

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            **{k: v for k, v in self.options.items() if k != "model_config"},
            **({"model": str(self.options["model_config"])} if "model_config" in self.options else {}),
        }

    def optimizer(self, values: "_TextValues", variables: list[dict[str, Any]]) -> Any:
        """The reflection optimizer for one proposal (a seam for tests and for other reflection optimizers)."""
        from ._reflection import ContrastiveReflection, create_template, load_template

        task = self.options.get("task_message_template", load_template("reflection/task_message.jinja"))
        options = {
            "system_prompt_template": load_template("contrastive_reflection/system_prompt.jinja"),
            **self.options,
            "task_message_template": _TaskMessage(create_template(task) if isinstance(task, str) else task, variables),
        }
        return ContrastiveReflection(values, **options)

    def begin(
        self,
        block: Block,
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        name = type(self).__name__
        keys = [k for k in block.keys if base is not None and k in base.params]
        if base is None or not keys:
            return _TextDraw({}, {"method": name, "from_candidate": None})
        held = {k: base.params[k] for k in keys}
        rewards = study.projection.rewards(base, study.combine, objective)
        if not rewards:
            return _TextDraw(held, {"method": name, "from_candidate": base.number, "status": "nothing to learn from"})
        variables = [{"name": k, "about": study.params[k].about, "max_chars": study.params[k].max_chars} for k in keys]
        text = _TextValues(held)
        optimizer = self.optimizer(text, variables)
        _feed(optimizer, rewards, study)
        origin: dict[str, Any] = {"method": name, "from_candidate": base.number, "trials": len(rewards)}
        try:
            optimizer.step()
        except Exception as exc:  # noqa: BLE001 - a failed reflection keeps the held text; the study goes on
            return _TextDraw(
                held,
                {
                    **origin,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                    "usage": usage(getattr(optimizer, "last_metrics", None)),
                },
            )
        values, dropped = dict(text.values), {}
        for k in keys:
            limit = study.params[k].max_chars
            if limit is not None and len(values[k]) > limit:
                dropped[k] = f"{len(values[k])} characters, over max_chars={limit}"
                values[k] = held[k]
        changed = [k for k in keys if values[k] != held[k]]
        origin.update(
            status="rewritten" if changed else "unchanged",
            changed=changed,
            usage=usage(getattr(optimizer, "last_metrics", None)),
            seconds=getattr(optimizer, "last_wall_clock_s", None),
        )
        if dropped:
            origin["dropped"] = dropped
        return _TextDraw(values, origin)


class _SkillsDraw(Draw):
    def __init__(self, values: dict[str, str | None], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_skills(self, name: str, default: str | None) -> str | None:
        return self.values.get(name, default)


class SkillCuration(Method):
    """Skill-library curation (harness-optimizer's ``SkillLibraryOptimizer``, ported unchanged): a model reads the held
    candidate's trials and the deployed skills, and decides which skills to create, revise or retire; or that the
    library should stay as it is, which is a legitimate outcome. Skills whose frontmatter cannot be loaded are rejected.
    Each new library is stored once, by content, under the study's artifacts. Options go to the optimizer.

    The optimizer gives its model a shell over the traces: run it where model-written shell commands are acceptable,
    with ``BYPASS_TOOL_CONSENT=true``."""

    kinds = ("skills",)

    def __init__(self, **options: Any) -> None:
        self.options = options

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            **{k: v for k, v in self.options.items() if k != "model_config"},
            **({"model": str(self.options["model_config"])} if "model_config" in self.options else {}),
        }

    def optimizer(self, skills: Any, output_folder: str) -> Any:
        """The curation optimizer for one proposal (a seam for tests)."""
        from ._reflection import SkillLibraryCuration, load_template

        options = {
            "system_prompt_template": load_template("skill_library/system_prompt.jinja"),
            "task_message_template": load_template("skill_library/task_message.jinja"),
            **self.options,
        }
        return SkillLibraryCuration(skills, output_folder=output_folder, **options)

    def begin(
        self,
        block: Block,
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        from ._reflection import SkillSet
        from .space.skills import store

        name = type(self).__name__
        keys = [k for k in block.keys if base is not None and k in base.params]
        if base is None or not keys:
            return _SkillsDraw({}, {"method": name, "from_candidate": None})
        held = {k: base.params[k] for k in keys}
        rewards = study.projection.rewards(base, study.combine, objective)
        if not rewards:
            return _SkillsDraw(held, {"method": name, "from_candidate": base.number, "status": "nothing to learn from"})
        values, origin = dict(held), {"method": name, "from_candidate": base.number, "trials": len(rewards)}
        for key in keys:
            library = SkillSet(skill_dir=held[key] or None, strict=False)
            folder = study.artifacts / "skill-decisions" / f"candidate-{len(study.candidates):03d}-{key}"
            optimizer = self.optimizer(library, str(folder))
            _feed(optimizer, rewards, study)
            try:
                optimizer.step()
            except Exception as exc:  # noqa: BLE001 - a failed curation keeps the held library; the study goes on
                return _SkillsDraw(
                    held,
                    {
                        **origin,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "usage": usage(getattr(optimizer, "last_metrics", None)),
                    },
                )
            decisions = getattr(optimizer, "_last_decisions", {}) or {}
            if any(decisions.values()):
                values[key] = store(library.members, study.artifacts / "skills")
            names = {
                k: [os.path.basename(os.path.normpath(x)) if isinstance(x, str) else x for x in v]
                for k, v in decisions.items()
            }
            origin[key] = {"decisions": names, "folder": str(folder)}
        changed = [k for k in keys if values[k] != held[k]]
        origin.update(
            status="rewritten" if changed else "unchanged",
            changed=changed,
            usage=usage(getattr(optimizer, "last_metrics", None)),
            seconds=getattr(optimizer, "last_wall_clock_s", None),
        )
        return _SkillsDraw(values, origin)


def usage(metrics: dict[str, Any] | None) -> dict[str, float]:
    """A method's own model usage in the trials' metric names (``tokens`` = input + output + cache)."""
    m = metrics or {}
    out = {
        name: float(m.get(name, 0) or 0)
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    }
    out["tokens"] = sum(out.values())
    return out


# ---- resolving the blocks -------------------------------------------------------------------------------------------
KINDS = {"structure": SELECTORS, "numbers": ("float", "int", "choice"), "text": ("text",), "skills": ("skills",)}


def _kind(types: set[str]) -> str | None:
    return next((name for name, kinds in KINDS.items() if types <= set(kinds)), None)


def resolve_blocks(
    params: dict[str, Any], given: Sequence[Block], references: Sequence[tuple[Any, tuple[str, ...], bool]]
) -> tuple[list[Block], dict[int, str]]:
    """Every variable in exactly one block: the given block that names it (naming a variable itself outranks naming a
    component it is declared in), else a default block for its type. Returns the blocks and each given block's name."""
    found: dict[int, list[tuple[tuple[str, ...], bool]]] = {}
    for obj, keys, direct in references:
        found.setdefault(id(obj), []).append((keys, direct))
    claims: dict[str, tuple[int, bool]] = {}
    for i, block in enumerate(given):
        for obj in block.variables:
            if id(obj) not in found:
                raise ValueError(
                    f"block {i + 1} holds {obj!r}, which the agent's @tunable does not declare: declare the variable "
                    "once, as a Python object, and give that same object to both"
                )
            for keys, direct in found[id(obj)]:
                for key in keys:
                    held = claims.get(key)
                    if held is None or (direct and not held[1]):
                        claims[key] = (i, direct)
                    elif held[0] != i and held[1] == direct:
                        raise ValueError(f"{key} is in blocks {held[0] + 1} and {i + 1}; give each variable one block")
    groups: list[tuple[str | None, Method, list[str]]] = [(b.name, b.method, []) for b in given]
    defaults: dict[str, list[str]] = {}
    for key, info in params.items():
        if key in claims:
            groups[claims[key][0]][2].append(key)
        else:
            defaults.setdefault(next(n for n, kinds in KINDS.items() if info.type in kinds), []).append(key)
    for i, (_, _, keys) in enumerate(groups):
        if not keys:
            raise ValueError(f"block {i + 1} holds no variable of its own (another block names all of them)")
    taken = {name for name, _, _ in groups if name}
    out, names = [], {}

    def unique(base: str) -> str:
        name, n = base, 2
        while name in taken:
            name, n = f"{base} {n}", n + 1
        taken.add(name)
        return name

    for i, (name, method, keys) in enumerate(groups):
        types = {params[k].type for k in keys}
        name = name or unique(_kind(types) or "block")
        names[id(given[i])] = name
        out.append((name, method, keys))
    for kind, keys in defaults.items():
        method = Reflection() if kind == "text" else SkillCuration() if kind == "skills" else TPE()
        out.append((unique(kind), method, keys))
    if len({name for name, _, _ in out}) != len(out):
        raise ValueError(f"two blocks have the same name: {[name for name, _, _ in out]}")
    blocks = []
    for name, method, keys in out:
        types = {params[k].type for k in keys}
        unsupported = types - set(method.kinds)
        if unsupported:
            raise ValueError(
                f"block {name!r} has {sorted(unsupported)} variables, which {type(method).__name__} "
                f"does not propose; put them in another block"
            )
        blocks.append(Block.resolved(name, method, keys, bool(types & set(SELECTORS))))
    return blocks, names
