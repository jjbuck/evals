"""The schedule: scopes and rounds at any depth, what each method is told, and what the record says about it."""

import json
from types import SimpleNamespace

import pytest
from strands import Agent

from strands_evals.optimization import (
    TPE,
    Alternate,
    AtMost,
    Block,
    Choice,
    Float,
    Independent,
    Int,
    Joint,
    Maximize,
    Method,
    Minimize,
    Nested,
    OneOf,
    Schedule,
    Search,
    Study,
    TextValue,
    component,
    tunable,
)
from strands_evals.optimization.methods import Draw

from .helpers import Says, experiment


class Window:
    def __init__(self, size, truncate):
        self.words = size // 10 + (5 if truncate else 0)


class Brief:
    words = 1


class Detailed:
    def __init__(self, depth):
        self.words = depth


class Summary:
    def __init__(self, ratio, keep, style):
        self.words = round(ratio * 10) + keep + style.words


class Broken:
    def __init__(self):
        raise RuntimeError("this manager cannot be built")


SIZE, TRUNCATE = Int(10, 80, default=40), Choice([True, False], default=True)
RATIO, KEEP, DEPTH = Float(0.1, 0.8, default=0.3), Int(2, 16, default=4), Int(1, 5, default=2)
STYLE = OneOf(component(Brief), component(Detailed, depth=DEPTH), labels=["brief", "detailed"])
WINDOW = component(Window, size=SIZE, truncate=TRUNCATE)
SUMMARY = component(Summary, ratio=RATIO, keep=KEEP, style=STYLE)
MANAGER = OneOf(WINDOW, SUMMARY, labels=["window", "summary"])
BREAKABLE = OneOf(WINDOW, SUMMARY, component(Broken), labels=["window", "summary", "broken"])
PROMPT = TextValue(default="Write.")


def agent(manager, system_prompt):
    return Agent(
        model=Says(" ".join(["w"] * manager.words + system_prompt.split())),
        system_prompt=system_prompt,
        callback_handler=None,
    )


create_managed = tunable(manager=MANAGER, system_prompt=PROMPT)(agent)
create_breakable = tunable(manager=BREAKABLE, system_prompt=PROMPT)(agent)


class _Choose(Draw):
    def __init__(self, values, origin):
        self.values, self.origin = values, origin

    def suggest_categorical(self, name, choices):
        return self.values.get(name, choices[0])

    def suggest_int(self, name, low, high, **kw):
        return self.values.get(name, low)


class Choose(Method):
    """Proposes the given choices in turn (the first option for anything not given); records what it was told."""

    kinds = ("one_of", "switch", "choice", "int")

    def __init__(self, *proposals):
        self.proposals, self.told, self.evidence = list(proposals), [], []

    def begin(self, block, base, history, structure, study, objective):
        values = self.proposals.pop(0) if self.proposals else {}
        return _Choose(values, {"method": "Choose", "given": dict(structure)})

    def observe(self, block, candidate, draw, study, objective):
        if draw is not None:
            self.told.append(None if candidate is None else candidate.number)
        elif candidate is not None:
            self.evidence.append(candidate.number)


class _Append(Draw):
    def __init__(self, text):
        self.text, self.origin = text, {"method": "Append"}

    def suggest_text(self, name, default):
        return self.text


class Append(Method):
    kinds = ("text",)

    def begin(self, block, base, history, structure, study, objective):
        return _Append(base.params["system_prompt"] + " More.")


EXPERIMENT = experiment()


STANZAS = Int(1, 6, default=2)


@tunable(stanzas=STANZAS)
def create_poet(stanzas):
    return Agent(model=Says(" ".join(["word"] * 10 * stanzas)), callback_handler=None)


def managed(schedule=None, *, structure=None, style=None, record=None, factory=create_managed, **kw):
    """The study the tests share: the choices (the nested style too, unless ``style`` gives it a block of its own),
    the numbers and the prompt in a block each; ``schedule(b)`` builds the schedule from them (``b.structure``,
    ``b.numbers``, ``b.text``, ``b.style``)."""
    b = SimpleNamespace(
        structure=Block(structure or TPE(seed=1), factory.bindings["manager"], *([] if style else [STYLE])),
        numbers=Block(TPE(seed=2), SIZE, TRUNCATE, RATIO, KEEP, DEPTH),
        text=Block(Append(), PROMPT),
        style=Block(style, STYLE, name="style") if style else None,
    )
    blocks = [x for x in (b.structure, b.numbers, b.text, b.style) if x is not None]
    return Study(
        agent=factory,
        experiment=EXPERIMENT,
        objectives=kw.pop("objectives", Maximize("WordCount")),
        search=Search(blocks=blocks, schedule=schedule(b) if schedule else None),
        record=record,
        log=None,
        **kw,
    )


def events(path):
    return [json.loads(line) for line in (path / "study.jsonl").read_text().splitlines()]


# ---- rounds ----------------------------------------------------------------------------------------------------------
def test_a_failed_outer_proposal_fails_its_draw_and_runs_no_inner_search(tmp_path):
    choose = Choose({"manager": "broken"}, {"manager": "summary"})
    study = managed(
        lambda b: Nested(outer=b.structure, inner=Joint(), inner_iterations=3),
        structure=choose,
        record=tmp_path,
        factory=create_breakable,
    )
    study.optimize(iterations=5)  # the baseline; a round that fails at its opener; a round of 1 + 3
    roles = [(t.role, t.status, t.params["manager"]) for t in study.candidates]
    assert roles[:3] == [
        ("baseline", "complete", "window"),
        ("outer", "invalid", "broken"),
        ("outer", "complete", "summary"),
    ]
    assert all(t.params["manager"] == "summary" for t in study.candidates[3:])  # never the baseline's stand-in
    assert choose.told == [None, max(study.candidates[2:], key=lambda t: t.metrics["WordCount"]).number]
    told = [e for e in events(tmp_path) if e["type"] == "told"]
    assert [(e["draw_candidate"], e["candidate"]) for e in told] == [(1, None), (2, choose.told[1])]


def test_rounds_at_every_depth_are_in_the_record(tmp_path):
    study = managed(
        lambda b: Nested(
            outer=b.structure, inner=Nested(outer=b.numbers, inner=Joint(), inner_iterations=2), inner_iterations=6
        ),
        record=tmp_path,
    )
    study.optimize(iterations=14)  # the baseline, then 2 rounds of 1 + 6 (each: 2 sub-rounds of 1 + 2)
    log = events(tmp_path)
    scopes = {e["id"]: e for e in log if e["type"] == "scope"}
    assert [(s["parent"], s["node"], s["round"], s["outer"]) for s in scopes.values()] == [
        (None, "", None, None),
        (1, "inner", 0, "structure"),
        (2, "inner.inner", 0, "numbers"),
        (2, "inner.inner", 1, "numbers"),
        (1, "inner", 1, "structure"),
        (5, "inner.inner", 0, "numbers"),
        (5, "inner.inner", 1, "numbers"),
        (5, "inner.inner", 2, "numbers"),  # sub-round 1 opened on candidate 8, so the round had one more to run
    ]
    repeats = [(e["scope"], e["role"], e["candidate"]) for e in log if e["type"] == "repeat"]
    assert repeats == [(7, "outer", 8)]
    candidates = [e for e in log if e["type"] == "candidate"]
    assert [(t["path"], t["role"]) for t in candidates[:8]] == [
        ([], "baseline"),
        ([1, 2], "outer"),
        ([1, 2, 3], "outer"),
        ([1, 2, 3], "step"),
        ([1, 2, 3], "step"),
        ([1, 2, 4], "outer"),
        ([1, 2, 4], "step"),
        ([1, 2, 4], "step"),
    ]
    told = [(e["scope"], e["block"]) for e in log if e["type"] == "told"]
    assert told == [
        (3, "numbers"),
        (4, "numbers"),
        (2, "structure"),
        (6, "numbers"),
        (7, "numbers"),
        (8, "numbers"),
        (5, "structure"),
    ]
    ran = {e["id"]: e["ran"] for e in log if e["type"] == "scope_end"}
    assert ran == {3: 3, 4: 3, 2: 7, 6: 3, 7: 2, 8: 1, 5: 7, 1: 14}  # 7 ran 2: it opened on candidate 8
    head = log[0]
    assert head["schedule"]["children"]["inner"]["children"]["inner"] == {
        "node": "inner.inner",
        "kind": "Joint",
        "children": {},
    }


def test_each_level_of_structure_can_be_its_own_block():
    study = Study(
        agent=create_managed,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        search=Search(blocks=[Block(TPE(), STYLE, name="style")]),
        log=None,
    )
    assert study.blocks["structure"].keys == ("manager",)  # the default block for the choice no block names
    assert study.blocks["style"].keys == ("manager.summary.style",)  # the choice itself; its depth is a number
    assert study.structural_order() == ["structure", "style"]
    schedule = study.schedule  # the default: rounds per level of structure, around everything else
    levels = (study.name_of(schedule.outer), study.name_of(schedule.inner.outer), type(schedule.inner.inner))
    assert levels == ("structure", "style", Joint)


def test_a_block_holds_declared_objects_and_a_component_stands_for_the_variables_in_it():
    def study(*blocks):
        search = Search(blocks=list(blocks))
        return Study(agent=create_managed, experiment=EXPERIMENT, objectives=Maximize("WordCount"), search=search)

    summary, keep = Block(TPE(seed=3), SUMMARY), Block(TPE(seed=4), KEEP)
    s = study(summary, keep)
    # naming a variable itself outranks naming the component it is declared in
    assert s.blocks[s.name_of(summary)].keys == (
        "manager.summary.ratio",
        "manager.summary.style",
        "manager.summary.style.detailed.depth",
    )
    assert s.blocks[s.name_of(keep)].keys == ("manager.summary.keep",)
    with pytest.raises(ValueError, match="in blocks 1 and 2"):
        study(Block(TPE(), KEEP), Block(TPE(), KEEP))
    with pytest.raises(ValueError, match="does not declare"):
        study(Block(TPE(), Int(2, 16, default=4)))  # equal to KEEP, but not the object the agent declares


def test_a_level_with_nothing_under_the_chosen_structure_is_skipped():
    choose = Choose({"manager": "window"}, {"manager": "summary"})
    study = managed(
        lambda b: Nested(
            outer=b.structure, inner=Nested(outer=b.style, inner=Joint(), inner_iterations=2), inner_iterations=4
        ),
        structure=choose,
        style=Choose({"manager.summary.style": "detailed"}),
    )
    study.optimize(iterations=9)
    # the window round's opening proposal is the baseline (a window), which it opens on without measuring it again
    window, summary = study.candidates[1:5], study.candidates[5:10]
    assert [t.proposed for t in window] == [("numbers", "text")] * 4  # no style rounds
    assert [t.proposed for t in summary] == [
        ("structure",),
        ("style",),
        ("numbers", "text"),
        ("numbers", "text"),
        ("style",),
    ]
    assert all(t.drawn and all(t.drawn.values()) for t in study.candidates[1:])  # no candidate that changes nothing


def test_a_structure_chosen_again_inherits_its_own_earlier_values(tmp_path):
    choose = Choose({"manager": "summary"}, {"manager": "window"})
    study = managed(
        lambda b: Nested(outer=b.structure, inner=Joint(), inner_iterations=0),
        structure=choose,
        objectives=Minimize("WordCount"),
        record=tmp_path,
    )  # the summary (fewer words) becomes the incumbent
    study.optimize(iterations=2)
    summary = study.candidates[1]
    assert summary.drawn == {"structure": ("manager", "manager.summary.style")}  # the nested choice is structure too
    assert summary.defaulted == ("manager.summary.ratio", "manager.summary.keep")
    # The window again, held at the summary (which has no window settings): its window settings come from the best
    # window candidate, the baseline, and nothing else differs, so it is the baseline, not measured again. The rounds
    # after it propose the window again too, and the run ends with nothing new to measure.
    log = events(tmp_path)
    assert {(e["role"], e["candidate"]) for e in log if e["type"] == "repeat"} == {("outer", 0)}
    assert len(study.candidates) == 2 and log[-1]["reason"] == "exhausted"


# ---- budgets and checks ----------------------------------------------------------------------------------------------
class Greedy(Schedule):
    """A custom schedule that ignores its budget."""

    def run(self, scope):
        for _ in range(10):
            scope.candidate(list(scope.blocks))


def test_the_budget_is_kept_by_the_scope():
    study = managed(lambda b: Nested(outer=b.structure, inner=Greedy(), inner_iterations=2))
    with pytest.raises(RuntimeError, match="no candidates left"):
        study.optimize(iterations=3)
    assert len(study.candidates) == 4


def test_schedules_are_checked_before_anything_runs():
    with pytest.raises(ValueError, match="not one of this study's blocks"):
        managed(lambda b: Alternate(b.numbers, Block(TPE(), SIZE)))
    with pytest.raises(ValueError, match="exists only for some values of manager"):
        managed(lambda b: Nested(outer=b.numbers, inner=Joint()))
    context = Block(TPE(), MANAGER, name="context")
    custom = Study(
        agent=create_managed,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        search=Search(blocks=[context]),
        log=None,
    )
    assert custom.name_of(custom.schedule.outer) == "context"  # rounds over what is structural, whatever its name


# ---- several objectives ----------------------------------------------------------------------------------------------
def test_with_independent_every_copy_learns_from_every_round():
    study = managed(
        lambda b: Nested(outer=b.structure, inner=Joint(), inner_iterations=2),
        objectives=Independent(Maximize("WordCount"), Minimize("tokens")),
    )
    # the baseline, then a round for each objective (the second opens on the baseline), then the first's again
    study.optimize(iterations=6)
    assert [t.objective for t in study.candidates[1:]] == [0, 0, 0, 1, 1, 0]
    for k in (0, 1):
        outer = study.method("structure", k)._studies[()]
        assert len([t for t in outer.trials if t.state.name == "COMPLETE"]) == 4  # the baseline and every round
    assert study.method("numbers", 1).seed == 1002  # copies are seeded apart


def test_with_independent_and_alternate_every_objective_proposes_every_block():
    study = managed(
        lambda b: Alternate(b.numbers, b.text), objectives=Independent(Maximize("WordCount"), Minimize("tokens"))
    )
    study.optimize(iterations=8)
    assert [(t.objective, t.proposed) for t in study.candidates[1:5]] == [
        (0, ("numbers",)),
        (0, ("text",)),
        (1, ("numbers",)),
        (1, ("text",)),
    ]


# ---- the record ------------------------------------------------------------------------------------------------------
def test_the_record_states_constraints_exactly_and_refuses_to_append_to_another_study(tmp_path):
    study = managed(lambda b: Joint(), constraints=[AtMost("tokens", 1234567)], record=tmp_path)
    study.optimize(iterations=1)
    log = events(tmp_path)
    assert [e["type"] for e in log] == ["study", "run", "candidate", "scope", "candidate", "scope_end", "finished"]
    head = log[0]
    assert (head["format"], head["schema"]) == ("strands_evals.optimization", 2)
    assert head["constraints"][0]["name"] == "tokens <= 1234567"
    assert head["metrics"]["WordCount"] == {"role": "evaluator", "range": [0, None]}
    assert head["metrics"]["tokens"] == {"role": "usage", "range": [0.0, None]}
    assert head["block_order"] == ["structure", "numbers", "text"]
    assert head["blocks"]["numbers"]["method"] == {"kind": "TPE", "seed": 2, "n_startup_trials": 4}
    candidate = log[4]
    assert (
        candidate["violations"] == [study.candidates[1].metrics["tokens"] - 1234567] and candidate["feasible"] is True
    )
    assert candidate["told"] == [[study.candidates[1].metrics["WordCount"]]] and candidate["best_after"]
    assert candidate["trials"][0]["transcript"]["format"] == "strands-messages"
    again = managed(lambda b: Joint(), record=tmp_path)  # building a study writes nothing
    with pytest.raises(FileExistsError):
        again.optimize(iterations=0)


class Interrupted(Schedule):
    """Stops the study inside a round (as when the machine running it goes away)."""

    def run(self, scope):
        with scope.child("inner", Joint(), blocks=["numbers", "text"], limit=4, round=0, outer="structure") as round_:
            round_.candidate(["structure"], role="outer", defer=["structure"])
            round_.candidate(["numbers", "text"])
            raise RuntimeError("reclaimed")


def test_an_interrupted_round_stays_open_in_the_record(tmp_path):
    study = managed(lambda b: Interrupted(), record=tmp_path)
    with pytest.raises(RuntimeError, match="reclaimed"):
        study.optimize(iterations=5)
    log = events(tmp_path)
    assert [e["type"] for e in log[-3:]] == ["candidate", "candidate", "stopped"]  # no told FAIL, no scope_end
    assert not [e for e in log if e["type"] in ("told", "scope_end")]


def test_each_landscape_draws_its_own_startup_points():
    tpe = TPE(seed=7)
    a, b = tpe._sampler(0), tpe._sampler(1)
    assert a._rng.rng.randint(1 << 30) != b._rng.rng.randint(1 << 30)


# ---- patience --------------------------------------------------------------------------------------------------------
def test_patience_stops_a_run_once_the_best_stops_changing(tmp_path):
    # the first proposal is the best this space has; nothing after it changes the best
    order = Choose(*({"stanzas": n} for n in [6, 1, 2, 3, 4, 5]))
    study = Study(
        agent=create_poet,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        search=Search(blocks=[Block(order, STANZAS)]),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=19, patience=3)
    assert len(study.candidates) == 5  # the baseline, the new best, then 3 candidates that changed nothing
    log = events(tmp_path)
    assert log[1] == {"type": "run", "started_at": log[1]["started_at"], "iterations": 19, "patience": 3}
    assert log[-1]["type"] == "finished" and log[-1]["reason"] == "patience"


def test_without_patience_a_run_uses_every_trial():
    stanzas = Int(1, 60, default=2)

    @tunable(stanzas=stanzas)
    def create_epic(stanzas):
        return Agent(model=Says(" ".join(["word"] * stanzas)), callback_handler=None)

    study = Study(
        agent=create_epic,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        search=Search(blocks=[Block(TPE(seed=1), stanzas)]),
        log=None,
    )
    study.optimize(iterations=7)
    assert len(study.candidates) == 8


# ---- repeats ---------------------------------------------------------------------------------------------------------
def test_values_already_measured_are_not_measured_again_and_a_run_with_nothing_new_ends_exhausted(tmp_path):
    study = Study(
        agent=create_poet,  # 6 values in all
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        search=Search(blocks=[Block(TPE(seed=1), STANZAS)]),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=19)
    measured = [t.params["stanzas"] for t in study.candidates]
    assert len(set(measured)) == len(measured) <= 6
    log = events(tmp_path)
    repeats = [e for e in log if e["type"] == "repeat"]
    assert repeats and all(e["role"] == "step" and e["told"] == ["numbers"] for e in repeats)
    # each repeat is a measured candidate's values, and its draw was told that candidate's result
    assert all(e["values"] == [study.candidates[e["candidate"]].metrics["WordCount"]] for e in repeats)
    assert log[-1]["reason"] == "exhausted"


def test_a_step_that_repeats_proposes_again():
    order = Choose({"stanzas": 2}, {"stanzas": 3})  # the first proposal is the baseline's value
    study = Study(
        agent=create_poet,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        search=Search(blocks=[Block(order, STANZAS)]),
        log=None,
    )
    study.optimize(iterations=1)
    assert [t.params["stanzas"] for t in study.candidates] == [2, 3]
    assert order.told == [0, 1]  # the repeated draw learned the baseline's result; the next draw, its own
