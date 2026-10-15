"""Optimize a Strands agent against a Strands Evals experiment.

    @tunable(
        system_prompt=TextValue(default="You answer support tickets concisely."),
        conversation_manager=OneOf(
            component(SlidingWindowConversationManager, window_size=Int(10, 80, default=40)),
            component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)),
        ),
    )
    def create_agent(system_prompt, conversation_manager): ...

    study = Study(agent=create_agent, experiment=experiment, repetitions=2,
                  objectives=Maximize("Helpfulness"), constraints=[AtMost("tokens", 4000)])
    study.optimize(iterations=30)
    study.best_candidate.params

A study evaluates **candidates**: values for the agent's tunable arguments. Each candidate runs every case of the
experiment ``repetitions`` times; each run is a **trial**, with a status (an agent that raised, or an evaluator that
failed, is a status, never a zero score). Estimates are case-level means with 95% intervals.

A **search** says how candidates are proposed: **blocks**, each a method and the variables it proposes (the objects
declared in ``@tunable``: ``Block(TPE(), window_size)``, ``Block(Reflection(), system_prompt, tools)``), and a
**schedule** that says which blocks change in each iteration, holding the rest at the incumbent: ``Joint()`` (all of
them), ``Alternate(...)`` (one at a time), or ``Nested(outer=..., inner=...)`` (rounds: an outer proposal, then a
bounded inner search, compared at its best; nests to any depth). By default every variable is in a block for its type
(choices and numbers by Optuna's TPE, ``TextValue`` by contrastive reflection, ``SkillLibrary`` by skill-library
curation) and the schedule is built from the space.

Several objectives are given wrapped in how they combine: ``Pareto(...)`` (jointly), ``Independent(...)`` (one copy of
every method per objective), ``WeightedSum(...)``, ``Chebyshev(...)`` or ``Lexicographic(...)``. A goal names a metric:
an evaluator's, the agent's usage (``tokens``, ``seconds``, ...), or a ``Derived`` one computed from those. With
``record=``, the study writes every candidate as it finishes, which the explorer reads.
"""

from .methods import TPE, Block, Draw, Method, Optuna, Reflection, SkillCuration
from .projection import (
    AtLeast,
    AtMost,
    Chebyshev,
    Combine,
    Derived,
    Estimate,
    Independent,
    Lexicographic,
    Maximize,
    Minimize,
    Pareto,
    WeightedSum,
)
from .schedules import Alternate, Joint, Nested, Schedule
from .search import Search
from .space import (
    Choice,
    Component,
    Float,
    Int,
    OneOf,
    SkillLibrary,
    Switch,
    TextValue,
    TunableFactory,
    Values,
    component,
    tool_descriptions,
    tunable,
)
from .study import Candidate, Scope, Study
from .trials import Trial, TrialInvalid, TrialStatus

__all__ = [
    "Alternate",
    "AtLeast",
    "AtMost",
    "Block",
    "Candidate",
    "Chebyshev",
    "Choice",
    "Combine",
    "Component",
    "Derived",
    "Draw",
    "Estimate",
    "Float",
    "Independent",
    "Int",
    "Joint",
    "Lexicographic",
    "Maximize",
    "Method",
    "Minimize",
    "Nested",
    "OneOf",
    "Optuna",
    "Pareto",
    "Reflection",
    "Schedule",
    "Search",
    "Scope",
    "SkillCuration",
    "SkillLibrary",
    "Study",
    "Switch",
    "TPE",
    "TextValue",
    "Trial",
    "TrialInvalid",
    "TrialStatus",
    "TunableFactory",
    "Values",
    "WeightedSum",
    "component",
    "tool_descriptions",
    "tunable",
]
