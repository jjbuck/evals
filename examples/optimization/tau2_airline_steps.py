"""Tune a deliberately weak agent (Claude Haiku 4.5) on τ²-bench airline tasks, in a ladder of studies from its
instructions alone up to its instructions, reasoning, write review and tool descriptions together. (tau2_airline.py is
one plain study.)

    BYPASS_TOOL_CONSENT=true python examples/optimization/tau2_airline_steps.py --study prompt

Each study searches on 10 of τ²'s airline train tasks. Afterwards the baseline and the best candidates run on 10 of
τ²'s airline test tasks, which the search never saw (held_out.json in the study's record). The baseline in every study
is τ²'s own agent: its instruction, temperature 0, no thinking, writes executed as made, the tools' own descriptions.

    prompt          the instructions before the policy (text: reflection)
    numbers         the temperature and the thinking budget (numbers: TPE)
    prompt_numbers  both, each candidate changing both (Joint)
    structure       the instructions; how the model reasons (directly, with a temperature, or thinking, with a budget);
                    whether each write call is held back once for a re-check, and what the agent is told then. Rounds
                    each choose the structure, then search its settings and the texts (Nested)
    full            structure, plus the 14 tools' descriptions, with tokens as a second objective (Pareto)

With limits (constraints), over the numbers space; a candidate that breaks a limit is infeasible, never the best:

    numbers_token_budget  the best reward within 75,000 tokens per conversation
    numbers_latency       the best reward within 40 seconds per conversation (the simulated user's turns included)
    numbers_cheapest      the fewest tokens among candidates with a reward of at least 0.7

(Reflection's agent runs a shell over the traces: run where model-written shell commands are acceptable.)
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models import CacheConfig
from strands.models.bedrock import BedrockModel
from tau2_workload import AGENT_INSTRUCTION, ConfirmWrites, DirectWrites, Reward, Tau2Task, cases, policy, tools

from strands_evals import Experiment
from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Block,
    Float,
    Int,
    Maximize,
    Minimize,
    OneOf,
    Pareto,
    Reflection,
    Search,
    Study,
    TextValue,
    component,
    tool_descriptions,
    tunable,
)
from strands_evals.optimization.trials import run_trials

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DOMAIN = "airline"
POLICY = policy(DOMAIN)
TOOLS = tools(DOMAIN)
TOOL_TEXTS = tool_descriptions(*TOOLS)  # one text per tool, starting from τ²'s own description

INSTRUCTIONS = TextValue(
    default=AGENT_INSTRUCTION,
    max_chars=4000,
    about="the agent's instructions, given before the airline's policy (the policy itself is fixed)",
)
TEMPERATURE = Float(0.0, 1.0, default=0.0)
THINKING_BUDGET = Int(0, 16384, step=1024, default=0)  # tokens of thinking per model call; 0 for none
CHECK = (
    "NOT EXECUTED. This action changes the customer's records and cannot be undone. Before executing it, re-check it "
    "against the policy: every precondition verified, arguments exactly right, and explicit user confirmation obtained "
    "where the policy requires it. If it is still correct, call it again with the same arguments; otherwise do what "
    "the policy requires instead."
)


def direct(temperature: float) -> dict:
    return {"temperature": temperature}


def thinking(budget_tokens: int) -> dict:
    return {"thinking_budget": budget_tokens}


REASONING = OneOf(
    component(direct, temperature=TEMPERATURE),
    component(thinking, budget_tokens=Int(1024, 16384, log=True, default=4096)),
)
CHECK_MESSAGE = TextValue(default=CHECK, max_chars=1500, about="what the agent is told when a write call is held back")
REVIEW = OneOf(
    component(DirectWrites),
    component(
        ConfirmWrites,
        message=CHECK_MESSAGE,
    ),
    labels=["direct", "confirm"],
)


def model(temperature: float = 0.0, thinking_budget: int = 0) -> BedrockModel:
    cache = CacheConfig(strategy="auto")
    if thinking_budget:  # thinking runs at the model's own temperature
        thinking = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}
        return BedrockModel(
            model_id=HAIKU, max_tokens=thinking_budget + 4096, additional_request_fields=thinking, cache_config=cache
        )
    return BedrockModel(model_id=HAIKU, max_tokens=4096, temperature=temperature, cache_config=cache)


def agent(instructions=AGENT_INSTRUCTION, llm=None, tools=TOOLS, hooks=()) -> Agent:
    return Agent(
        model=llm or model(),
        system_prompt=f"<instructions>\n{instructions}\n</instructions>\n<policy>\n{POLICY}\n</policy>",
        tools=tools,
        hooks=list(hooks),
        conversation_manager=NullConversationManager(),  # the whole conversation, as τ²'s own agent keeps it
        callback_handler=None,
    )


@tunable(instructions=INSTRUCTIONS)
def prompt(instructions):
    return agent(instructions)


@tunable(temperature=TEMPERATURE, thinking_budget=THINKING_BUDGET)
def numbers(temperature, thinking_budget):
    return agent(llm=model(temperature, thinking_budget))


@tunable(instructions=INSTRUCTIONS, temperature=TEMPERATURE, thinking_budget=THINKING_BUDGET)
def prompt_numbers(instructions, temperature, thinking_budget):
    return agent(instructions, model(temperature, thinking_budget))


@tunable(instructions=INSTRUCTIONS, reasoning=REASONING, review=REVIEW)
def structure(instructions, reasoning, review):
    return agent(instructions, model(**reasoning), hooks=[review])


@tunable(instructions=INSTRUCTIONS, reasoning=REASONING, review=REVIEW, tools=TOOL_TEXTS)
def full(instructions, reasoning, review, tools):
    return agent(instructions, model(**reasoning), tools=tools, hooks=[review])


REWARD = Maximize("reward")
# Each study: its factory, its goals, and the texts it rewrites.
STUDIES = {
    "prompt": (prompt, dict(objectives=REWARD), [INSTRUCTIONS]),
    "numbers": (numbers, dict(objectives=REWARD), []),
    "prompt_numbers": (prompt_numbers, dict(objectives=REWARD), [INSTRUCTIONS]),
    "structure": (structure, dict(objectives=REWARD), [INSTRUCTIONS, CHECK_MESSAGE]),
    "full": (
        full,
        dict(objectives=Pareto(Maximize("reward"), Minimize("tokens"))),
        [INSTRUCTIONS, CHECK_MESSAGE, TOOL_TEXTS],
    ),
    "numbers_token_budget": (numbers, dict(objectives=REWARD, constraints=[AtMost("tokens", 75_000)]), []),
    "numbers_latency": (numbers, dict(objectives=REWARD, constraints=[AtMost("seconds", 40)]), []),
    "numbers_cheapest": (numbers, dict(objectives=Minimize("tokens"), constraints=[AtLeast("reward", 0.7)]), []),
}
FACTORIES = {name: factory for name, (factory, _, _) in STUDIES.items()}


def make_study(name: str, n_cases: int, record: str | None) -> Study:
    factory, goals, texts = STUDIES[name]
    # The texts are rewritten by a stronger model than the agent's; every other variable has its default block.
    rewrite = Block(Reflection(model_config={"model_id": "us.anthropic.claude-sonnet-4-6"}), *texts) if texts else None
    return Study(
        agent=factory,
        experiment=Experiment(cases=cases(DOMAIN, "train", n_cases), evaluators=[Reward(name="reward")]),
        task=Tau2Task(DOMAIN),  # one τ² conversation per trial
        **goals,
        search=Search(blocks=[rewrite] if rewrite else []),
        max_concurrency=n_cases,  # every case at once: a candidate takes as long as its longest conversation
        record=record,
    )


def held_out(study: Study, n_cases: int, repetitions: int) -> list[dict]:
    """The baseline and the best candidates on test tasks the search never saw."""
    experiment = Experiment(cases=cases(DOMAIN, "test", n_cases), evaluators=[Reward(name="reward")])
    chosen = [study.baseline, *(c for c in study.best_candidates if c is not study.baseline)]
    out = []
    for c in chosen:
        run = run_trials(
            study.agent,
            c.params,
            experiment,
            repetitions=repetitions,
            max_concurrency=n_cases * repetitions,
            task=study.task,
            retries=study.retries,
        )
        trials = [study.projection.prepare(t) for t in asyncio.run(run)]
        estimates = study.projection.estimates(trials)
        out.append(
            {
                "candidate": c.number,
                "params": c.params,
                "search": {m: e.mean for m, e in c.estimates.items()},
                "held_out": {m: {"mean": e.mean, "ci": e.ci and list(e.ci), "n": e.n} for m, e in estimates.items()},
                "trials": [
                    {"case": t.case, "repetition": t.repetition, "status": t.status.value, **t.metrics}
                    for t in trials
                ],
            }
        )
        means = {m: e.mean for m, e in estimates.items()}
        violations = study.projection.violations(means)
        out[-1]["held_out_feasible"] = all(v is not None and v <= 0 for v in violations)
        goals = "  ".join(f"{m} {means[m]:.3g} (search {c.metrics[m]:.3g})" for m in study.projection.goal_metrics)
        limits = f"  limits {'met' if out[-1]['held_out_feasible'] else 'broken'}" if violations else ""
        print(f"held out: candidate {c.number}: {goals}{limits}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=STUDIES, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cases", type=int, default=10, help="train tasks searched on, and test tasks held out")
    parser.add_argument("--held-out-repetitions", type=int, default=2)
    args = parser.parse_args()
    record = f"studies/tau2-airline-{args.study}-{time.strftime('%Y%m%d-%H%M%S')}"
    study = make_study(args.study, args.cases, record)
    study.optimize(iterations=args.iterations)
    result = held_out(study, args.cases, args.held_out_repetitions)
    Path(record, "held_out.json").write_text(json.dumps(result, indent=1, default=str))
