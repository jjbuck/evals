"""Optimize how a Strands agent picks its model: no router, an LLM router, or strands decider on SageMaker.

One study searches three things:

    router        which method routes: "none" (one fixed model), "llm" (an LLM picks), "decider" (strands decider)
    llm prompt    the LLM router's system prompt          (a TextValue, rewritten by Reflection)
    decider       the decider question's instructions     (a TextValue, rewritten by Reflection)

Routing happens once per conversation: on the agent's first model call, the router reads the first user prompt and
picks a model from POOL; every later call of that conversation goes to the same model. RoutedModel does this inside
the agent, so it works with any benchmark's own task loop (a simulated user, a world), not only `agent(case.input)`.

The goal is a joint Pareto front of reward and dollars. Dollars = the agent's tokens at the chosen model's prices +
the router's own cost (the LLM router's tokens, or a fixed price per decider call). With reward alone the study would
just ship the strongest model with no router. The study's own token count (``Result.cost``) does not include the
LLM router's tokens; the dollars do.

The "decider" router joins the search only when ROUTING_DECIDER_ENDPOINT names a SageMaker endpoint that serves
strands decider (https://github.com/strands-labs/strands-decider, ``POST /v1/systemone``). A study refuses to start
until every price it needs is set.

    python examples/optimization/model_routing.py --selfcheck          # offline, no AWS
    BYPASS_TOOL_CONSENT=true python examples/optimization/model_routing.py --benchmark toy --iterations 3

A benchmark plugs in through load_benchmark(): it returns an Experiment whose evaluators include the reward
evaluator, and the task that runs one case (the default: agent(case.input)).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import boto3
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.models.model import Model

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    Choice,
    Maximize,
    Minimize,
    OneOf,
    Pareto,
    Split,
    Study,
    TextValue,
    TrialInvalid,
    component,
    tunable,
)
from strands_evals.optimization.trials import invoke
from strands_evals.types.evaluation import EnvironmentState, EvaluationOutput

# ---------------------------------------------------------------------------------------------------------------------
# The model pool. Prices in USD per million tokens, by Strands usage key.
# ponytail: placeholders (None) on purpose; a cost study with made-up prices is worse than one that refuses to run.
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolModel:
    model_id: str
    description: str  # what the routers are told about this model
    prices: dict[str, float | None] = field(
        default_factory=lambda: dict.fromkeys(
            ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens")
        )
    )


POOL: dict[str, PoolModel] = {
    "small": PoolModel(
        os.environ.get("ROUTING_SMALL_MODEL", "global.anthropic.claude-haiku-5-5"),
        "fast and cheap; good for short, well-specified tasks",
    ),
    "large": PoolModel(
        os.environ.get("ROUTING_LARGE_MODEL", "global.anthropic.claude-sonnet-5-5"),
        "slower and more expensive; better at long, ambiguous or multi-step tasks",
    ),
}
DEFAULT_MODEL = "small"
MAX_TOKENS = 32_768

ROUTER_LLM = os.environ.get("ROUTING_ROUTER_MODEL", "global.anthropic.claude-haiku-5-5")
ROUTER_LLM_PRICES: dict[str, float | None] = dict.fromkeys(("inputTokens", "outputTokens"))

# A SageMaker endpoint that serves strands decider, such as "strands-decider-2b-qwen3-5-v1-2610". Unset: no decider.
DECIDER_ENDPOINT = os.environ.get("ROUTING_DECIDER_ENDPOINT")
# ponytail: the endpoint's hourly price / the calls it serves an hour; replace with a measured figure.
DECIDER_PRICE_PER_CALL: float | None = None


def unpriced() -> list[str]:
    """The prices a study needs that are not set yet."""
    missing = [f"POOL[{label!r}] {k}" for label, m in POOL.items() for k, v in m.prices.items() if v is None]
    missing += [f"ROUTER_LLM_PRICES {k}" for k, v in ROUTER_LLM_PRICES.items() if v is None]
    if DECIDER_ENDPOINT and DECIDER_PRICE_PER_CALL is None:
        missing.append("DECIDER_PRICE_PER_CALL")
    return missing


def dollars(usage: dict[str, Any], prices: dict[str, float | None]) -> float:
    """Usage (Strands keys) at prices per million tokens. A missing price for a used kind is an error."""
    total = 0.0
    for key, n in usage.items():
        if key not in prices or not n:
            continue
        if prices[key] is None:
            raise ValueError(f"no price for {key}: set the prices in POOL / ROUTER_LLM_PRICES in {__file__}")
        total += n * prices[key] / 1e6
    return total


# ---------------------------------------------------------------------------------------------------------------------
# Routers. Each picks a POOL label from the conversation's first prompt and says what it cost.
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Route:
    model: str
    router: str
    detail: str = ""
    cost: float = 0.0  # the router's own cost, USD
    seconds: float = 0.0


def first_prompt(messages: list[dict[str, Any]]) -> str:
    """The text of the first user message."""
    for message in messages:
        if message.get("role") == "user":
            return "\n".join(b["text"] for b in message.get("content", []) if "text" in b)
    return ""


class NoRouter:
    """No router: one model for every conversation."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    async def route(self, prompt: str) -> Route:
        return Route(self.model, "none")


LLM_ROUTER_PROMPT = TextValue(
    default=(
        "You route a user's request to one of these models:\n"
        + "\n".join(f"- {label}: {m.description}" for label, m in POOL.items())
        + "\nAnswer with the model's name only."
    ),
    about="the system prompt of an LLM that reads a user's first message and names the model to answer it",
    max_chars=4000,
)


def parse_label(text: str) -> str | None:
    """The first POOL label in a router's reply, or None."""
    words = [w.strip(".,:;\"'`*").lower() for w in text.split()]
    return next((w for w in words if w in POOL), None)


class LLMRouter:
    """An LLM reads the first prompt and names a model. A reply with no model name routes to the default."""

    def __init__(self, prompt: str, model_id: str = ROUTER_LLM):
        self.prompt, self.model_id = prompt, model_id

    async def route(self, prompt: str) -> Route:
        start = time.monotonic()
        agent = Agent(
            model=BedrockModel(model_id=self.model_id, max_tokens=64),
            system_prompt=self.prompt,
            callback_handler=None,
        )
        reply = str(await agent.invoke_async(prompt))
        usage = dict(agent.event_loop_metrics.accumulated_usage)
        label = parse_label(reply)
        return Route(
            label or DEFAULT_MODEL,
            "llm",
            f"replied {reply.strip()[:80]!r}" + ("" if label else f"; no model name, so {DEFAULT_MODEL}"),
            dollars(usage, ROUTER_LLM_PRICES),
            time.monotonic() - start,
        )


DECIDER_INSTRUCTIONS = TextValue(
    default="Which model should answer this request?",
    about="the instructions of a strands decider choice question that picks the model for a user's first message",
    max_chars=2000,
)


def decider_request(prompt: str, instructions: str) -> dict[str, Any]:
    """The /v1/systemone body: one choice question over the pool."""
    criteria = {label: m.description for label, m in POOL.items()}
    return {
        "state": prompt,
        "questions": {"route": {"type": "choice", "instructions": instructions, "criteria": criteria}},
    }


class DeciderRouter:
    """strands decider on a SageMaker endpoint answers one choice question. Endpoint errors are infrastructure."""

    def __init__(self, instructions: str, endpoint: str | None = DECIDER_ENDPOINT):
        if not endpoint:
            raise ValueError("set ROUTING_DECIDER_ENDPOINT to a SageMaker endpoint that serves strands decider")
        self.instructions, self.endpoint = instructions, endpoint

    async def route(self, prompt: str) -> Route:
        if DECIDER_PRICE_PER_CALL is None:
            raise ValueError(f"set DECIDER_PRICE_PER_CALL in {__file__}")
        start = time.monotonic()
        try:
            response = await asyncio.to_thread(
                runtime().invoke_endpoint,
                EndpointName=self.endpoint,
                ContentType="application/json",
                Body=json.dumps(decider_request(prompt, self.instructions)),
            )
        except Exception as exc:  # throttling, 5xx, a cold endpoint: the trial measured nothing
            raise TrialInvalid(f"decider endpoint {self.endpoint}: {exc}") from exc
        answer = json.loads(response["Body"].read())["answers"]["route"]
        probabilities = ", ".join(f"{k} {v:.2f}" for k, v in answer["probabilities"].items())
        return Route(answer["choice"], "decider", probabilities, DECIDER_PRICE_PER_CALL, time.monotonic() - start)


_runtime = None


def runtime():
    """One sagemaker-runtime client for the process (boto3 clients are thread-safe; sessions are not)."""
    global _runtime
    if _runtime is None:
        _runtime = boto3.client("sagemaker-runtime")
    return _runtime


# ---------------------------------------------------------------------------------------------------------------------
# RoutedModel: routes on the first call, then is the chosen model.
# ---------------------------------------------------------------------------------------------------------------------


class RoutedModel(Model):
    def __init__(self, router: Any, models: dict[str, Model]):
        self.router, self.models, self.route = router, models, None
        self._lock = asyncio.Lock()

    def _target(self) -> Model:
        return self.models[self.route.model if self.route else DEFAULT_MODEL]

    async def _ensure_routed(self, messages: list[dict[str, Any]]) -> Model:
        async with self._lock:
            if self.route is None:
                route = await self.router.route(first_prompt(messages))
                if route.model not in self.models:
                    route = Route(
                        DEFAULT_MODEL, route.router, f"{route.detail}; unknown model {route.model!r}", route.cost
                    )
                self.route = route
        return self._target()

    def update_config(self, **model_config: Any) -> None:
        for model in self.models.values():
            model.update_config(**model_config)

    def get_config(self) -> Any:
        return self._target().get_config()

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        model = await self._ensure_routed(messages)
        async for event in model.stream(messages, tool_specs, system_prompt, **kwargs):
            yield event

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        model = await self._ensure_routed(prompt)
        async for event in model.structured_output(output_model, prompt, system_prompt=system_prompt, **kwargs):
            yield event


def pool_models() -> dict[str, Model]:
    """A fresh Bedrock model per pool entry, for one trial."""
    return {label: BedrockModel(model_id=m.model_id, max_tokens=MAX_TOKENS) for label, m in POOL.items()}


# ---------------------------------------------------------------------------------------------------------------------
# The tunable agent.
# ---------------------------------------------------------------------------------------------------------------------

ROUTERS = {
    "none": component(NoRouter, model=Choice(list(POOL), default=DEFAULT_MODEL)),
    "llm": component(LLMRouter, prompt=LLM_ROUTER_PROMPT),
}
if DECIDER_ENDPOINT:
    ROUTERS["decider"] = component(DeciderRouter, instructions=DECIDER_INSTRUCTIONS)
ROUTER = OneOf(*ROUTERS.values(), labels=list(ROUTERS))


@tunable(router=ROUTER)
def create_agent(router, system_prompt: str | None = None, tools: list | None = None):
    # ponytail: the benchmark's own harness (create_harness, its tools) goes here when a benchmark needs one.
    return Agent(
        model=RoutedModel(router, pool_models()),
        system_prompt=system_prompt,
        tools=tools or [],
        callback_handler=None,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Cost: the task wrapper prices the conversation; the Dollars evaluator reports it.
# ---------------------------------------------------------------------------------------------------------------------


def routed(task=invoke):
    """Run a benchmark's task, then attach the route and the conversation's dollars as environment state."""

    async def run(agent, case):
        result = task(agent, case)
        if inspect.isawaitable(result):
            result = await result
        out = dict(result) if isinstance(result, dict) else {"output": result}
        model: RoutedModel = agent.model
        route = model.route or Route(DEFAULT_MODEL, "unrouted")  # an agent that never called its model
        usage = dict(agent.event_loop_metrics.accumulated_usage)
        agent_cost = dollars(usage, POOL[route.model].prices)
        state = {
            "model": route.model,
            "router": route.router,
            "detail": route.detail,
            "router_cost": route.cost,
            "agent_cost": agent_cost,
            "dollars": agent_cost + route.cost,
        }
        out["environment_state"] = [
            *(out.get("environment_state") or []),
            EnvironmentState(name="routing", state=state),
        ]
        return out

    return run


class Dollars(Evaluator):
    """The conversation's cost in USD. Its reason also tells the rewriter where the router sent the conversation."""

    score_range = (0.0, None)

    def evaluate(self, case):
        state = next(s.state for s in case.actual_environment_state or [] if s.name == "routing")
        reason = (
            f"{state['router']} routed to {state['model']} ({state['detail']}); "
            f"${state['dollars']:.5f} = agent ${state['agent_cost']:.5f} + router ${state['router_cost']:.5f}"
        )
        return [EvaluationOutput(score=state["dollars"], test_pass=True, reason=reason)]

    async def evaluate_async(self, case):
        return self.evaluate(case)


# ---------------------------------------------------------------------------------------------------------------------
# Benchmarks. Each returns (experiment, task, reward evaluator name).
# ---------------------------------------------------------------------------------------------------------------------


class Contains(Evaluator):
    """1 when the expected answer is in the output."""

    def evaluate(self, case):
        ok = str(case.expected_output).lower() in str(case.actual_output).lower()
        return [EvaluationOutput(score=float(ok), test_pass=ok, reason="found" if ok else "missing")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


TOY = [
    ("capital", "What is the capital of France? One word.", "Paris"),
    ("sum", "What is 17 + 25? Number only.", "42"),
    ("primes", "How many primes are there below 100? Number only.", "25"),
    ("days", "How many days are there from 2024-02-27 to 2024-03-02, counting both ends? Number only.", "5"),
    ("letters", "How many times does the letter r appear in 'strawberry raspberry'? Number only.", "6"),
    ("bits", "How many 1 bits does 2026 have in binary? Number only.", "7"),
    ("roman", "Write 1994 in Roman numerals.", "MCMXCIV"),
    ("chess", "How many squares does a standard chessboard have, of every size from 1x1 to 8x8? Number only.", "204"),
]


def load_benchmark(name: str) -> tuple[Experiment, Any, str]:
    if name == "toy":
        cases = [Case(name=n, input=q, expected_output=a) for n, q, a in TOY]
        return Experiment(cases=cases, evaluators=[Contains(name="reward"), Dollars(name="dollars")]), invoke, "reward"
    # ponytail: add τ², AppWorld, ... here, reusing examples/optimization/*_workload.py; keep Dollars in the evaluators.
    raise SystemExit(f"unknown benchmark {name!r}")


def make_study(args) -> Study:
    if missing := unpriced():  # before any model call: a trial that cannot be priced is spent and measures nothing
        raise SystemExit(f"set these prices in {__file__}: {', '.join(missing)}")
    experiment, task, reward = load_benchmark(args.benchmark)
    return Study(
        agent=create_agent,
        experiment=experiment,
        task=routed(task),
        # One joint Pareto search. "dollars" is an evaluator's score, not a usage or Derived metric, so the study does
        # not treat it as a cost and does not split the search into reward-then-cost phases (PrimaryFirst).
        objectives=Pareto(Maximize(reward), Minimize("dollars")),
        repetitions=args.repetitions,
        split=Split(train=0.5, validation=0.25, test=0.25),
        max_concurrency=args.concurrency,
        record=f"studies/routing-{args.benchmark}-{time.strftime('%Y%m%d-%H%M%S')}",
    )


# ---------------------------------------------------------------------------------------------------------------------
# Offline self-check: routing and pricing plumbing, with a scripted model and router.
# ---------------------------------------------------------------------------------------------------------------------


def selfcheck() -> None:
    class Scripted(Model):
        def __init__(self, text):
            self.text, self.calls = text, 0

        def update_config(self, **model_config):
            pass

        def get_config(self):
            return {}

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            self.calls += 1
            for event in (
                {"messageStart": {"role": "assistant"}},
                {"contentBlockStart": {"start": {}}},
                {"contentBlockDelta": {"delta": {"text": self.text}}},
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "end_turn"}},
                {
                    "metadata": {
                        "usage": {"inputTokens": 1000, "outputTokens": 100, "totalTokens": 1100},
                        "metrics": {"latencyMs": 1},
                    }
                },
            ):
                yield event

        async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
            raise NotImplementedError
            yield

    class Always:
        async def route(self, prompt):
            assert prompt == "hard question", prompt
            return Route("large", "scripted", "test", cost=0.001)

    assert parse_label("Large.") == "large" and parse_label("**small**") == "small" and parse_label("neither") is None
    assert decider_request("x", "pick")["questions"]["route"]["criteria"].keys() == POOL.keys()

    saved = {label: dict(m.prices) for label, m in POOL.items()}
    for m in POOL.values():
        m.prices.update(inputTokens=1.0, outputTokens=5.0, cacheReadInputTokens=0.1, cacheWriteInputTokens=1.25)
    try:
        small, large = Scripted("from small"), Scripted("from large")
        agent = Agent(model=RoutedModel(Always(), {"small": small, "large": large}), callback_handler=None)
        out = asyncio.run(routed()(agent, Case(name="c", input="hard question")))
        asyncio.run(agent.invoke_async("follow-up"))  # the second call goes to the routed model, not a new route
    finally:  # the test prices must not reach a study
        for label, prices in saved.items():
            POOL[label].prices.update(prices)
    assert "from large" in out["output"] and large.calls == 2 and small.calls == 0, (large.calls, small.calls)

    state = out["environment_state"][-1].state
    assert abs(state["agent_cost"] - (1000 * 1.0 + 100 * 5.0) / 1e6) < 1e-12, state
    assert abs(state["dollars"] - state["agent_cost"] - 0.001) < 1e-12, state
    data = type("D", (), {"actual_environment_state": out["environment_state"]})()
    [scored] = Dollars().evaluate(data)
    assert scored.score == state["dollars"] and "routed to large" in scored.reason, scored
    print("selfcheck ok:", scored.reason)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selfcheck", action="store_true", help="offline check of routing and pricing; no AWS")
    parser.add_argument("--benchmark", default="toy")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        study = make_study(args)
        study.optimize(iterations=args.iterations)
        print("ships:", study.best_candidate.params)  # noqa: T201
