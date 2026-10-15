"""The τ²-bench workload: a Strands agent as τ²'s customer-service agent, τ²'s tasks as cases, and τ²'s reward as the
score. Not a study: see tau2_airline.py (one study) and tau2_airline_steps.py (a ladder of them).

τ² runs each conversation. Its simulated user talks to the agent, and it executes every tool call the agent makes
itself, because its evaluator replays those calls against a fresh database. So the agent never runs a τ² tool: a hook
interrupts each call, τ² executes it, and τ²'s result resumes the agent. Everything else (system prompt, model,
conversation manager, hooks) is the agent's own, built by the study's factory.

    pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench"
"""

from __future__ import annotations

import json
import random
import sys
from functools import cache
from typing import Any

from loguru import logger
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry, MessageAddedEvent
from strands.tools.tools import PythonAgentTool
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.environment.toolkit import ToolType, get_tool_types
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, get_tasks, load_task_splits

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import TrialInvalid
from strands_evals.types.evaluation import EvaluationOutput

__all__ = ["AGENT_INSTRUCTION", "ConfirmWrites", "DirectWrites", "Reward", "Tau2Task", "cases", "policy", "tools"]

logger.remove()  # τ² logs every message it routes
logger.add(sys.stderr, level="ERROR")

# τ²'s simulated user: the same for every candidate, so only the agent varies (τ²'s leaderboard uses gpt-4.1).
USER_LLM = "bedrock/us.anthropic.claude-sonnet-4-6"
NO_REPLY = "(no reply)"
NUDGE = "(Your last turn contained no message. Reply to the customer or call a tool.)"


@cache
def _environment(domain: str) -> Any:
    return build_environment(domain)


def policy(domain: str) -> str:
    """The domain's policy, which the agent must follow (fixed: it is part of the benchmark)."""
    return _environment(domain).get_policy()


def tools(domain: str) -> list[PythonAgentTool]:
    """The domain's tools as the agent sees them: their names, descriptions and inputs. τ² executes every call."""

    def executed_by_tau2(tool_use: dict[str, Any], **_: Any) -> dict[str, Any]:
        raise RuntimeError(f"{tool_use['name']} ran outside τ²: run the agent with Tau2Task")

    out = []
    for t in _environment(domain).get_tools():
        fn = t.openai_schema["function"]
        spec = {"name": fn["name"], "description": fn["description"], "inputSchema": {"json": fn["parameters"]}}
        out.append(PythonAgentTool(fn["name"], spec, executed_by_tau2))
    return out


def write_tools(domain: str) -> frozenset[str]:
    """The tools that change the customer's records."""
    return frozenset(n for n, kind in get_tool_types(_environment(domain).tools).items() if kind == ToolType.WRITE)


class DirectWrites(HookProvider):
    """Write calls go to τ² as the agent makes them."""

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        pass


class ConfirmWrites(HookProvider):
    """Each write call is returned unexecuted once, with ``message``; the agent re-checks it and calls it again."""

    def __init__(self, message: str, domain: str = "airline") -> None:
        self.message, self.writes = message, write_tools(domain)
        self._reviewed: set[str] = set()
        self._held: set[str] = set()  # by toolUseId: a call resumed after an interrupt gets the same decision

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        use = event.tool_use
        if use["name"] not in self.writes:
            return
        signature = json.dumps([use["name"], use["input"]], sort_keys=True)
        if use["toolUseId"] in self._held or signature not in self._reviewed:
            self._held.add(use["toolUseId"])
            self._reviewed.add(signature)
            event.cancel_tool = self.message


class _DeferToTau2(HookProvider):
    """Interrupts each τ² tool call; on resume, the call returns the result τ² gave."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        use = event.tool_use
        if use["name"] not in self.names or event.cancel_tool:
            return
        result = event.interrupt(f"tau2:{use['toolUseId']}", reason={**use})
        text, failed = result["content"] or "", bool(result["error"])

        def returns(tool_use: dict[str, Any], **_: Any) -> dict[str, Any]:
            status = "error" if failed else "success"
            return {"toolUseId": tool_use["toolUseId"], "status": status, "content": [{"text": text}]}

        event.selected_tool = PythonAgentTool(use["name"], event.selected_tool.tool_spec, returns)


class _NoThinkingTail(HookProvider):
    """Bedrock rejects an assistant message that ends in a reasoning block, which a turn occasionally does."""

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(MessageAddedEvent, self._on_message)

    def _on_message(self, event: MessageAddedEvent) -> None:
        content = event.message.get("content") or []
        if event.message.get("role") != "assistant" or not content or "reasoningContent" not in content[-1]:
            return
        while content and "reasoningContent" in content[-1] and any("reasoningContent" not in b for b in content):
            content.pop()
        if all("reasoningContent" in b for b in content):
            content.append({"text": NO_REPLY})


class _Bridge(HalfDuplexAgent[None]):
    """τ²'s agent interface over a Strands agent: τ² tool calls leave as interrupts and come back as their results."""

    def __init__(self, agent: Any, tau2_tools: list[Any], domain_policy: str) -> None:
        super().__init__(tools=tau2_tools, domain_policy=domain_policy)
        self.agent = agent
        agent.hooks.add_hook(_DeferToTau2({t.name for t in tau2_tools}))
        agent.hooks.add_hook(_NoThinkingTail())
        self._pending: list[Any] = []

    def get_init_state(self, message_history: list[Message] | None = None) -> None:
        if message_history:  # τ² opens with a scripted greeting; Bedrock needs a user turn first
            text = "\n".join(m.content or "" for m in message_history if isinstance(m, AssistantMessage))
            self.agent.messages = [
                {"role": "user", "content": [{"text": "(customer connected)"}]},
                {"role": "assistant", "content": [{"text": text}]},
            ]

    def generate_next_message(self, message: ValidAgentInputMessage, state: None) -> tuple[AssistantMessage, None]:
        if isinstance(message, UserMessage):
            prompt: Any = message.content or ""
        else:
            results = {m.id: m for m in (message.tool_messages if isinstance(message, MultiToolMessage) else [message])}
            prompt = [
                {
                    "interruptResponse": {
                        "interruptId": i.id,
                        "response": {
                            "content": results[i.reason["toolUseId"]].content,
                            "error": results[i.reason["toolUseId"]].error,
                        },
                    }
                }
                for i in self._pending
            ]
        result = self.agent(prompt)
        for _ in range(2):
            if result.stop_reason == "interrupt" or self._text(result):
                break
            result = self.agent(NUDGE)
        if result.stop_reason == "interrupt":
            self._pending = list(result.interrupts)
            calls = [
                ToolCall(id=i.reason["toolUseId"], name=i.reason["name"], arguments=i.reason["input"])
                for i in self._pending
            ]
            return AssistantMessage(role="assistant", content=None, tool_calls=calls), state
        self._pending = []
        text = self._text(result)
        if not text:
            raise RuntimeError("the agent gave no reply after two nudges")
        return AssistantMessage(role="assistant", content=text), state

    @staticmethod
    def _text(result: Any) -> str:
        text = "".join(b.get("text", "") for b in result.message.get("content", [])).strip()
        return "" if text == NO_REPLY else text


class Tau2Run:
    """A finished conversation; its text is how it ended and the agent's last message. (A plain object: Strands Evals
    keeps a task's output as it is.)"""

    def __init__(self, simulation: SimulationRun, domain: str) -> None:
        self.simulation, self.domain = simulation, domain

    def __str__(self) -> str:
        said = [m.content for m in self.simulation.messages if isinstance(m, AssistantMessage) and m.content]
        reason = getattr(self.simulation.termination_reason, "value", self.simulation.termination_reason)
        return f"{reason} after {len(self.simulation.messages)} messages: {(said or [''])[-1][:300]}"


def _user_side(exc: BaseException) -> bool:
    """Whether the failure was τ²'s simulated user's (its LLM calls go through litellm), not the agent's."""
    for e in (exc, exc.__cause__, exc.__context__):
        if e is None:
            continue
        if type(e).__module__.startswith(("litellm", "httpx", "openai")):
            return True
        if isinstance(e, ValueError) and str(e).startswith("UserMessage must have"):
            return True  # the simulated user produced an empty turn
    return False


class Tau2Task:
    """The study's task: one τ² conversation between the agent and τ²'s simulated user, on the case's τ² task."""

    def __init__(self, domain: str = "airline", user_llm: str = USER_LLM, max_steps: int = 200, max_errors: int = 10):
        self.domain, self.user_llm, self.max_steps, self.max_errors = domain, user_llm, max_steps, max_errors

    def __call__(self, agent: Any, case: Case) -> dict[str, Any]:
        task = _task(self.domain, case.metadata["task_id"])
        env = build_environment(self.domain)
        bridge = _Bridge(agent, env.get_tools(), env.get_policy())
        try:
            user = build_user("user_simulator", env, task, llm=self.user_llm, llm_args={"temperature": 0.0})
            orchestrator = Orchestrator(
                domain=self.domain,
                agent=bridge,
                user=user,
                environment=env,
                task=task,
                max_steps=self.max_steps,
                max_errors=self.max_errors,
            )
            simulation = orchestrator.run()
        except Exception as exc:
            if _user_side(exc):
                raise TrialInvalid(f"τ²'s simulated user failed: {type(exc).__name__}: {exc}") from exc
            raise
        if simulation.termination_reason in (TerminationReason.USER_ERROR, TerminationReason.INFRASTRUCTURE_ERROR):
            raise TrialInvalid(f"τ² ended the conversation: {simulation.termination_reason.value}")
        return {"output": Tau2Run(simulation, self.domain)}


@cache
def _task(domain: str, task_id: str) -> Any:
    return get_tasks(domain, task_ids=[task_id])[0]


class Reward(Evaluator):
    """τ²'s reward for the conversation: 1 when the database ends as the task requires and the agent told the user
    what it had to (airline), else 0."""

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        run: Tau2Run = case.actual_output
        task = _task(run.domain, case.metadata["task_id"])
        info = evaluate_simulation(
            simulation=run.simulation, task=task, evaluation_type=EvaluationType.ALL, solo_mode=False, domain=run.domain
        )
        parts = ", ".join(f"{getattr(k, 'value', k)}={v:g}" for k, v in (info.reward_breakdown or {}).items())
        return [EvaluationOutput(score=info.reward, test_pass=info.reward == 1.0, reason=parts or str(info.info))]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return self.evaluate(case)


def cases(domain: str, split: str, n: int, seed: int = 0) -> list[Case]:
    """``n`` of the domain's tasks from τ²'s ``split`` ("train" or "test"), drawn with ``seed``."""
    ids = sorted(load_task_splits(domain)[split], key=lambda i: (len(i), i))
    chosen = sorted(random.Random(seed).sample(ids, n), key=lambda i: (len(i), i))
    return [
        Case(name=f"{domain}-{i}", input=f"τ² {domain} task {i}", metadata={"domain": domain, "task_id": i})
        for i in chosen
    ]
