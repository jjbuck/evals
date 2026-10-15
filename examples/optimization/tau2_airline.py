"""Tune Claude Haiku 4.5 as τ²-bench's airline customer-service agent: its instructions, its tools' descriptions, and
how much it thinks before it acts.

    BYPASS_TOOL_CONSENT=true python examples/optimization/tau2_airline.py

Each case is one of τ²'s airline tasks. τ² runs the conversation (tau2_workload.py): its simulated user talks to the
agent, τ² executes every tool call the agent makes, and τ²'s reward (1 when the airline's records end as the task
requires) is the score. The baseline is τ²'s own agent: its instruction, temperature 0, no thinking, the tools' own
descriptions. tau2_airline_steps.py runs a ladder of studies like this one, and checks the best on held-out tasks.

(The texts are rewritten by contrastive reflection, whose agent runs a shell over the traces: run this where
model-written shell commands are acceptable, such as a devbox or a container.)
"""

import time

from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.models import CacheConfig
from strands.models.bedrock import BedrockModel
from tau2_workload import AGENT_INSTRUCTION, Reward, Tau2Task, cases, policy, tools

from strands_evals import Experiment
from strands_evals.optimization import Int, Maximize, Study, TextValue, tool_descriptions, tunable

POLICY = policy("airline")  # the airline's rules, part of the benchmark: never tuned


@tunable(
    instructions=TextValue(
        default=AGENT_INSTRUCTION,
        max_chars=4000,
        about="the agent's instructions, given before the airline's policy",
    ),
    tools=tool_descriptions(*tools("airline")),  # one text per tool, starting from τ²'s own description
    thinking_budget=Int(0, 16384, step=1024, default=0),  # tokens of thinking per model call; 0 for none
)
def create_agent(instructions, tools, thinking_budget):
    if thinking_budget:  # thinking runs at the model's own temperature
        settings = {"additional_request_fields": {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}}
    else:
        settings = {"temperature": 0.0}
    model = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        max_tokens=thinking_budget + 4096,
        cache_config=CacheConfig(strategy="auto"),
        **settings,
    )
    return Agent(
        model=model,
        system_prompt=f"<instructions>\n{instructions}\n</instructions>\n<policy>\n{POLICY}\n</policy>",
        tools=tools,
        conversation_manager=NullConversationManager(),  # the whole conversation, as τ²'s own agent keeps it
        callback_handler=None,
    )


study = Study(
    agent=create_agent,
    experiment=Experiment(cases=cases("airline", "train", 10), evaluators=[Reward(name="reward")]),
    task=Tau2Task("airline"),  # each trial is one τ² conversation, not one call of the agent
    objectives=Maximize("reward"),
    max_concurrency=10,  # every case at once
    record=f"studies/tau2-airline-{time.strftime('%Y%m%d-%H%M%S')}",
)

if __name__ == "__main__":
    study.optimize(iterations=10)
