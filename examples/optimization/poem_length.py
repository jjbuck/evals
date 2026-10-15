"""Steel thread: learn a system prompt that makes a Strands agent write poems of 40 to 60 words.

The system prompt is rewritten by contrastive reflection, whose agent uses a shell over the traces: run this where
model-written shell commands are acceptable (a devbox or a container).

    BYPASS_TOOL_CONSENT=true python examples/optimization/poem_length.py
"""

import time

from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import Maximize, Study, TextValue, tunable
from strands_evals.types.evaluation import EvaluationOutput


@tunable(system_prompt=TextValue(default="You write poems about the topic you are given."))
def create_poet(system_prompt):
    return Agent(system_prompt=system_prompt, callback_handler=None)


class WordCount(Evaluator):
    """Passes when the poem has 40 to 60 words."""

    def evaluate(self, case):
        words = len(str(case.actual_output).split())
        passed = 40 <= words <= 60
        return [EvaluationOutput(score=float(passed), test_pass=passed, reason=f"{words} words (40 to 60 passes)")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


topics = ["the sea", "a city at night", "first snow", "an old dog", "the last train home", "a garden in August"]
experiment = Experiment(cases=[Case(name=t, input=t) for t in topics], evaluators=[WordCount()])

study = Study(
    agent=create_poet,
    experiment=experiment,
    objectives=Maximize("WordCount"),
    repetitions=2,
    record=f"studies/poem-length-{time.strftime('%Y%m%d-%H%M%S')}",
)

if __name__ == "__main__":
    study.optimize(iterations=4)
    print(study.report())  # noqa: T201
    for candidate in study.candidates[1:]:
        text = candidate.origin.get("text", {})
        print(  # noqa: T201
            f"candidate {candidate.number}: reflection {text.get('status')}, {text.get('trials')} trials, "
            f"search cost {text.get('usage', {}).get('tokens', 0):,.0f} tokens"
        )
    print("\nbest system prompt:\n" + study.best_candidate.params["system_prompt"])  # noqa: T201
