from enum import Enum
from pydantic import BaseModel, Field
from typing_extensions import TypeVar

from strands import Agent

from ..types.evaluation import EvaluationData, EvaluationOutput
from .evaluator import Evaluator
from ..types.trace import EvaluationLevel, TurnLevelInput
from .prompt_templates.helpfulness import get_template

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class HelpfulnessScore(str, Enum):
    """Categorical helpfulness ratings."""
    NOT_HELPFUL = "Not helpful at all"
    VERY_UNHELPFUL = "Very unhelpful"
    SOMEWHAT_UNHELPFUL = "Somewhat unhelpful"
    NEUTRAL = "Neutral/Mixed"
    SOMEWHAT_HELPFUL = "Somewhat helpful"
    VERY_HELPFUL = "Very helpful"
    ABOVE_AND_BEYOND = "Above and beyond"


class HelpfulnessRating(BaseModel):
    """Structured output for helpfulness evaluation."""
    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: HelpfulnessScore = Field(description="Categorical helpfulness rating")


class HelpfulnessEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates helpfulness of agent responses from the user's perspective."""
    
    evaluation_level = EvaluationLevel.TURN_LEVEL

    _score_mapping = {
        HelpfulnessScore.NOT_HELPFUL: 0.0,
        HelpfulnessScore.VERY_UNHELPFUL: 0.167,
        HelpfulnessScore.SOMEWHAT_UNHELPFUL: 0.333,
        HelpfulnessScore.NEUTRAL: 0.5,
        HelpfulnessScore.SOMEWHAT_HELPFUL: 0.667,
        HelpfulnessScore.VERY_HELPFUL: 0.833,
        HelpfulnessScore.ABOVE_AND_BEYOND: 1.0
    }

    def __init__(
        self,
        version: str = "v0",
        model: str | None = None,
        system_prompt: str | None = None,
        include_inputs: bool = True,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model
        self.include_inputs = include_inputs

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        parsed_input = self._parse_trajectory(evaluation_case)
        prompt = self._format_prompt(parsed_input)
        
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        rating = evaluator_agent.structured_output(HelpfulnessRating, prompt)
        
        normalized_score = self._score_mapping[rating.score]
        return EvaluationOutput(
            score=normalized_score,
            test_pass=normalized_score >= 0.5,
            reason=rating.reasoning
        )

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        parsed_input = self._parse_trajectory(evaluation_case)
        prompt = self._format_prompt(parsed_input)
        
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        rating = await evaluator_agent.structured_output_async(HelpfulnessRating, prompt)
        
        normalized_score = self._score_mapping[rating.score]
        return EvaluationOutput(
            score=normalized_score,
            test_pass=normalized_score >= 0.5,
            reason=rating.reasoning
        )
    
    def _format_prompt(self, parsed_input: TurnLevelInput) -> str:
        """Format evaluation prompt from parsed turn data."""
        parts = []
        
        if parsed_input.conversation_history:
            history_str = "\n".join([
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in parsed_input.conversation_history
            ])
            parts.append(f"# Previous turns:\n{history_str}")
        
        parts.append(f"# Target turn to evaluate:\nUser: {parsed_input.user_prompt}\nAssistant: {parsed_input.agent_response}")
        
        return "\n\n".join(parts)
