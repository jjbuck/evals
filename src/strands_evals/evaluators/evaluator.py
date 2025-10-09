import inspect

from typing_extensions import Any, Generic, TypeVar

from ..types.evaluation import EvaluationData, EvaluationOutput
from ..types.trace import Session, AgentInvocationSpan, ToolExecutionSpan
from ..types.trace import EvaluationLevel, TurnLevelInput, ToolLevelInput

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Evaluator(Generic[InputT, OutputT]):
    """
    Base class for evaluators.

    Evaluators can assess the performance of a task on all test cases.
    Subclasses must implement the `evaluate` method.
    """

    # Optional: subclasses can set this to enable trace parsing
    evaluation_level: EvaluationLevel | None = None

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases asynchronously.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses, especially if you want to run evaluations asynchronously."
        )
    
    def _parse_turn_level(self, session: Session) -> TurnLevelInput | None:
        """Extract turn-level data from Session."""
        conversation_history = []
        last_turn = None
        
        for trace in session.traces:
            for span in trace.spans:
                if isinstance(span, AgentInvocationSpan):
                    if last_turn:
                        conversation_history.append({
                            "user": last_turn.user_prompt,
                            "assistant": last_turn.agent_response
                        })
                    last_turn = span
        
        if not last_turn:
            return None
        
        return TurnLevelInput(
            user_prompt=last_turn.user_prompt,
            agent_response=last_turn.agent_response,
            conversation_history=conversation_history
        )
    
    def _parse_tool_level(self, session: Session, user_query: str = "") -> list[ToolLevelInput]:
        """Extract tool-level data from Session."""
        tool_inputs = []
        
        for trace in session.traces:
            for span in trace.spans:
                if isinstance(span, ToolExecutionSpan):
                    tool_inputs.append(ToolLevelInput(
                        tool_name=span.tool_call.name,
                        parameters=span.tool_call.parameters,
                        result=span.tool_result.content,
                        user_query=user_query
                    ))
        
        return tool_inputs
    
    def _parse_trajectory(self, evaluation_case: EvaluationData[InputT, OutputT]) -> Any:
        """Parse Session trajectory based on evaluation level."""
        trajectory = evaluation_case.actual_trajectory
        
        if not isinstance(trajectory, Session):
            raise TypeError(
                f"Trace parsing requires actual_trajectory to be a Session object, "
                f"got {type(trajectory).__name__}."
            )
        
        if self.evaluation_level == EvaluationLevel.TURN_LEVEL:
            return self._parse_turn_level(trajectory)
        elif self.evaluation_level == EvaluationLevel.TOOL_LEVEL:
            return self._parse_tool_level(trajectory, str(evaluation_case.input))
        else:
            raise ValueError(f"Unsupported evaluation level: {self.evaluation_level}")

    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the name of the evaluator type.

        Returns:
            str: The name of the evaluator type.
        """
        return cls.__name__

    def to_dict(self) -> dict:
        """
        Convert the evaluator into a dictionary.

        Returns:
            dict: A dictionary containing the evaluator's information. Omit private attributes
            (attributes starting with '_') and attributes with default values.
        """
        _dict = {"evaluator_type": self.get_type_name()}

        # Get default values from __init__ signature
        sig = inspect.signature(self.__init__)
        defaults = {k: v.default for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty}
        for k, v in self.__dict__.items():
            if not k.startswith("_") and (k not in defaults or v != defaults[k]):
                _dict[k] = v
        return _dict
