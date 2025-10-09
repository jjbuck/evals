"""
Generic trace types for agent observability.

These types represent standard observability primitives for agents.
"""

from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(str, Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class EvaluationLevel(str, Enum):
    """Type of evaluation based on trace granularity."""
    TURN_LEVEL = "turn"
    CONVERSATION_LEVEL = "conversation"
    TOOL_LEVEL = "tool"


class TurnLevelInput(BaseModel):
    """Parsed input for turn-level evaluations."""
    user_prompt: str
    agent_response: str
    conversation_history: list[dict[str, str]] = []


class ToolLevelInput(BaseModel):
    """Parsed input for tool-level evaluations."""
    tool_name: str
    parameters: dict[str, Any]
    result: str
    user_query: str = ""
    available_tools: list[str] = []


class ToolCall(BaseModel):
    name: str
    parameters: dict
    tool_call_id: Optional[str] = None


class ToolResult(BaseModel):
    content: str
    error: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolConfig(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class TextContent(BaseModel):
    content_type: ContentType = ContentType.TEXT
    text: str


class ToolCallContent(ToolCall):
    content_type: ContentType = ContentType.TOOL_USE


class ToolResultContent(ToolResult):
    content_type: ContentType = ContentType.TOOL_RESULT


class UserMessage(BaseModel):
    role: Role = Role.USER
    content: List[Union[TextContent, ToolResultContent]]


class AssistantMessage(BaseModel):
    role: Role = Role.ASSISTANT
    content: List[Union[TextContent, ToolCallContent]]


class SpanInfo(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    session_id: str
    parent_span_id: Optional[str] = None


class BaseSpan(BaseModel):
    span_info: SpanInfo
    metadata: Optional[Dict] = {}


class InferenceSpan(BaseSpan):
    operation_name: str = "inference"
    messages: List[Union[UserMessage, AssistantMessage]]


class ToolExecutionSpan(BaseSpan):
    operation_name: str = "tool_execution"
    tool_call: ToolCall
    tool_result: ToolResult


class AgentInvocationSpan(BaseSpan):
    operation_name: str = "invoke_agent"
    user_prompt: str
    agent_response: str
    available_tools: List[ToolConfig]


class Trace(BaseModel):
    spans: List[Union[InferenceSpan, ToolExecutionSpan, AgentInvocationSpan]]
    trace_id: str
    session_id: str


class Session(BaseModel):
    traces: List[Trace]
    session_id: str
