import pytest

from strands_evals.types.trace import (
    Session,
    Trace,
    InferenceSpan,
    ToolExecutionSpan,
    AgentInvocationSpan,
    SpanInfo,
    UserMessage,
    AssistantMessage,
    TextContent,
    ToolCallContent,
    ToolCall,
    ToolResult,
    ToolConfig,
    Role,
    ContentType,
    EvaluationLevel,
    TurnLevelInput,
    ToolLevelInput,
)


def test_span_info_creation():
    """Test SpanInfo model creation"""
    span_info = SpanInfo(
        trace_id="trace123",
        span_id="span456",
        session_id="session789",
        parent_span_id="parent123"
    )
    
    assert span_info.trace_id == "trace123"
    assert span_info.span_id == "span456"
    assert span_info.session_id == "session789"
    assert span_info.parent_span_id == "parent123"


def test_text_content_creation():
    """Test TextContent model creation"""
    content = TextContent(text="Hello world")
    
    assert content.text == "Hello world"
    assert content.content_type == ContentType.TEXT


def test_tool_call_content_creation():
    """Test ToolCallContent model creation"""
    tool_call = ToolCallContent(
        name="calculator",
        parameters={"expression": "2+2"},
        tool_call_id="call123"
    )
    
    assert tool_call.name == "calculator"
    assert tool_call.parameters == {"expression": "2+2"}
    assert tool_call.content_type == ContentType.TOOL_USE


def test_user_message_creation():
    """Test UserMessage model creation"""
    message = UserMessage(content=[TextContent(text="Hello")])
    
    assert message.role == Role.USER
    assert len(message.content) == 1
    assert message.content[0].text == "Hello"


def test_assistant_message_creation():
    """Test AssistantMessage model creation"""
    message = AssistantMessage(content=[TextContent(text="Hi there")])
    
    assert message.role == Role.ASSISTANT
    assert len(message.content) == 1


def test_inference_span_creation():
    """Test InferenceSpan model creation"""
    span_info = SpanInfo(session_id="test")
    messages = [
        UserMessage(content=[TextContent(text="Hello")]),
        AssistantMessage(content=[TextContent(text="Hi")])
    ]
    
    span = InferenceSpan(span_info=span_info, messages=messages)
    
    assert span.operation_name == "inference"
    assert len(span.messages) == 2


def test_tool_execution_span_creation():
    """Test ToolExecutionSpan model creation"""
    span_info = SpanInfo(session_id="test")
    tool_call = ToolCall(name="calculator", parameters={"expr": "2+2"})
    tool_result = ToolResult(content="4")
    
    span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=tool_call,
        tool_result=tool_result
    )
    
    assert span.operation_name == "tool_execution"
    assert span.tool_call.name == "calculator"
    assert span.tool_result.content == "4"


def test_agent_invocation_span_creation():
    """Test AgentInvocationSpan model creation"""
    span_info = SpanInfo(session_id="test")
    
    span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="What is 2+2?",
        agent_response="4",
        available_tools=[]
    )
    
    assert span.operation_name == "invoke_agent"
    assert span.user_prompt == "What is 2+2?"
    assert span.agent_response == "4"


def test_trace_creation():
    """Test Trace model creation"""
    span_info = SpanInfo(session_id="test")
    span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="test",
        agent_response="response",
        available_tools=[]
    )
    
    trace = Trace(spans=[span], trace_id="trace1", session_id="test")
    
    assert trace.trace_id == "trace1"
    assert trace.session_id == "test"
    assert len(trace.spans) == 1


def test_session_creation():
    """Test Session model creation"""
    span_info = SpanInfo(session_id="test")
    span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="test",
        agent_response="response",
        available_tools=[]
    )
    trace = Trace(spans=[span], trace_id="trace1", session_id="test")
    session = Session(traces=[trace], session_id="test")
    
    assert session.session_id == "test"
    assert len(session.traces) == 1


def test_turn_level_input_creation():
    """Test TurnLevelInput model creation"""
    turn_input = TurnLevelInput(
        user_prompt="What is 2+2?",
        agent_response="4",
        conversation_history=[{"user": "Hi", "assistant": "Hello"}]
    )
    
    assert turn_input.user_prompt == "What is 2+2?"
    assert turn_input.agent_response == "4"
    assert len(turn_input.conversation_history) == 1


def test_tool_level_input_creation():
    """Test ToolLevelInput model creation"""
    tool_input = ToolLevelInput(
        tool_name="calculator",
        parameters={"expression": "2+2"},
        result="4",
        user_query="Calculate 2+2"
    )
    
    assert tool_input.tool_name == "calculator"
    assert tool_input.parameters == {"expression": "2+2"}
    assert tool_input.result == "4"
    assert tool_input.user_query == "Calculate 2+2"



