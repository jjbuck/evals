import json
from unittest.mock import Mock

import pytest
from strands_evals.converters import StrandsInMemoryConverter
from strands_evals.types.trace import (
    Session,
    InferenceSpan,
    AgentInvocationSpan,
    UserMessage,
    AssistantMessage,
)


CHAT_SPAN_JSON = """{
    "name": "chat",
    "context": {
        "trace_id": "0x0471cb8c57e50345f0dc1e88b8ec5889",
        "span_id": "0x866bf5ae7e025ead",
        "trace_state": "[]"
    },
    "kind": "SpanKind.CLIENT",
    "parent_id": "0x74433a7c5bd2b5ac",
    "attributes": {
        "gen_ai.operation.name": "chat"
    },
    "events": [
        {
            "name": "gen_ai.user.message",
            "attributes": {
                "content": "[{\\"text\\": \\"What is the capital of France?\\"}]"
            }
        },
        {
            "name": "gen_ai.choice",
            "attributes": {
                "finish_reason": "end_turn",
                "message": "[{\\"text\\": \\"The capital of France is Paris.\\"}]"
            }
        }
    ]
}"""

AGENT_SPAN_JSON = """{
    "name": "invoke_agent Strands Agents",
    "context": {
        "trace_id": "0x0471cb8c57e50345f0dc1e88b8ec5889",
        "span_id": "0x013ef17add67d90f",
        "trace_state": "[]"
    },
    "kind": "SpanKind.CLIENT",
    "parent_id": null,
    "attributes": {
        "gen_ai.operation.name": "invoke_agent"
    },
    "events": [
        {
            "name": "gen_ai.user.message",
            "attributes": {
                "content": "[{\\"text\\": \\"What is the capital of France?\\"}]"
            }
        },
        {
            "name": "gen_ai.choice",
            "attributes": {
                "message": "The capital of France is Paris.\\n",
                "finish_reason": "end_turn"
            }
        }
    ]
}"""

DIFFERENT_TRACE_SPAN_JSON = """{
    "name": "chat",
    "context": {
        "trace_id": "0xb3da612bebc19c44b982ee68320f3f57",
        "span_id": "0xb25ee9e8f0034787",
        "trace_state": "[]"
    },
    "kind": "SpanKind.CLIENT",
    "parent_id": "0xd1eba0f540504402",
    "attributes": {
        "gen_ai.operation.name": "chat"
    },
    "events": [
        {
            "name": "gen_ai.user.message",
            "attributes": {
                "content": "[{\\"text\\": \\"What color is the ocean?\\"}]"
            }
        },
        {
            "name": "gen_ai.choice",
            "attributes": {
                "finish_reason": "end_turn",
                "message": "[{\\"text\\": \\"Blue\\"}]"
            }
        }
    ]
}"""


def create_mock_span(json_str: str) -> Mock:
    span = Mock()
    span.to_json.return_value = json_str
    return span


def test_convert_chat_span():
    converter = StrandsInMemoryConverter()
    span = create_mock_span(CHAT_SPAN_JSON)
    
    session = converter.convert([span], session_id="test-session")

    assert isinstance(session, Session)
    assert session.session_id == "test-session"
    assert len(session.traces) == 1
    assert session.traces[0].trace_id == "0471cb8c57e50345f0dc1e88b8ec5889"
    assert len(session.traces[0].spans) == 1
    
    inference_span = session.traces[0].spans[0]
    assert isinstance(inference_span, InferenceSpan)
    assert len(inference_span.messages) == 2
    assert isinstance(inference_span.messages[0], UserMessage)
    assert inference_span.messages[0].content[0].text == "What is the capital of France?"
    assert isinstance(inference_span.messages[1], AssistantMessage)
    assert inference_span.messages[1].content[0].text == "The capital of France is Paris."


def test_convert_agent_invocation_span():
    converter = StrandsInMemoryConverter()
    span = create_mock_span(AGENT_SPAN_JSON)
    
    session = converter.convert([span], session_id="test-session")

    assert isinstance(session, Session)
    assert len(session.traces) == 1
    
    agent_span = session.traces[0].spans[0]
    assert isinstance(agent_span, AgentInvocationSpan)
    assert agent_span.user_prompt == "What is the capital of France?"
    assert agent_span.agent_response == "The capital of France is Paris.\n"
    assert agent_span.span_info.parent_span_id is None


def test_convert_multiple_spans_same_trace():
    converter = StrandsInMemoryConverter()
    chat_span = create_mock_span(CHAT_SPAN_JSON)
    agent_span = create_mock_span(AGENT_SPAN_JSON)
    
    session = converter.convert([chat_span, agent_span], session_id="test-session")

    assert len(session.traces) == 1
    assert len(session.traces[0].spans) == 2
    assert session.traces[0].trace_id == "0471cb8c57e50345f0dc1e88b8ec5889"


def test_convert_multiple_traces():
    converter = StrandsInMemoryConverter()
    span1 = create_mock_span(CHAT_SPAN_JSON)
    span2 = create_mock_span(DIFFERENT_TRACE_SPAN_JSON)
    
    session = converter.convert([span1, span2], session_id="test-session")

    assert len(session.traces) == 2
    trace_ids = {trace.trace_id for trace in session.traces}
    assert "0471cb8c57e50345f0dc1e88b8ec5889" in trace_ids
    assert "b3da612bebc19c44b982ee68320f3f57" in trace_ids


def test_convert_empty_spans():
    converter = StrandsInMemoryConverter()
    session = converter.convert([], session_id="test-session")

    assert isinstance(session, Session)
    assert session.session_id == "test-session"
    assert len(session.traces) == 0


def test_span_info_extraction():
    converter = StrandsInMemoryConverter()
    span = create_mock_span(CHAT_SPAN_JSON)
    
    session = converter.convert([span], session_id="test-session")

    span_info = session.traces[0].spans[0].span_info
    assert span_info.trace_id == "0471cb8c57e50345f0dc1e88b8ec5889"
    assert span_info.span_id == "866bf5ae7e025ead"
    assert span_info.session_id == "test-session"
    assert span_info.parent_span_id == "74433a7c5bd2b5ac"


def test_removes_0x_prefix_from_ids():
    converter = StrandsInMemoryConverter()
    span = create_mock_span(CHAT_SPAN_JSON)
    
    session = converter.convert([span], session_id="test")

    assert "0x" not in session.traces[0].trace_id
    assert "0x" not in session.traces[0].spans[0].span_info.span_id
    assert "0x" not in session.traces[0].spans[0].span_info.parent_span_id
