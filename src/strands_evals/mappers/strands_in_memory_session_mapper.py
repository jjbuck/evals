import json
import logging
from collections import defaultdict
from datetime import datetime

from typing_extensions import Any, Union

from ..types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolCallContent,
    ToolExecutionSpan,
    ToolResultContent,
    Trace,
    UserMessage,
)
from .session_mapper import SessionMapper

logger = logging.getLogger(__name__)


class StrandsInMemorySessionMapper(SessionMapper):
    """Maps OpenTelemetry in-memory spans to Session format for evaluation."""

    def map_to_session(self, otel_spans: list[Any], session_id: str) -> Session:
        """
        Map OTEL spans to Session format.

        Args:
            otel_spans: List of OpenTelemetry span objects
            session_id: Session identifier

        Returns:
            Session object ready for evaluation
        """
        span_dicts = [json.loads(span.to_json()) for span in otel_spans]

        traces_by_id = defaultdict(list)
        for span_dict in span_dicts:
            trace_id = span_dict["context"]["trace_id"]
            traces_by_id[trace_id].append(span_dict)

        traces: list[Trace] = []
        for trace_id, spans in traces_by_id.items():
            trace = self._convert_trace(trace_id, spans, session_id)
            if not trace.spans:
                logger.warning(f"Trace {trace_id} has no valid spans, skipping")
                continue
            traces.append(trace)

        return Session(traces=traces, session_id=session_id)

    def _convert_trace(self, trace_id: str, otel_spans: list[dict], session_id: str) -> Trace:
        converted_spans: list[Union[InferenceSpan, ToolExecutionSpan, AgentInvocationSpan]] = []

        for otel_span in otel_spans:
            operation_name = otel_span.get("attributes", {}).get("gen_ai.operation.name", "")

            if operation_name == "chat":
                inference_span = self._convert_inference_span(otel_span, session_id)
                if inference_span.messages:
                    converted_spans.append(inference_span)
            elif operation_name == "invoke_agent":
                agent_span = self._convert_agent_invocation_span(otel_span, session_id)
                converted_spans.append(agent_span)

        return Trace(spans=converted_spans, trace_id=trace_id.replace("0x", ""), session_id=session_id)

    def _parse_timestamp(self, timestamp: Any) -> datetime:
        if isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return datetime.fromtimestamp(int(timestamp) / 1e9)

    def _convert_inference_span(self, otel_span: dict, session_id: str) -> InferenceSpan:
        span_info = SpanInfo(
            trace_id=otel_span["context"]["trace_id"].replace("0x", ""),
            span_id=otel_span["context"]["span_id"].replace("0x", ""),
            session_id=session_id,
            parent_span_id=(otel_span.get("parent_id") or "").replace("0x", "") or None,
            start_time=self._parse_timestamp(otel_span["start_time"]),
            end_time=self._parse_timestamp(otel_span["end_time"]),
        )

        messages: list[Union[UserMessage, AssistantMessage]] = []
        events = otel_span.get("events", [])

        for event in events:
            if event["name"] == "gen_ai.user.message":
                content_str = event["attributes"].get("content", "[]")
                content_list = json.loads(content_str)

                user_content: list[Union[TextContent, ToolResultContent]] = [
                    TextContent(text=item["text"]) for item in content_list if "text" in item
                ]

                if user_content:
                    messages.append(UserMessage(content=user_content))

            elif event["name"] == "gen_ai.choice":
                message_str = event["attributes"].get("message", "[]")
                message_list = json.loads(message_str)

                assistant_content: list[Union[TextContent, ToolCallContent]] = [
                    TextContent(text=item["text"]) for item in message_list if "text" in item
                ]

                if assistant_content:
                    messages.append(AssistantMessage(content=assistant_content))

        return InferenceSpan(span_info=span_info, messages=messages, metadata={})

    def _convert_agent_invocation_span(self, otel_span: dict, session_id: str) -> AgentInvocationSpan:
        span_info = SpanInfo(
            trace_id=otel_span["context"]["trace_id"].replace("0x", ""),
            span_id=otel_span["context"]["span_id"].replace("0x", ""),
            session_id=session_id,
            parent_span_id=(otel_span.get("parent_id") or "").replace("0x", "") or None,
            start_time=self._parse_timestamp(otel_span["start_time"]),
            end_time=self._parse_timestamp(otel_span["end_time"]),
        )

        events = otel_span.get("events", [])
        user_prompt = ""
        agent_response = ""

        for event in events:
            if event["name"] == "gen_ai.user.message":
                content_str = event["attributes"].get("content", "[]")
                content_list = json.loads(content_str)
                user_prompt = content_list[0].get("text", "") if content_list else ""

            elif event["name"] == "gen_ai.choice":
                message_str = event["attributes"].get("message", "")
                if message_str:
                    try:
                        message_list = json.loads(message_str)
                        agent_response = message_list[0].get("text", "") if message_list else message_str
                    except json.JSONDecodeError:
                        agent_response = message_str

        return AgentInvocationSpan(
            span_info=span_info, user_prompt=user_prompt, agent_response=agent_response, available_tools=[], metadata={}
        )
