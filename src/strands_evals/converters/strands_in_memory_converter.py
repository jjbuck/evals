"""
StrandsInMemoryConverter - Converts OpenTelemetry in-memory spans to Session format
"""
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..types.trace import (
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
    ToolResultContent,
    ToolCall,
    ToolResult,
    ToolConfig,
)


class StrandsInMemoryConverter:
    """Converts OpenTelemetry in-memory spans to Session format for evaluation."""
    
    def convert(self, otel_spans: List[Any], session_id: str) -> Session:
        """
        Convert OTEL spans to Session format.
        
        Args:
            otel_spans: List of OpenTelemetry span objects
            session_id: Session identifier
            
        Returns:
            Session object ready for evaluation
        """
        # Parse spans to dicts
        span_dicts = [json.loads(span.to_json()) for span in otel_spans]
        
        # Group spans by trace_id
        traces_by_id = defaultdict(list)
        for span_dict in span_dicts:
            trace_id = span_dict["context"]["trace_id"]
            traces_by_id[trace_id].append(span_dict)
        
        # Convert each trace
        traces = []
        for trace_id, spans in traces_by_id.items():
            trace = self._convert_trace(trace_id, spans, session_id)
            if trace:
                traces.append(trace)
        
        return Session(traces=traces, session_id=session_id)
    
    def _convert_trace(self, trace_id: str, otel_spans: List[Dict], session_id: str) -> Optional[Trace]:
        """Convert a group of OTEL spans with same trace_id to a Trace."""
        converted_spans = []
        
        for otel_span in otel_spans:
            operation_name = otel_span.get("attributes", {}).get("gen_ai.operation.name", "")
            
            if operation_name == "chat":
                span = self._convert_inference_span(otel_span, session_id)
                if span:
                    converted_spans.append(span)
            elif operation_name == "invoke_agent":
                span = self._convert_agent_invocation_span(otel_span, session_id)
                if span:
                    converted_spans.append(span)
        
        if not converted_spans:
            return None
            
        return Trace(
            spans=converted_spans,
            trace_id=trace_id.replace("0x", ""),
            session_id=session_id
        )
    
    def _convert_inference_span(self, otel_span: Dict, session_id: str) -> Optional[InferenceSpan]:
        """Convert OTEL chat span to InferenceSpan."""
        span_info = SpanInfo(
            trace_id=otel_span["context"]["trace_id"].replace("0x", ""),
            span_id=otel_span["context"]["span_id"].replace("0x", ""),
            session_id=session_id,
            parent_span_id=(otel_span.get("parent_id") or "").replace("0x", "") or None
        )
        
        messages = []
        events = otel_span.get("events", [])
        
        for event in events:
            if event["name"] == "gen_ai.user.message":
                content_str = event["attributes"].get("content", "[]")
                content_list = json.loads(content_str)
                
                user_content = [
                    TextContent(text=item["text"]) 
                    for item in content_list if "text" in item
                ]
                
                if user_content:
                    messages.append(UserMessage(content=user_content))
            
            elif event["name"] == "gen_ai.choice":
                message_str = event["attributes"].get("message", "[]")
                message_list = json.loads(message_str)
                
                assistant_content = [
                    TextContent(text=item["text"]) 
                    for item in message_list if "text" in item
                ]
                
                if assistant_content:
                    messages.append(AssistantMessage(content=assistant_content))
        
        if not messages:
            return None
            
        return InferenceSpan(
            span_info=span_info,
            messages=messages,
            metadata={}
        )
    
    def _convert_agent_invocation_span(self, otel_span: Dict, session_id: str) -> Optional[AgentInvocationSpan]:
        """Convert OTEL invoke_agent span to AgentInvocationSpan."""
        span_info = SpanInfo(
            trace_id=otel_span["context"]["trace_id"].replace("0x", ""),
            span_id=otel_span["context"]["span_id"].replace("0x", ""),
            session_id=session_id,
            parent_span_id=(otel_span.get("parent_id") or "").replace("0x", "") or None
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
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=[],
            metadata={}
        )
