from strands import Agent
from strands_evals import Case, Dataset
from strands_evals.evaluators import HelpfulnessEvaluator

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import trace

from strands_evals.converters import StrandsInMemoryConverter

#======================================
# SETUP TELEMETRY
#======================================
# 1. Set up the tracer provider
provider = TracerProvider()
trace.set_tracer_provider(provider)

# 2. Set up the in-memory exporter
memory_exporter = InMemorySpanExporter()
span_processor = BatchSpanProcessor(memory_exporter)
provider.add_span_processor(span_processor)


#======================================
# SETUP AND RUN STRANDS EVAL
#======================================

# 1. Define a task function
def user_task_function(query: str) -> str:

    agent = Agent(callback_handler=None)
    agent_response = agent(query)
    print(agent_response)

    finished_spans = memory_exporter.get_finished_spans()

    print([span.to_json() for span in finished_spans])
    converter = StrandsInMemoryConverter()
    session = converter.convert(finished_spans, session_id='test-session')

    print(session)

    #traces = fetch_traces_from_langfuse()
    #converter = LangGraphLangFuseConverter()

    return {
        'output': str(agent_response),
        'trajectory': session
    }

# 2. Create test cases
test_cases = [
    Case[str, str](
        name="knowledge-1",
        input="What is the capital of France?",
        metadata={"category": "knowledge"}
    ),
    Case[str, str](
        name="knowledge-2",
        input="What color is the ocean?",
        metadata={"category": "knowledge"}
    )
]

# 3. Create an evaluator
evaluator = HelpfulnessEvaluator()

# 4. Create a dataset
dataset = Dataset[str, str](
    cases=test_cases,
    evaluator=evaluator
)

# 5. Run evaluations
report = dataset.run_evaluations(user_task_function)
report.run_display()


'''
agent = Agent(callback_handler=None)
agent_response = agent('What is the capital of France?')

finished_spans = memory_exporter.get_finished_spans()
traces = [span.to_json() for span in finished_spans]
print(traces)
converter = StrandsInMemoryConverter()
session = converter.convert(finished_spans, session_id='test-session')

Session.model_validate(session)
'''

#converter = StrandsCloudWatchConverter()
#converter = LangGraphCloudWatchConverter()
#converter = StrandsLangFuseConverter()
#converter = LangGraphLangFuseConverter()
#converter = .... # expand as we learn more about different frameworks and exporters
# If customer's trace does not fit one of these converters/mappers, customer will have to do conversion themselves
# Need to provide documentation on formats to help customers do this
#evaluator = HelpfulnessEvaluator()
#evaluator.evaluate(trace)