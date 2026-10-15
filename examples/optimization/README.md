# Optimization examples

Studies with `strands_evals.optimization`: a Strands agent factory whose arguments are declared `@tunable`, scored by
a Strands Evals `Experiment`. A study evaluates **candidates** (values for the tunable arguments); each candidate runs
every case `repetitions` times, and each run is a **trial**. The design is in
[docs/optimization/design.md](../../docs/optimization/design.md).

| File | What it shows |
|---|---|
| `poem_length.py` | The smallest study: one text variable (the system prompt), rewritten by contrastive reflection until the poems are 40 to 60 words long. |
| `document_reader.py` | The annotated study. The agent reads a chained document page by page; the study tunes its system prompt, its tools' descriptions, the choice between two conversation managers and each manager's settings, for answer quality and tokens together, within a quality floor and a token budget. Every argument is commented with its default and the alternatives. |
| `document_reader_workload.py` | The workload the document-reader studies share: the documents, the `open_document` and `read_page` tools, the cases, and the `Answer` evaluator. Not a study. |
| `document_reader_steps.py` | The document-reader study in six steps, each adding one idea: one number (the window's size); a categorical setting and tokens as a second objective; text alone (the system prompt); text and numbers as two blocks; a choice between conversation managers, searched in rounds; and the tools' descriptions with a quality floor and a token budget. |
| `tau2_airline.py` | One plain study on τ²-bench: Claude Haiku 4.5 as the airline's customer-service agent, tuning its instructions, its tools' descriptions and its thinking budget for τ²'s reward. |
| `tau2_airline_steps.py` | A ladder of τ²-bench studies, from the instructions alone to instructions, reasoning, write review and tool descriptions together, plus three with limits (a token budget, a latency limit, a reward floor); each checks its best candidates on held-out tasks. |
| `tau2_workload.py` | The τ²-bench workload: the agent as τ²'s agent (τ² executes its tool calls), τ²'s tasks as cases, the `Tau2Task` that runs one conversation per trial, and the `Reward` evaluator. Not a study. Needs `pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench"`. |
| `simulated_document_reader.py` | The steps with a scripted model and a scripted reflection: everything else (tools, Strands' agent loop and conversation managers, evaluator, study, record) is real. No network; a 15-candidate study takes seconds. For end-to-end checks and for developing the explorer. |

## Running

Install the optional dependencies (Optuna for TPE, Jinja2 for the reflection templates):

```bash
pip install "strands-agents-evals[optimization]"
```

From the repository root:

```bash
python examples/optimization/simulated_document_reader.py --out studies --step 4

BYPASS_TOOL_CONSENT=true python examples/optimization/poem_length.py
BYPASS_TOOL_CONSENT=true python examples/optimization/document_reader.py
python examples/optimization/document_reader_steps.py --step 1
BYPASS_TOOL_CONSENT=true python examples/optimization/document_reader_steps.py --step 5
```

Without `--step`, the simulator runs every step; `--iterations` sets how many iterations each study runs after its
baseline.

The real studies call Amazon Bedrock (Claude Haiku 4.5 for the agents; Claude Sonnet 4 for reflection, by default),
so they need AWS credentials and model access in your region.

**The reflection agent runs a shell.** Contrastive reflection (the default method for text variables) gives its model
a shell over the trial transcripts, and `BYPASS_TOOL_CONSENT=true` lets that shell run without asking. Run these
studies where model-written shell commands are acceptable, such as a devbox or a container. The simulator replaces
reflection with a scripted rewrite and runs no shell.

## The record

With `record=` (the document-reader studies write theirs under `studies/`, the simulator under `--out`), a study
writes `study.jsonl`, one event per line, as each candidate finishes, with each trial's transcript under
`transcripts/` (see `src/strands_evals/optimization/record.py`). The explorer app in `explorer/` reads these records.
A directory holds one study: give each run its own.

A version of the document-reader study with each document as a packaged task, the agent acting in the task's
container, comes with Strands Evals' Harbor integration.
