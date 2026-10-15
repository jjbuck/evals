# strands_evals.optimization: what the framework is, and the plan to get there

Status: plan of record. Ported on 2026-10-14 from harness-optimizer's `docs/design/framework.md` (plan of record
2026-10-13; §3.5, §3.6 and §4 updated 2026-10-14). The units are renamed, §3.7 gives the module layout in Strands
Evals, and §4 is checked against this module.

**Terms.** A study evaluates **candidates**: values z for the agent's tunable arguments. Each candidate runs every case
of the experiment `repetitions` times, and each run is a **trial**. (In harness-optimizer a candidate was a "trial",
and a trial an "attempt".)

## 1. The problem it solves

Fix the model. Choose the harness decision z = (F, a, η, φ) that maximizes the task quality of a Strands agent on a
workload, under constraints, or recover the Pareto set when no trade-off is declared (whitepaper §4.1). The four parts:

- **F, control flow:** the program that runs the components. This is later work.
- **a, component structure:** which components, which strategies, which branches. Examples:
  - `SlidingWindowConversationManager` vs `SummarizingConversationManager`;
  - a categorical option inside one of them.
- **η, settings:** numbers, categories and text (`window_size`, `should_truncate_results`, `system_prompt`, a tool's
  description, a skill library).
- **φ, learned weights:** only exist when a chosen strategy declares a trainable controller. This is later work.

The problem is **joint** in z but **hierarchical**: choosing a decides which η and φ exist. So the solver works in
**blocks**:
- each block gets a type-appropriate method;
- a **schedule** says when each block moves: joint, alternating, nested, and later a learned Oψ;
- every candidate is scored by the same evaluator.

## 2. Requirements (from the user, the whitepaper, the SPA notes, and the original harness-optimizer)

| # | Requirement | Source |
|---|---|---|
| R1 | An ordinary Strands agent factory with `@tunable` declarations. No Formula objects in the user's code. | SPA §07, user |
| R2 | Strands Evals is the evaluation signal: in-process `Experiment` + evaluators, and packaged benchmarks via Harbor. Not tied to one benchmark. | user |
| R3 | Formal contracts: grader score → trial metric → aggregate → objective / constraint. | user, whitepaper D.2 |
| R4 | Objectives and constraints are user-defined, and either can be tokens, cost or latency. | user |
| R5 | Several objectives, with a selectable combination: Pareto, weighted sum, Chebyshev, lexicographic, independent. | user |
| R6 | Hierarchical joint block optimization with schedules over the blocks. Methods by block type. Outer choices are compared at their inner best, with shared evidence. | whitepaper §4.2–4.5, user |
| R7 | Two levels of structure: a categorical option inside a component, and a choice between components that each have their own settings. | user |
| R8 | Reuse the existing core logic, such as harness-optimizer's reflection optimizers, rather than reinventing it. | user, blog |
| R9 | `Study(...).optimize()`. Python objects are the source of truth; YAML and JSON Schema come later. No one-way doors. | user |
| R10 | Compare against the baseline: absolute by default, difference and ratio paired on (case, repetition). | user |
| R11 | Honest measurement: repetitions, cases as the unit, intervals, a held-out check of the winner. A failed or invalid trial never silently scores zero. | whitepaper D.2/E, harness-lab results |
| R12 | A durable, resumable record of every candidate: values, proposals and their reasons, metrics, trials. An explorer UI and a CLI read it. | user |
| R13 | Runs on Harbor (Docker / EC2) with concurrency, retries of infrastructure failures, and budgets (candidates, trials, hours). Report the optimizer's own cost too. | user, whitepaper §4.1 SearchCost |
| R14 | The simplest high-quality developer experience first, grown one verified stage at a time. | user |
| R15 | Benchmarks that prove generality: a toy steel thread, ContextBench, and τ² banking. | user |

## 3. The design

### 3.1 Contracts (whitepaper §4.2)

```
target       @tunable factory            builds H(z) fresh for every trial
candidate    flat dotted values z        serializable (Values); stable key; measured as one Candidate
evaluator    run_trials(...)             -> trials: (case, repetition, status, metrics, trajectory, artifacts)
projection   metrics, objectives,        trial metric -> case mean -> compare to baseline -> objective/constraint;
             constraints                 several objectives combined as their wrapper says (Pareto, WeightedSum, ...)
search       blocks + methods + schedule which blocks change, holding the rest at an incumbent
study        candidates, record,         durable; resumable; held-out selection; provenance and cost
             budgets, selection
```

### 3.2 Declarations: what the user writes

```python
@tunable(
    system_prompt=TextValue(default="You answer support tickets concisely."),          # η, text
    conversation_manager=OneOf(                                                         # a, structure
        component(SlidingWindowConversationManager,
                  window_size=Int(4, 40, default=40),                                   # η, number
                  should_truncate_results=Choice([True, False], default=True)),         # η, category inside a component
        component(SummarizingConversationManager,
                  summary_ratio=Float(0.1, 0.8, default=0.3),
                  preserve_recent_messages=Int(4, 16, default=10)),
        labels=["sliding_window", "summarizing"]),
)
def create_agent(system_prompt, conversation_manager):
    return Agent(system_prompt=system_prompt, conversation_manager=conversation_manager)
```

Text is not only the system prompt: `tools=tool_descriptions(search, fetch)` makes each tool's description a
`TextValue` (the factory gets copies of the tools with the candidate's descriptions), and `skills=SkillLibrary(...)`
makes a library of Agent Skills a value (the factory gets the path of one version of the library).

Later: `Program(...)` for F, and `Weights(...)` for φ declared by a strategy.

### 3.3 Evaluation: one trial record, two adapters

Every adapter returns the same **trial**:
- `case` and `repetition`;
- `status`: `completed`, `agent_failure`, or invalid (`infrastructure_invalid`, `verifier_invalid` or `cancelled`);
- `metrics`: the grader's scores by name, plus usage (`input_tokens`, `output_tokens`, `cache_read_tokens`,
  `cache_write_tokens`, `tokens`, `seconds`, `model_calls`, `removed_messages`);
- the trajectory (messages or session) and artifacts.

The adapters:
- **`Experiment`** (in process): the factory builds a fresh agent for every trial, and the evaluators score it. An
  exception in the agent is an `agent_failure`, not a zero.
- **Packaged benchmark** (pending: Strands Evals has no Harbor integration yet): each candidate is one run of the
  benchmark with Harbor's executor, and each of Harbor's task results maps onto a trial. The factory may declare the
  resources a benchmark provides (`sandbox`, `tools`); one that does not declare them gets neither, so the same factory
  fits both adapters.

`invalid` trials are retried (throttling by Strands Evals' own backoff; a task's `TrialInvalid` and connection failures
`retries=` times). A failure policy declares how an `agent_failure` counts (`agent_error_score=`: the worst end of the
metric's range, a number, or excluded).

### 3.4 Projection

- **A metric** is a name in the trial's metrics, with a range so intervals can be clipped: an evaluator's score (0 to 1
  unless the evaluator declares `score_range`), the agent's usage (0 or more), or `Derived(name, fn, range=...)`,
  computed from each trial's other metrics and then estimated like any other.
- **Aggregate:** repetitions are averaged within a case, and the case is the unit. Report the mean and a 95% interval.
- **Objective:** `Maximize`/`Minimize(metric, compare="absolute"|"difference"|"ratio")`. Difference and ratio pair each
  case and repetition with the baseline, which is re-run as the comparison needs.
- **Constraint:** `AtMost`/`AtLeast(metric, limit, compare=...)`. Violation ≤ 0 means the constraint is met.
- **Several objectives** are given wrapped in how they combine, `objectives=Pareto(Maximize(...), Minimize(...))`; one
  objective needs no wrapper.
  - `Pareto(...)` keeps the front.
  - `WeightedSum(...)`, `Chebyshev(...)` and `Lexicographic(...)` scalarize for methods and selection.
  - `Independent(...)` gives one method instance per objective.

### 3.5 Optimization: the search (blocks, methods, schedules)

**Blocks.** A block is a method and the variables it proposes, given as the objects declared in `@tunable`:
`Block(TPE(), window_size, pin_first)`, `Block(Reflection(), system_prompt, tools)`. A value (`Int`, `TextValue`, ...)
and a choice (`OneOf`, `Switch`) are each one variable; a component (such as `tool_descriptions(...)`) stands for every
variable declared in it, unless another block names one of them itself. There are no names to match and no rules by
type: a block holds exactly the objects it is given. A variable no block holds goes to a default block for its kind:
`structure` (a), `numbers`, `text` and `skills` (η), and later `flow` (F) and `weights` (φ). A block that holds a
choice is *structural*; structural blocks are proposed first, outermost first. `Search(blocks=..., schedule=...,
redraws=...)` puts the blocks and the schedule together, and is the study's one argument about how to search.

**Methods.** A method proposes values for its block while the rest is held, and observes results, either its own or as
shared evidence. The methods:
- `Optuna(sampler)` and `TPE`: one Optuna study per outer structure.
- `Reflection`: harness-optimizer's `ContrastiveReflectionOptimizer`, ported unchanged; later `MultiAgentOptimizer`.
- `SkillCuration`: harness-optimizer's `SkillLibraryOptimizer`, ported unchanged, for skill libraries.
- Later: structural and program proposers, and SGD for φ.

Each method sees the projection, not raw metrics. Its scalar or vector feedback comes from the objectives, constraints
and how the objectives combine.

**Schedules.** A schedule refers to the `Block` objects themselves.
- `Joint()`: every block proposes in every iteration.
- `Alternate(numbers, texts, iterations=1)`: one block at a time; with `Independent`, an objective's turn is a whole
  cycle.
- `Nested(outer=choose_manager, inner=Joint(), inner_iterations=4)`: rounds. Each round proposes the outer block, then
  runs a bounded inner
  search with the outer values fixed; the outer method is told the round's best (FAIL when the round has no comparable
  candidate, e.g. the chosen structure cannot be built, in which case no inner search runs). A level whose outer block
  has no variables under the chosen structure is skipped.
- Schedules compose to any depth (F over a over η), with `Nested(inner=Nested(...))`. The default is rounds over the
  structural blocks, outermost first, around `Joint()`.
- A schedule runs in a **scope** (`Scope` in `study.py`): the blocks it may change, its candidate budget, and the
  candidates it holds values from. Schedules create candidates only through `scope.candidate(...)` and rounds only
  through `scope.child(...)`, so the budget is kept in one place, every result reaches the right methods (evidence to
  the scope's blocks, a deferred draw told once), and every candidate's place in the schedule is recorded. A custom
  schedule implements `run(scope)`, or just `blocks(k, in_scope)`, and gets the same guarantees.
- Schedules are checked when the study is built (`Schedule.check`): every block they name is one of the search's, and
  an outer block may not depend on a choice in its
  inner blocks.
- A variable the held candidate lacks (it exists only under a choice the held candidate did not make) is inherited
  from the best candidate that has it, else defaulted; both are recorded.
- **Values already measured are never measured again.** A step whose proposal repeats a measured candidate tells its
  methods that candidate's result and proposes again (`Search(redraws=3)` times); a step that still repeats has
  nothing new to measure, and its schedule moves on (Joint stops, Alternate skips the block, a round ends). A round
  whose opening proposal repeats a measured candidate opens on it without measuring it again or counting it against
  the budget, so a structure chosen a second time continues its inner search from its best so far; rounds stop once
  `redraws + 1` in a row measure nothing new. A run that ends this way finishes `exhausted`. Re-measuring a
  candidate to reduce its noise is a separate, deliberate decision (allocation), not a side effect of a repeat.
- Later, a learned Oψ behind the same interface.

**Incumbent and selection.** The held candidate is the best feasible one under the objectives' combination. A schedule may hold a
different candidate, such as the best within its inner search. Final selection re-measures the best on held-out cases.

### 3.6 Study: record, budgets, surfaces

- **Record:** an append-only event log under the study's directory (`record.py`; format `strands_evals.optimization`,
  schema 2): the study header (goals, how they combine, the schedule tree with node ids, blocks and their order, the space, every
  metric's role and range, the cases and repetitions); `run` per `optimize()`; `scope` and `scope_end` for every
  schedule run and round; one `candidate` event per candidate (values, trials, estimates, violations by constraint
  index, its scope path and role, what was proposed, drawn, inherited and defaulted, what each method copy was told,
  and the best candidates after it); `repeat` for every proposal that repeated a measured candidate; `told` for every
  deferred draw; `finished` (with its reason: `candidates`, `patience` or `exhausted`) or `stopped`. Transcripts are stored
  by content hash. It is the source for resuming a study, the explorer and the summaries: the explorer renders
  recorded facts and infers nothing. Building a `Study` writes nothing; the record opens on the first `optimize()`,
  and refuses a directory that already holds a study.
- **Budgets:** candidates, trials and hours. The optimizer's own model usage (reflection) is recorded as search cost.
- **Surfaces:** the Python API first. Serialization to YAML and JSON Schema follows from the declarations, which are
  plain data. The CLI (`run`, `resume`, `status`, `ui`, `export`) reads both. The explorer shows the history of every
  objective, constraint, token count and parameter by candidate, the trade-off view, and drill-down to each trial's
  transcript.

### 3.7 Module layout in Strands Evals

```
src/strands_evals/optimization/
  __init__.py      the public API: from strands_evals.optimization import Study, Maximize, tunable, ...
  space/           declarations: Float, Int, Choice, TextValue, SkillLibrary (domains.py); component, OneOf
                   (component.py); Switch (switch.py); @tunable, TunableFactory, Values (factory.py);
                   tool_descriptions (tools.py); skill-library checks and versions (skills.py); the one recursive
                   model of a space that they share: keys, defaults, building, validation, serialization (space.py)
  trials.py        one candidate's trials: run_trials, Trial, TrialStatus, the usage metrics
  projection.py    metrics (Derived too), estimates, objectives, constraints, how objectives combine
  methods.py       Block, Method, Draw; Optuna, TPE, Reflection, SkillCuration; resolving blocks to variables
  schedules.py     Schedule, Joint, Alternate, Nested; the default schedule
  search.py        Search: the blocks, the schedule and redraws
  study.py         Study, Candidate, Scope
  record.py        the event log
  effects.py       the effects model fitted to a record (effects.json)
  _reflection/     the reflection and curation optimizers ported from harness-optimizer, and their templates
tests/strands_evals/optimization/   unit tests (scripted models, no network)
examples/optimization/              the poem and document-reader studies, and a simulated document reader
explorer/                           the study explorer, which reads records
```

Optuna (TPE) and Jinja2 (the reflection templates) are the `optimization` extra; BayesBench, for the effects model, is
installed separately (see `pyproject.toml`). Reflection's shell tool comes from `strands-agents-tools`, already a
dependency.

## 4. Where the code stands (branch `feat/optimization`), and what to fix

The module was ported from harness-optimizer's branch `feat/study-minimal`, with the units renamed; its verdicts carry
over except where a row says otherwise.

| Area | Plan | Code now | Verdict |
|---|---|---|---|
| Declarations | §3.2 | `Float/Int/Choice/TextValue`, `component`, `OneOf`, `Switch`, two levels of structure; `tool_descriptions` and `SkillLibrary` | **Matches** |
| No Formula in user code | R1 | Only an internal adapter for the reused optimizers | **Matches** |
| Blocks by type, custom blocks | §3.5 | Default blocks by kind; `Block(method, *variables)` over the declared objects | **Matches** |
| Methods | §3.5 | `TPE`/`Optuna` (per-structure studies, shared evidence), `Reflection` (reuses `ContrastiveReflectionOptimizer`) and `SkillCuration` (reuses `SkillLibraryOptimizer`) | **Matches** |
| Policies | §3.5 | `Joint`, `Alternate`, `Nested` to any depth, on scopes; checked at construction; inactive levels skipped; failed outer proposals told FAIL | **Matches** |
| Trial status | §3.3, R11 | The study's task wrapper classifies every failure; agent failures, infrastructure and evaluator failures are statuses, never zeros; an out-of-range score is unavailable, not a crash | **Matches** |
| Repetitions, intervals | §3.4, R11 | `Study(repetitions=...)` (Strands Evals has no `Experiment.repetitions`: the study runs each case that many times itself); case-level means with clipped 95% t-intervals | **Matches** |
| Compare to baseline | §3.4, R10 | Absolute only (candidate 0 is the baseline) | **Missing**; the door is open |
| Combine modes | §3.4, R5 | `Pareto`, `Independent`, `WeightedSum`, `Chebyshev`, `Lexicographic`; held candidate measured against the baseline in each objective's unit | **Matches** |
| Feedback to methods | §3.5 | Methods are told the projection (the combination's `told`, with constraint violations); reflection gets per-trial rewards in [0, 1] | **Matches** |
| Usage metrics | §3.3 | tokens (input, output, cache read/write), seconds, model calls, removed messages; summarizer calls unmetered until the Strands fix | **Matches**, with the upstream gap |
| Packaged / Harbor benchmarks | §3.3, R2 | None in Strands Evals yet. harness-optimizer ran the document-reader tasks through Harbor (on a Harbor fork with identity, thread-limit and credential fixes); the factory already takes the `sandbox` and `tools` resources | **Pending** Strands Evals' Harbor integration; τ² MCP tools still missing |
| Record and resume | §3.6, R12 | The event log (format `strands_evals.optimization`, schema 1) with schedule provenance (scopes, told, best_after); resume not yet | **Partial:** record done, resume missing |
| Held-out selection | §3.5, R11 | None | **Missing** |
| Budgets and search cost | §3.6, R13 | Candidate budgets kept by scopes; each proposal's own model usage recorded in its origin | **Partial:** no trial or hour budgets |
| Concurrency | R13 | A candidate's trials run concurrently (`max_concurrency`); candidates run one at a time; the study runs its own event loop, so `optimize()` cannot be called inside a running one | **Partial** |
| CLI, YAML, explorer | §3.6 | The explorer (`explorer/`) renders the record; no CLI or YAML | **Partial:** CLI and YAML missing, by design for now |
| Examples | R15 | `poem_length.py` (the steel thread; in harness-optimizer, 0% → 100% on a real run); `document_reader.py` and its steps (`document_reader_steps.py`), whose pages are chained (each page gives the key to the next) so the agent must read them in order and the conversation managers act; `simulated_document_reader.py`, the same studies with a scripted model | **Matches** for the mechanics. The value of structure search on real workloads needs ContextBench (stage 4). |

The document reader's pages are chained because of an earlier finding: with pages the agent could name up front, it
read all 8 in **one parallel tool call** (2 cycles), so neither manager ever acted and every configuration cost the
same.

## 5. Plan, in verified stages

Each stage ends with tests and one real run. Nothing in a later stage starts before its predecessor is verified.

1. **Measurement core.**
   - Trial record with status and failure policy, so there are no silent zeros.
   - Full usage, including cache tokens and seconds.
   - Repetitions; case-level means with intervals.
   - Methods fed from the projection instead of raw metrics.
   - Composable `Nested`.
   - **Verify:** unit tests. A real run of the poem study with 2 repetitions. A two-level structure example on a
     workload where the managers act: the agent must read pages sequentially, which a tool that reveals the next
     page's key enforces.
2. **Record and resume.**
   - The event log under the study directory.
   - `optimize()` resumes from it.
   - Search cost recorded.
   - **Verify:** kill the run mid-candidate, resume, and check that no trial is duplicated.
3. **Projection completeness.**
   - Comparison against the baseline (difference, ratio).
   - Combine modes.
   - Held-out selection.
   - **Verify:** tests on a seeded synthetic evaluator with known optima.
4. **Packaged benchmarks.**
   - The Harbor adapter, on Strands Evals' Harbor integration.
   - The runtime-identity fix harness-optimizer carried in its Harbor fork, upstream.
   - **Verify:** ContextBench on the devbox. Two levels of conversation-manager structure, plus the system prompt,
     under `Nested`.
5. **τ² banking.**
   - Pass the task's MCP tools to the factory (the `tools` resource). This fills the gap in Harbor's Strands adapter.
   - **Verify:** two τ² tasks on the devbox, with the same study code shape as ContextBench.
6. **Surfaces.**
   - YAML and JSON Schema from the declarations.
   - CLI.
   - The explorer on the event log.
7. **Later blocks.** `flow` (F) with program proposers; `weights` (φ); a learned schedule.

**Effect estimates (added 2026-10-14; after the explorer rebuild).** TPE is built to find a good configuration, not to
measure effects: it keeps no posterior over outcomes, ignores measurement noise, shares no evidence across structures,
and Optuna's default of 10 random startup trials (candidates, here) makes short studies effectively random search.
BayesBench (`nonatofabio/bayesbench`) keeps a conjugate Gaussian posterior over main effects and two-way interactions
of categorical factors, with heteroscedastic noise. In order:

1. **Analysis.** Fit the effects model to a study's record: per option (at any depth) its effect on each goal, with
   credible intervals and the model's misspecification diagnostics, shown in the explorer (Overview, Search space).
   Read-only on existing records. **Verify:** calibration on the fixtures (known effects) and on the matrix-v2 runs;
   the diagnostics must pass before step 2. In the code: `effects.py`, which every run writes as `effects.json` (when
   BayesBench is installed).
2. **A method for categorical blocks.** `Block(EffectsModel(...), manager)`: the same posterior, choosing by
   Thompson sampling (an optimization acquisition, not BayesBench's information gain), told repetition-level variance;
   with a stopping rule (drop an option once P(worse than the incumbent) > 0.95; stop the study when the leader is
   known to a chosen precision). TPE stays for numbers, reflection for text. **Verify:** against TPE on the fixtures
   and a real document-reader rerun, at the same candidate budget.
3. **TPE's startup trials.** Make `n_startup_trials` a documented `TPE` setting with a default suited to short studies
   (not Optuna's 10), and record it in the header. Independent of 1 and 2. Done: `TPE(n_startup_trials=4)` by
   default, recorded in the header.

**Joint step (enhancement, decided 2026-10-14; not now).** The optimizer stays block-coordinate: each block's method
proposes its own variables from the same held candidate, and methods learn from the same results but not from each
other's proposals. The enhancement is one model-based step over all variables, text included: (1) a joint surrogate
over numbers, choices and text features (rewrite lineage, rules present, possibly embeddings) with interactions, the
black-box analog of the gradient and Hessian; (2) candidates generated per type (reflection writes several text
edits in one call; sampling proposes numbers and choices); (3) a joint choice among combinations by Thompson sampling
or expected improvement, falling back to the coordinate step when the surrogate's diagnostics flag. It subsumes the
"method for categorical blocks" item above.
