"""Language-valued variables: tool descriptions as text, skill libraries, and reflection over several of them."""

import os
from pathlib import Path

import pytest
from strands import Agent, tool

from strands_evals.optimization import (
    Block,
    Maximize,
    Reflection,
    Search,
    SkillLibrary,
    Study,
    TextValue,
    Values,
    tool_descriptions,
    tunable,
)
from strands_evals.optimization.methods import SkillCuration
from strands_evals.optimization.space.skills import check_library, store

from .helpers import Says, experiment


@tool
def search(query: str) -> str:
    """Search the catalog."""
    return query


@tool
def click(target: str) -> str:
    """Click an element."""
    return target


# ---- tool descriptions -----------------------------------------------------------------------------------------------
def test_each_tool_description_is_a_text_variable_and_the_tools_are_never_changed():
    @tunable(tools=tool_descriptions(search, click))
    def build(tools):
        return tools

    params = {p.key: p for p in build.params()}
    assert set(params) == {"tools.search", "tools.click"}
    assert (params["tools.search"].type, params["tools.search"].default, params["tools.search"].max_chars) == (
        "text",
        "Search the catalog.",
        800,
    )
    values = {k: p.default for k, p in params.items()} | {"tools.search": "Search the catalog by keyword."}
    built = build(**build.bind(Values(values)))
    assert [t.tool_spec["description"] for t in built] == ["Search the catalog by keyword.", "Click an element."]
    assert search.tool_spec["description"] == "Search the catalog."
    with pytest.raises(ValueError, match="over max_chars=800"):
        build.validate(Values(values | {"tools.click": "x" * 801}))


# ---- reflection over several texts ----------------------------------------------------------------------------------
class Scripted:
    """Stands in for the reflection optimizer: records what it was given, then submits the edits it was told to."""

    seen: list = []

    def __init__(self, values, variables, edits):
        self.values, self.variables, self.edits = values, variables, edits
        self.last_metrics, self.last_wall_clock_s = {"input_tokens": 100, "output_tokens": 10}, 1.0

    def add_rollouts(self, rollouts):
        pass

    def add_rewards(self, rewards):
        pass

    def step(self):
        Scripted.seen.append(self.variables)
        self.values.update_params(self.edits)


class Edits(Reflection):
    def __init__(self, edits):
        super().__init__()
        self.edits = edits

    def optimizer(self, values, variables):
        return Scripted(values, variables, self.edits)


HELP = TextValue(default="Help.", about="the system prompt")
TOOLS = tool_descriptions(search, click)


@tunable(system_prompt=HELP, tools=TOOLS)
def create(system_prompt, tools):
    return Agent(model=Says("ok"), system_prompt=system_prompt, tools=tools, callback_handler=None)


def test_reflection_is_told_what_each_text_is_and_changes_only_what_it_submits():
    edits = {"tools.search": "Search the catalog by keyword; returns ids.", "tools.click": "y" * 900}
    study = Study(
        agent=create,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(blocks=[Block(Edits(edits), HELP, TOOLS)]),
        log=None,
    )
    study.optimize(iterations=1)
    candidate = study.candidates[1]
    told = {v["name"]: v for v in Scripted.seen[-1]}
    assert told["system_prompt"]["about"] == "the system prompt" and told["system_prompt"]["max_chars"] is None
    assert told["tools.search"]["max_chars"] == 800 and "search tool" in told["tools.search"]["about"]
    assert candidate.params["tools.search"] == edits["tools.search"]
    assert candidate.params["tools.click"] == "Click an element."  # over its limit: dropped, the held text kept
    assert candidate.params["system_prompt"] == "Help."  # not submitted: unchanged
    origin = candidate.origin["text"]
    assert origin["changed"] == ["tools.search"] and "over max_chars=800" in origin["dropped"]["tools.click"]


# ---- skill libraries ------------------------------------------------------------------------------------------------
def skill(root: Path, name: str, description: str | None = "Use when testing.") -> None:
    (root / name).mkdir(parents=True)
    front = f"---\nname: {name}\ndescription: {description}\n---\n" if description else ""
    (root / name / "SKILL.md").write_text(front + f"# {name}\n")


def test_a_skill_library_is_checked_and_stored_once_by_content(tmp_path):
    lib = tmp_path / "lib"
    skill(lib, "web-search")
    assert check_library(str(lib)) is None
    skill(lib, "broken", description=None)
    assert "broken" in check_library(str(lib))
    a = store({"web-search": str(lib / "web-search")}, tmp_path / "store")
    b = store({"web-search": str(lib / "web-search")}, tmp_path / "store")
    assert a == b and os.path.isfile(Path(a) / "web-search" / "SKILL.md")


class Curator:
    """Stands in for the skill-library optimizer: writes a decision tree, then applies it as the real one does."""

    def __init__(self, formula, output_folder):
        self.formula, self.output_folder = formula, output_folder
        self.last_metrics, self.last_wall_clock_s, self._last_decisions = {}, 1.0, {}

    def add_rollouts(self, rollouts):
        pass

    def add_rewards(self, rewards):
        pass

    def step(self):
        from strands_evals.optimization._reflection import collect_decisions

        skill(Path(self.output_folder) / "create", "checkout", "Use when buying.")
        self._last_decisions = collect_decisions(self.output_folder)
        self.formula.update_params({"decisions_dir": self.output_folder})


class Curate(SkillCuration):
    def optimizer(self, formula, output_folder):
        return Curator(formula, output_folder)


def test_skill_curation_proposes_a_new_library_version(tmp_path):
    lib = tmp_path / "lib"
    skill(lib, "web-search")

    library = SkillLibrary(default=str(lib))

    @tunable(skills=library)
    def create_with_skills(skills):
        return Agent(model=Says(" ".join(sorted(os.listdir(skills)))), callback_handler=None)

    study = Study(
        agent=create_with_skills,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(blocks=[Block(Curate(), library)]),
        record=tmp_path / "record",
        log=None,
    )
    assert study.blocks["skills"].keys == ("skills",)
    study.optimize(iterations=1)
    new = study.candidates[1].params["skills"]
    assert sorted(os.listdir(new)) == ["checkout", "web-search"] and new.startswith(str(tmp_path / "record"))
    assert study.candidates[1].origin["skills"]["skills"]["decisions"]["create"] == ["checkout"]
    assert study.candidates[1].metrics["WordCount"] == 2  # the agent was built with the new library
