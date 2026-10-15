"""Schedules: which blocks change in each iteration, and which candidate the other blocks are held at.

- ``Joint()``: every block in scope proposes in every iteration.
- ``Alternate(numbers, texts, iterations=1)``: one block at a time, in turn (block-coordinate updates), each for
  ``iterations`` iterations.
- ``Nested(outer=choose_manager, inner=Joint(), inner_iterations=4)``: rounds. Each round proposes the outer block,
  then runs ``inner`` for ``inner_iterations`` more with the outer values fixed; the outer method is told the round's
  best result, so outer choices are compared at their inner best. ``inner`` may itself be ``Nested``, to any depth. A
  round whose outer proposal repeats a measured candidate opens on it; rounds stop once ``redraws + 1`` of them in a
  row measure nothing new.

Blocks are given as the ``Block`` objects of the study's ``Search``. A schedule runs in a ``Scope`` (``study.py``): the
blocks it may change, the iterations it may still run, and the candidates it holds values from.
``scope.candidate(blocks)`` proposes and measures one candidate; ``scope.child(...)`` opens a round. The study records
every scope and every candidate's place in them, so the schedule at any depth is in the record. A custom schedule
implements ``run(scope)`` (or just ``blocks(k, in_scope)``) and gets the same record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .methods import Block
    from .study import Scope, Study


class Schedule:
    def blocks(self, k: int, in_scope: Sequence[str]) -> list[str]:
        """The blocks (by name) to update in the k-th iteration of a run, given the blocks in scope."""
        raise NotImplementedError

    def run(self, scope: "Scope") -> None:
        """Run iterations in ``scope`` until its budget is spent (or nothing in it can change)."""
        k = 0
        while scope.budget > 0:
            if scope.candidate(self.blocks(k, scope.blocks), step=k) is None:
                return
            k += 1

    def check(self, blocks: Sequence[str], study: "Study") -> None:
        """Raise if this schedule cannot run over ``blocks`` (called before anything runs)."""

    def describe(self, study: "Study") -> dict[str, Any]:
        """The schedule's kind and settings, for the record (its sub-schedules come from ``children``)."""
        return {"kind": type(self).__name__}

    def children(self) -> dict[str, "Schedule"]:
        """The schedules this one runs inside its rounds, by name."""
        return {}


class Joint(Schedule):
    def blocks(self, k: int, in_scope: Sequence[str]) -> list[str]:
        return list(in_scope)


class Alternate(Schedule):
    def __init__(self, *blocks: "Block", iterations: int = 1) -> None:
        """``blocks``: the blocks in turn (default: every block in scope); ``iterations``: how many each turn runs."""
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.order, self.iterations = list(blocks), iterations

    def describe(self, study: "Study") -> dict[str, Any]:
        return {"kind": "Alternate", "blocks": [study.name_of(b) for b in self.order], "iterations": self.iterations}

    def check(self, blocks: Sequence[str], study: "Study") -> None:
        outside = [study.name_of(b) for b in self.order if study.name_of(b) not in blocks]
        if outside:
            raise ValueError(f"Alternate names blocks outside its scope: {outside} (the scope is {list(blocks)})")

    def run(self, scope: "Scope") -> None:
        """One block per iteration, in turn; a block with nothing to propose under the current structure is skipped.
        With ``Independent``, an objective's turn is one whole cycle of the blocks."""
        order = [scope.study.name_of(b) for b in self.order] or list(scope.blocks)
        cycle = len(order) * self.iterations
        k, misses, objective = 0, 0, None
        while scope.budget > 0 and misses < cycle:
            if k % cycle == 0:
                objective = scope.turn()
            block = order[(k // self.iterations) % len(order)]
            made = scope.candidate([block], step=k, objective=objective) if block in scope.blocks else None
            misses = 0 if made is not None else misses + 1
            k += 1


class Nested(Schedule):
    def __init__(self, outer: "Block", inner: Schedule | None = None, inner_iterations: int = 4) -> None:
        """``outer``: the block each round proposes first; ``inner``: what runs under it (default ``Joint()``), for
        ``inner_iterations`` iterations per round."""
        if inner_iterations < 0:
            raise ValueError("inner_iterations must be at least 0")
        self.outer, self.inner, self.inner_iterations = outer, inner or Joint(), inner_iterations

    def describe(self, study: "Study") -> dict[str, Any]:
        return {"kind": "Nested", "outer": study.name_of(self.outer), "inner_iterations": self.inner_iterations}

    def children(self) -> dict[str, Schedule]:
        return {"inner": self.inner}

    def check(self, blocks: Sequence[str], study: "Study") -> None:
        outer = study.name_of(self.outer)
        if outer not in blocks:
            raise ValueError(f"no block {outer!r} to search over; the blocks in scope are {list(blocks)}")
        inner = [b for b in blocks if b != outer]
        for key in study.blocks[outer].keys:
            for selector in study.ancestors(key):
                if study.block_of[selector] in inner:
                    raise ValueError(
                        f"Nested(outer={outer!r}): {key} exists only for some values of {selector}, which is in "
                        f"the inner block {study.block_of[selector]!r}; search {study.block_of[selector]!r} in an "
                        f"enclosing round instead"
                    )
        self.inner.check(inner, study)

    def run(self, scope: "Scope") -> None:
        outer = scope.study.name_of(self.outer)
        if outer not in scope.blocks:  # nothing of the outer block exists under this structure: no rounds here
            self.inner.run(scope)
            return
        inner = [b for b in scope.blocks if b != outer]
        r, empty = 0, 0
        while scope.budget > 0 and empty <= scope.study.search.redraws:
            measured = len(scope.study.candidates)
            objective = scope.turn()
            with scope.child(
                "inner",
                self.inner,
                blocks=inner,
                limit=1 + self.inner_iterations,
                round=r,
                outer=outer,
                objective=objective,
            ) as round_:
                opener = round_.candidate([outer], role="outer", defer=[outer], base=scope.incumbent(objective))
                if opener is None:
                    return
                round_.narrow(opener)  # the inner blocks that exist under the structure just chosen
                if opener.eligible:
                    self.inner.run(round_)
                round_.tell(outer, round_.best())  # no eligible candidate in the round: the outer draw failed
            # A round that opened on a measured candidate and measured nothing new under it: rounds that keep doing
            # so have nothing left to try.
            empty = empty + 1 if len(scope.study.candidates) == measured else 0
            r += 1


def default_schedule(study: "Study") -> Schedule:
    """Rounds over the structural blocks, outermost first (a block whose choices exist only under another's comes
    after it), around a joint search of everything else; ``Joint()`` when there is nothing to nest."""
    structural = study.structural_order()
    others = [b for b in study.blocks if b not in structural]
    levels = structural if others else structural[:-1]
    schedule: Schedule = Joint()
    for name in reversed(levels):
        schedule = Nested(outer=study.blocks[name], inner=schedule)
    return schedule
