"""How a study searches: its blocks (each a method and the variables it proposes), the schedule that says when each
block moves, and how often a proposal that repeats a measured candidate is made again.

    choose_manager = Block(TPE(), manager)
    tune_manager = Block(TPE(), window_size, pin_first, summary_ratio, preserve_recent)
    rewrite_texts = Block(Reflection(), system_prompt, tools)

    Search(
        blocks=[choose_manager, tune_manager, rewrite_texts],
        schedule=Nested(outer=choose_manager, inner=Joint(), inner_iterations=4),
    )

Without blocks, every variable is in a default block for its type; without a schedule, the study builds rounds over
the structural blocks around a joint search of the rest (``default_schedule``).
"""

from __future__ import annotations

from typing import Sequence

from .methods import Block
from .schedules import Schedule


class Search:
    def __init__(self, blocks: Sequence[Block] = (), schedule: Schedule | None = None, redraws: int = 3) -> None:
        """``redraws``: how many more times a step proposes when its proposal repeats a measured candidate (each time
        its methods are told that candidate's result); a step that still repeats has nothing new to measure, and its
        schedule moves on."""
        if redraws < 0:
            raise ValueError("redraws must be at least 0")
        wrong = [b for b in blocks if not isinstance(b, Block)]
        if wrong:
            raise TypeError(f"Search(blocks=...) takes Block objects, not {wrong}")
        if len({id(b) for b in blocks}) != len(blocks):
            raise ValueError("the same Block is given twice")
        self.blocks, self.schedule, self.redraws = list(blocks), schedule, redraws
