"""Fitness helpers for skill tree evolution."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .models import SkillTree
from .renderer import render_skill_markdown


@dataclass(frozen=True)
class FitnessBreakdown:
    """Detailed fitness calculation for a skill tree."""

    validation_score: float
    node_penalty: float
    depth_penalty: float
    token_penalty: float
    fitness: float


def compute_fitness(
    tree: SkillTree,
    validation_score: float,
    *,
    node_penalty_weight: float = 0.01,
    depth_penalty_weight: float = 0.02,
    token_penalty_weight: float = 0.0,
    free_depth: int = 4,
    rendered_tokens: int | None = None,
) -> FitnessBreakdown:
    """Compute fitness as validation score minus complexity penalties."""
    if rendered_tokens is None:
        rendered_tokens = len(render_skill_markdown(tree).split())

    node_penalty = node_penalty_weight * tree.count_nodes()
    depth_penalty = depth_penalty_weight * max(0, tree.max_depth() - free_depth)
    token_penalty = token_penalty_weight * (rendered_tokens / 1000.0)
    fitness = validation_score - node_penalty - depth_penalty - token_penalty

    return FitnessBreakdown(
        validation_score=validation_score,
        node_penalty=node_penalty,
        depth_penalty=depth_penalty,
        token_penalty=token_penalty,
        fitness=fitness,
    )


def category_balanced_score(scores: dict[str, list[float]]) -> float:
    """Average per-category scores, then average categories.

    This prevents large categories from dominating parent selection.
    """
    category_scores = [
        mean(category_values)
        for category_values in scores.values()
        if category_values
    ]
    if not category_scores:
        return 0.0
    return mean(category_scores)
