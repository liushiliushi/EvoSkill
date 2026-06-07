"""LLM-backed skill tree evolution operators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.schemas import SkillTreeOperationResponse

from .maintainer import maintain_tree
from .models import SkillTree


@dataclass(frozen=True)
class LLMTreeMutation:
    """Result from an LLM-backed skill tree mutation."""

    tree: SkillTree
    reasoning: str
    operator: str
    cost_usd: float = 0.0


async def e2_crossover_with_llm(
    agent: Any,
    parent_a: SkillTree,
    parent_b: SkillTree,
    *,
    failures: str = "",
    child_name: str | None = None,
) -> LLMTreeMutation:
    """Use an LLM to fuse two parent trees into a meaningfully different child."""
    query = _build_e2_query(parent_a, parent_b, failures, child_name)
    return await _run_tree_operator(agent, query, fallback_tree=parent_a, expected_operator="e2")


async def m1_reflection_update_with_llm(
    agent: Any,
    tree: SkillTree,
    *,
    failures: str,
    child_name: str | None = None,
) -> LLMTreeMutation:
    """Use failure reflection to update the relevant path in one tree."""
    query = _build_m1_query(tree, failures, child_name)
    return await _run_tree_operator(agent, query, fallback_tree=tree, expected_operator="m1")


async def m2_random_semantic_mutation_with_llm(
    agent: Any,
    tree: SkillTree,
    *,
    focus: str = "",
    child_name: str | None = None,
) -> LLMTreeMutation:
    """Use an LLM to randomly generalize, specialize, or rewrite one branch."""
    query = _build_m2_query(tree, focus, child_name)
    return await _run_tree_operator(agent, query, fallback_tree=tree, expected_operator="m2")


async def maintain_with_llm(
    agent: Any,
    tree: SkillTree,
    *,
    max_depth: int = 4,
    max_nodes: int = 40,
    child_name: str | None = None,
) -> LLMTreeMutation:
    """Use an LLM to clean duplicate, conflicting, or overly broad branches."""
    query = _build_maintain_query(tree, max_depth, max_nodes, child_name)
    fallback = maintain_tree(tree, max_depth=max_depth, max_nodes=max_nodes, child_name=child_name)
    return await _run_tree_operator(agent, query, fallback_tree=fallback, expected_operator="maintain")


async def _run_tree_operator(
    agent: Any,
    query: str,
    *,
    fallback_tree: SkillTree,
    expected_operator: str,
) -> LLMTreeMutation:
    trace = await agent.run(query)
    output = getattr(trace, "output", None)
    if output is None:
        return LLMTreeMutation(
            tree=fallback_tree,
            reasoning=getattr(trace, "parse_error", None) or "LLM tree operator failed; used fallback tree.",
            operator=expected_operator,
            cost_usd=getattr(trace, "total_cost_usd", 0.0),
        )

    if not isinstance(output, SkillTreeOperationResponse):
        output = SkillTreeOperationResponse.model_validate(output)

    return LLMTreeMutation(
        tree=output.tree,
        reasoning=output.reasoning,
        operator=output.operator or expected_operator,
        cost_usd=getattr(trace, "total_cost_usd", 0.0),
    )


def _build_e2_query(
    parent_a: SkillTree,
    parent_b: SkillTree,
    failures: str,
    child_name: str | None,
) -> str:
    return f"""## Operator
e2: LLM-guided crossover.

Create a child skill tree that combines useful branches from both parents, but is not just a mechanical concatenation.

Child name: {child_name or parent_a.name + "-e2"}

## Parent A
{_tree_json(parent_a)}

## Parent B
{_tree_json(parent_b)}

## Recent Failures
{failures or "No failure context provided."}

Return the improved tree."""


def _build_m1_query(tree: SkillTree, failures: str, child_name: str | None) -> str:
    return f"""## Operator
m1: reflection update.

Use the failures to update only the relevant conditions/actions/reflections. Do not rewrite the whole tree unless necessary.

Child name: {child_name or tree.name + "-m1"}

## Current Tree
{_tree_json(tree)}

## Failures
{failures}

Return the updated tree."""


def _build_m2_query(tree: SkillTree, focus: str, child_name: str | None) -> str:
    return f"""## Operator
m2: random semantic mutation.

Pick one useful branch and generalize, specialize, or rewrite it to produce a meaningfully different candidate tree.

Child name: {child_name or tree.name + "-m2"}

## Current Tree
{_tree_json(tree)}

## Optional Focus
{focus or "No focus provided."}

Return the mutated tree."""


def _build_maintain_query(
    tree: SkillTree,
    max_depth: int,
    max_nodes: int,
    child_name: str | None,
) -> str:
    return f"""## Operator
maintain: tree cleanup.

Clean this tree so it is easy for a smaller model to route through.

Budgets:
- max_depth: {max_depth}
- max_nodes: {max_nodes}

Child name: {child_name or tree.name}

## Tree
{_tree_json(tree)}

Return the maintained tree."""


def _tree_json(tree: SkillTree) -> str:
    return json.dumps(tree.model_dump(), ensure_ascii=False, indent=2)
