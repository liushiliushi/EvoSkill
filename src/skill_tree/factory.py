"""Helpers for creating initial skill trees."""

from __future__ import annotations

import re

from .models import SkillNode, SkillTree


def tree_from_proposal(
    *,
    name: str,
    proposal: str,
    justification: str = "",
) -> SkillTree:
    """Create a small initial tree from a natural-language skill proposal."""
    root_rule = f"the task needs this capability: {_one_line(proposal)}"
    children = [
        SkillNode(
            type="action",
            instruction=_one_line(proposal),
            metadata={"source": "proposal"},
        )
    ]
    if justification.strip():
        children.append(
            SkillNode(
                type="reflection",
                note=_one_line(justification),
                metadata={"source": "justification"},
            )
        )
    return SkillTree(
        name=normalize_skill_tree_name(name),
        description=_one_line(proposal)[:180],
        root=SkillNode(type="condition", rule=root_rule, children=children),
        metadata={"source": "proposal"},
    )


def normalize_skill_tree_name(value: str) -> str:
    """Normalize arbitrary text to lowercase kebab-case."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    return normalized or "skill-tree"


def _one_line(value: str) -> str:
    return " ".join(value.strip().split())
