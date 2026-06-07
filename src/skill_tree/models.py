"""Pydantic models for structured skill trees.

A skill tree is a structured alternative to a free-form SKILL.md body. It keeps
conditions, actions, and reflections separate so mutation operators can edit
subtrees without rewriting the whole skill.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


NodeType = Literal["condition", "action", "reflection"]

_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillNode(BaseModel):
    """One node in a skill tree."""

    type: NodeType
    rule: str | None = None
    instruction: str | None = None
    note: str | None = None
    children: list["SkillNode"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_node_shape(self) -> "SkillNode":
        if self.type == "condition":
            if not _has_text(self.rule):
                raise ValueError("condition nodes require a non-empty rule")
            return self

        if self.type == "action":
            if not _has_text(self.instruction):
                raise ValueError("action nodes require a non-empty instruction")
            if self.children:
                raise ValueError("action nodes must be leaves")
            return self

        if self.type == "reflection":
            if not _has_text(self.note):
                raise ValueError("reflection nodes require a non-empty note")
            if self.children:
                raise ValueError("reflection nodes must be leaves")
            return self

        return self

    def iter_nodes(self) -> list["SkillNode"]:
        """Return all nodes in preorder."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.iter_nodes())
        return nodes

    def count_nodes(self) -> int:
        """Return the total number of nodes in this subtree."""
        return len(self.iter_nodes())

    def max_depth(self) -> int:
        """Return subtree depth, counting this node as depth 1."""
        if not self.children:
            return 1
        return 1 + max(child.max_depth() for child in self.children)

    def content_key(self) -> tuple[str, str]:
        """Return a stable key used for simple duplicate detection."""
        if self.type == "condition":
            return self.type, _normalize_text(self.rule or "")
        if self.type == "action":
            return self.type, _normalize_text(self.instruction or "")
        return self.type, _normalize_text(self.note or "")


class SkillTree(BaseModel):
    """A structured skill represented as a decision tree."""

    name: str
    description: str = ""
    root: SkillNode
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tree_shape(self) -> "SkillTree":
        if not _SKILL_NAME_RE.match(self.name):
            raise ValueError(
                "skill tree name must be lowercase kebab-case, e.g. numeric-table-answering"
            )
        if self.root.type != "condition":
            raise ValueError("skill tree root must be a condition node")
        return self

    def iter_nodes(self) -> list[SkillNode]:
        """Return all nodes in preorder."""
        return self.root.iter_nodes()

    def count_nodes(self) -> int:
        """Return the total number of nodes in the tree."""
        return self.root.count_nodes()

    def max_depth(self) -> int:
        """Return tree depth, counting root as depth 1."""
        return self.root.max_depth()

    def clone(self, *, name: str | None = None) -> "SkillTree":
        """Return a deep copy, optionally with a new name."""
        data = self.model_dump()
        if name is not None:
            data["name"] = name
        return SkillTree.model_validate(data)


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())
