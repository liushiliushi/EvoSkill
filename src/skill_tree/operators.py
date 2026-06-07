"""Low-cost skill tree evolution operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .models import SkillNode, SkillTree


TreePath = tuple[int, ...]


def get_node_at_path(tree_or_node: SkillTree | SkillNode, path: Sequence[int]) -> SkillNode:
    """Return the node at a child-index path.

    The empty path points to the root node.
    """
    node = tree_or_node.root if isinstance(tree_or_node, SkillTree) else tree_or_node
    for index in path:
        try:
            node = node.children[index]
        except IndexError as exc:
            raise ValueError(f"invalid skill tree path: {tuple(path)}") from exc
    return node


def crossover(
    recipient_tree: SkillTree,
    donor_tree: SkillTree,
    *,
    recipient_path: TreePath = (),
    donor_path: TreePath = (),
    child_name: str | None = None,
) -> SkillTree:
    """Create an e1 child by appending a donor subtree under a recipient condition.

    This operator does not call an LLM. It is intentionally deterministic so it
    can be tested and used as a cheap mutation candidate generator.
    """
    child = recipient_tree.clone(name=child_name or f"{recipient_tree.name}-x")
    recipient_node = get_node_at_path(child, recipient_path)
    if recipient_node.type != "condition":
        raise ValueError("recipient path must point to a condition node")

    donor_node = get_node_at_path(donor_tree, donor_path).model_copy(deep=True)
    recipient_node.children.append(donor_node)
    _deduplicate_children(recipient_node)

    child.metadata = {
        **child.metadata,
        "operator": "e1_crossover",
        "parents": [recipient_tree.name, donor_tree.name],
        "recipient_path": list(recipient_path),
        "donor_path": list(donor_path),
    }
    return child


def prune_branches(
    tree: SkillTree,
    *,
    min_hits: int | None = None,
    min_contribution: float | None = None,
    child_name: str | None = None,
) -> SkillTree:
    """Create an m3 child by deleting weak branches using node metadata.

    A node is pruned only when the relevant metadata key is present. This avoids
    deleting fresh branches before they have been evaluated.
    """
    child = tree.clone(name=child_name)
    child.root.children = [
        pruned
        for child_node in child.root.children
        if (pruned := _prune_node(child_node, min_hits, min_contribution)) is not None
    ]
    child.metadata = {**child.metadata, "operator": "m3_prune"}
    return child


def _prune_node(
    node: SkillNode,
    min_hits: int | None,
    min_contribution: float | None,
) -> SkillNode | None:
    if _should_prune(node.metadata, min_hits, min_contribution):
        return None

    if node.children:
        node.children = [
            pruned
            for child_node in node.children
            if (pruned := _prune_node(child_node, min_hits, min_contribution)) is not None
        ]
        _deduplicate_children(node)
    return node


def _should_prune(
    metadata: dict[str, Any],
    min_hits: int | None,
    min_contribution: float | None,
) -> bool:
    if metadata.get("protected"):
        return False
    if min_hits is not None and "hits" in metadata and metadata["hits"] < min_hits:
        return True
    if (
        min_contribution is not None
        and "contribution" in metadata
        and metadata["contribution"] < min_contribution
    ):
        return True
    return False


def _deduplicate_children(node: SkillNode) -> None:
    seen: set[tuple[str, str]] = set()
    unique_children: list[SkillNode] = []
    for child in node.children:
        key = child.content_key()
        if key in seen:
            continue
        seen.add(key)
        unique_children.append(child)
    node.children = unique_children
