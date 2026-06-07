"""Rule-based skill tree maintenance."""

from __future__ import annotations

from .models import SkillNode, SkillTree


def maintain_tree(
    tree: SkillTree,
    *,
    max_depth: int = 4,
    max_nodes: int = 40,
    child_name: str | None = None,
) -> SkillTree:
    """Clean a tree without calling an LLM.

    This removes duplicate sibling branches and trims the least protected tail
    branches when the tree exceeds a size budget.
    """
    maintained = tree.clone(name=child_name)
    _dedupe_subtree(maintained.root)
    _trim_depth(maintained.root, current_depth=1, max_depth=max_depth)

    while maintained.count_nodes() > max_nodes and _drop_last_unprotected_leaf(maintained.root):
        pass

    maintained.metadata = {
        **maintained.metadata,
        "maintained": True,
        "max_depth": max_depth,
        "max_nodes": max_nodes,
    }
    return maintained


def _dedupe_subtree(node: SkillNode) -> None:
    seen: set[tuple[str, str]] = set()
    unique_children: list[SkillNode] = []
    for child in node.children:
        key = child.content_key()
        if key in seen:
            continue
        seen.add(key)
        _dedupe_subtree(child)
        unique_children.append(child)
    node.children = unique_children


def _trim_depth(node: SkillNode, *, current_depth: int, max_depth: int) -> None:
    if current_depth >= max_depth:
        node.children = []
        return
    for child in node.children:
        _trim_depth(child, current_depth=current_depth + 1, max_depth=max_depth)


def _drop_last_unprotected_leaf(node: SkillNode) -> bool:
    for index in range(len(node.children) - 1, -1, -1):
        child = node.children[index]
        if child.metadata.get("protected"):
            continue
        if not child.children:
            node.children.pop(index)
            return True
        if _drop_last_unprotected_leaf(child):
            return True
    return False
