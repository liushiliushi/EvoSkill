"""Render skill trees into the SKILL.md format used by current harnesses."""

from __future__ import annotations

from .models import SkillNode, SkillTree


def render_skill_markdown(
    tree: SkillTree,
    *,
    compatibility: str | None = "opencode",
) -> str:
    """Render a skill tree as a repo-local SKILL.md file.

    This keeps the current EvoSkill skill discovery path unchanged: agents still
    read Markdown skill files, while the source representation can be a tree.
    """
    description = tree.description or f"Structured skill tree: {tree.name}"
    frontmatter = [
        "---",
        f"name: {tree.name}",
        f"description: {description}",
    ]
    if compatibility:
        frontmatter.append(f"compatibility: {compatibility}")
    frontmatter.append("---")

    body = [
        "",
        "Use this skill as a routing tree. Follow only the branches that match the current task.",
        "",
        "## Routing Tree",
        *_render_node(tree.root, depth=0),
        "",
    ]
    return "\n".join(frontmatter + body)


def _render_node(node: SkillNode, *, depth: int) -> list[str]:
    indent = "  " * depth

    if node.type == "condition":
        lines = [f"{indent}- If: {node.rule}"]
        for child in node.children:
            lines.extend(_render_node(child, depth=depth + 1))
        return lines

    if node.type == "action":
        return [f"{indent}- Do: {node.instruction}"]

    return [f"{indent}- Remember: {node.note}"]
