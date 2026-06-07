from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_options, resolve_project_root
from src.schemas import SkillTreeOperationResponse
from src.agent_profiles.skill_tree_operator.prompt import (
    SKILL_TREE_OPERATOR_SYSTEM_PROMPT,
)


def get_project_root() -> str:
    """Backward-compatible project-root helper."""
    return str(resolve_project_root())


SKILL_TREE_OPERATOR_TOOLS = [
    "Read",
    "Glob",
    "Grep",
]


def get_skill_tree_operator_options(
    model: str | None = None,
    project_root: str | Path | None = None,
) -> Any:
    return build_options(
        system=SKILL_TREE_OPERATOR_SYSTEM_PROMPT.strip(),
        schema=SkillTreeOperationResponse.model_json_schema(),
        tools=SKILL_TREE_OPERATOR_TOOLS,
        project_root=project_root,
        model=model,
        setting_sources=["user", "project"],
        permission_mode="default",
    )


def make_skill_tree_operator_options(
    *,
    project_root: str | Path | None = None,
    model: str | None = None,
):
    return get_skill_tree_operator_options(model=model, project_root=project_root)


skill_tree_operator_options = get_skill_tree_operator_options()
