"""Storage helpers for skill tree sources and rendered skills."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import SkillTree
from .renderer import render_skill_markdown


TREE_DIR = ".evoskill/skill_trees"
SKILLS_DIR = ".claude/skills"


def save_skill_tree(project_root: str | Path, tree: SkillTree) -> Path:
    """Save a skill tree YAML source under .evoskill/skill_trees/."""
    root = Path(project_root)
    path = root / TREE_DIR / f"{tree.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(tree.model_dump(), f, sort_keys=False, allow_unicode=True)
    return path


def load_skill_tree(path: str | Path) -> SkillTree:
    """Load a skill tree from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return SkillTree.model_validate(data)


def load_project_skill_trees(project_root: str | Path) -> list[SkillTree]:
    """Load all project skill tree YAML files in deterministic order."""
    tree_dir = Path(project_root) / TREE_DIR
    if not tree_dir.exists():
        return []
    return [
        load_skill_tree(path)
        for path in sorted(tree_dir.glob("*.yaml"))
    ]


def render_project_skill(project_root: str | Path, tree: SkillTree) -> Path:
    """Render one tree into the existing .claude/skills/<name>/SKILL.md format."""
    root = Path(project_root)
    path = root / SKILLS_DIR / tree.name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_skill_markdown(tree))
    return path


def save_and_render_skill_tree(project_root: str | Path, tree: SkillTree) -> tuple[Path, Path]:
    """Save YAML source and rendered Markdown for a skill tree."""
    tree_path = save_skill_tree(project_root, tree)
    skill_path = render_project_skill(project_root, tree)
    return tree_path, skill_path


def skill_tree_complexity(project_root: str | Path) -> tuple[int, int]:
    """Return total node count and max depth for project skill trees."""
    trees = load_project_skill_trees(project_root)
    if not trees:
        return 0, 0
    return sum(tree.count_nodes() for tree in trees), max(tree.max_depth() for tree in trees)
