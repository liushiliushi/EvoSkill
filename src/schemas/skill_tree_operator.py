from pydantic import BaseModel, Field

from src.skill_tree.models import SkillTree


class SkillTreeOperationResponse(BaseModel):
    """Structured response from the skill tree operator agent."""

    tree: SkillTree
    reasoning: str
    operator: str = Field(
        description="Operator used to produce the tree, e.g. e2, m1, m2, maintain."
    )
