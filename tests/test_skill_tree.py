"""Tests for structured skill tree models and low-cost evolution operators."""

import asyncio

import pytest
import yaml
from pydantic import ValidationError

from src.skill_tree import (
    SkillNode,
    SkillTree,
    category_balanced_score,
    compute_fitness,
    crossover,
    get_node_at_path,
    load_project_skill_trees,
    load_skill_tree,
    maintain_tree,
    prune_branches,
    render_skill_markdown,
    save_and_render_skill_tree,
    save_skill_tree,
    skill_tree_complexity,
    tree_from_proposal,
)
from src.skill_tree.llm_ops import (
    e2_crossover_with_llm,
    m1_reflection_update_with_llm,
    m2_random_semantic_mutation_with_llm,
    maintain_with_llm,
)


def _sample_tree() -> SkillTree:
    return SkillTree(
        name="numeric-table-answering",
        description="Answer numeric table questions carefully.",
        root=SkillNode(
            type="condition",
            rule="task involves table numbers",
            children=[
                SkillNode(
                    type="action",
                    instruction="Check row labels, column labels, and units before calculating.",
                ),
                SkillNode(
                    type="condition",
                    rule="answer requires a percentage",
                    children=[
                        SkillNode(
                            type="action",
                            instruction="Convert the final ratio to a percentage and include the percent sign.",
                        ),
                    ],
                ),
            ],
        ),
    )


class TestSkillTreeModels:
    def test_yaml_construction(self):
        data = yaml.safe_load(
            """
name: numeric-table-answering
description: Answer numeric table questions carefully.
root:
  type: condition
  rule: task involves table numbers
  children:
    - type: action
      instruction: Check row labels and units before calculating.
"""
        )
        tree = SkillTree.model_validate(data)
        assert tree.name == "numeric-table-answering"
        assert tree.count_nodes() == 2
        assert tree.max_depth() == 2

    def test_condition_requires_rule(self):
        with pytest.raises(ValidationError, match="condition nodes require"):
            SkillNode(type="condition")

    def test_action_requires_instruction(self):
        with pytest.raises(ValidationError, match="action nodes require"):
            SkillNode(type="action")

    def test_action_must_be_leaf(self):
        with pytest.raises(ValidationError, match="action nodes must be leaves"):
            SkillNode(
                type="action",
                instruction="Do something.",
                children=[SkillNode(type="action", instruction="Nested action.")],
            )

    def test_name_must_be_kebab_case(self):
        with pytest.raises(ValidationError, match="lowercase kebab-case"):
            SkillTree(
                name="Bad Name",
                root=SkillNode(type="condition", rule="task matches"),
            )

    def test_root_must_be_condition(self):
        with pytest.raises(ValidationError, match="root must be a condition"):
            SkillTree(
                name="bad-root",
                root=SkillNode(type="action", instruction="Do something."),
            )


class TestSkillTreeRenderer:
    def test_render_skill_markdown_uses_existing_skill_format(self):
        rendered = render_skill_markdown(_sample_tree())
        assert rendered.startswith("---\nname: numeric-table-answering")
        assert "compatibility: opencode" in rendered
        assert "## Routing Tree" in rendered
        assert "- If: task involves table numbers" in rendered
        assert "- Do: Check row labels" in rendered

    def test_render_can_omit_compatibility(self):
        rendered = render_skill_markdown(_sample_tree(), compatibility=None)
        assert "compatibility:" not in rendered


class TestSkillTreeFitness:
    def test_compute_fitness_penalizes_nodes(self):
        small = SkillTree(
            name="small-tree",
            root=SkillNode(
                type="condition",
                rule="task matches",
                children=[SkillNode(type="action", instruction="Do the important step.")],
            ),
        )
        large = _sample_tree()

        small_fit = compute_fitness(small, validation_score=0.8)
        large_fit = compute_fitness(large, validation_score=0.8)

        assert small_fit.node_penalty < large_fit.node_penalty
        assert small_fit.fitness > large_fit.fitness

    def test_category_balanced_score_averages_categories_first(self):
        score = category_balanced_score(
            {
                "table": [1.0, 1.0, 0.0],
                "format": [0.0],
            }
        )
        assert score == pytest.approx(((2 / 3) + 0.0) / 2)

    def test_empty_category_scores_return_zero(self):
        assert category_balanced_score({}) == 0.0
        assert category_balanced_score({"empty": []}) == 0.0


class TestSkillTreeOperators:
    def test_get_node_at_path(self):
        tree = _sample_tree()
        node = get_node_at_path(tree, (1, 0))
        assert node.type == "action"
        assert "percentage" in node.instruction

    def test_crossover_appends_donor_subtree(self):
        recipient = _sample_tree()
        donor = SkillTree(
            name="format-answering",
            root=SkillNode(
                type="condition",
                rule="task has strict final-answer format",
                children=[SkillNode(type="action", instruction="Return only the final answer.")],
            ),
        )

        child = crossover(
            recipient,
            donor,
            recipient_path=(),
            donor_path=(),
            child_name="numeric-format-answering",
        )

        assert child.name == "numeric-format-answering"
        assert child.metadata["operator"] == "e1_crossover"
        assert child.count_nodes() == recipient.count_nodes() + donor.count_nodes()
        assert "format-answering" in child.metadata["parents"]

    def test_crossover_requires_condition_recipient(self):
        with pytest.raises(ValueError, match="condition node"):
            crossover(
                _sample_tree(),
                _sample_tree(),
                recipient_path=(0,),
                donor_path=(),
                child_name="bad-crossover",
            )

    def test_crossover_deduplicates_identical_sibling(self):
        tree = _sample_tree()
        child = crossover(
            tree,
            tree,
            recipient_path=(),
            donor_path=(0,),
            child_name="deduped-tree",
        )
        assert child.count_nodes() == tree.count_nodes()

    def test_prune_branches_removes_low_hit_branch(self):
        tree = SkillTree(
            name="prune-example",
            root=SkillNode(
                type="condition",
                rule="task matches",
                children=[
                    SkillNode(
                        type="condition",
                        rule="useful branch",
                        metadata={"hits": 5},
                        children=[SkillNode(type="action", instruction="Keep this.")],
                    ),
                    SkillNode(
                        type="condition",
                        rule="unused branch",
                        metadata={"hits": 0},
                        children=[SkillNode(type="action", instruction="Drop this.")],
                    ),
                ],
            ),
        )

        pruned = prune_branches(tree, min_hits=1, child_name="pruned-example")

        assert pruned.name == "pruned-example"
        assert pruned.metadata["operator"] == "m3_prune"
        assert len(pruned.root.children) == 1
        assert pruned.root.children[0].rule == "useful branch"

    def test_prune_ignores_nodes_without_metadata(self):
        tree = _sample_tree()
        pruned = prune_branches(tree, min_hits=1)
        assert pruned.count_nodes() == tree.count_nodes()


class TestSkillTreeFactoryAndMaintenance:
    def test_tree_from_proposal_creates_small_tree(self):
        tree = tree_from_proposal(
            name="Percent Tables!",
            proposal="For percentage table tasks, verify units before calculating.",
            justification="The agent missed percent formatting.",
        )
        assert tree.name == "percent-tables"
        assert tree.root.type == "condition"
        assert tree.count_nodes() == 3

    def test_maintain_tree_deduplicates_siblings(self):
        tree = SkillTree(
            name="duplicate-tree",
            root=SkillNode(
                type="condition",
                rule="task matches",
                children=[
                    SkillNode(type="action", instruction="Check units."),
                    SkillNode(type="action", instruction="Check units."),
                ],
            ),
        )
        maintained = maintain_tree(tree)
        assert maintained.count_nodes() == 2
        assert maintained.metadata["maintained"] is True

    def test_maintain_tree_trims_depth(self):
        tree = SkillTree(
            name="deep-tree",
            root=SkillNode(
                type="condition",
                rule="level 1",
                children=[
                    SkillNode(
                        type="condition",
                        rule="level 2",
                        children=[
                            SkillNode(
                                type="condition",
                                rule="level 3",
                                children=[
                                    SkillNode(
                                        type="condition",
                                        rule="level 4",
                                        children=[
                                            SkillNode(type="action", instruction="Too deep."),
                                        ],
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        )
        maintained = maintain_tree(tree, max_depth=3)
        assert maintained.max_depth() == 3


class TestSkillTreeStorage:
    def test_save_and_load_skill_tree(self, tmp_path):
        tree = _sample_tree()
        path = save_skill_tree(tmp_path, tree)
        loaded = load_skill_tree(path)
        assert loaded == tree

    def test_save_and_render_skill_tree(self, tmp_path):
        tree_path, skill_path = save_and_render_skill_tree(tmp_path, _sample_tree())
        assert tree_path.exists()
        assert skill_path.exists()
        assert "## Routing Tree" in skill_path.read_text()

    def test_load_project_skill_trees_sorted(self, tmp_path):
        first = tree_from_proposal(name="b-tree", proposal="Do B.")
        second = tree_from_proposal(name="a-tree", proposal="Do A.")
        save_skill_tree(tmp_path, first)
        save_skill_tree(tmp_path, second)
        names = [tree.name for tree in load_project_skill_trees(tmp_path)]
        assert names == ["a-tree", "b-tree"]

    def test_skill_tree_complexity(self, tmp_path):
        save_skill_tree(tmp_path, _sample_tree())
        nodes, depth = skill_tree_complexity(tmp_path)
        assert nodes == _sample_tree().count_nodes()
        assert depth == _sample_tree().max_depth()


class _FakeTrace:
    def __init__(self, output=None, parse_error=None, total_cost_usd=0.25):
        self.output = output
        self.parse_error = parse_error
        self.total_cost_usd = total_cost_usd


class _FakeAgent:
    def __init__(self, output):
        self.output = output
        self.queries = []

    async def run(self, query):
        self.queries.append(query)
        return _FakeTrace(output=self.output)


class TestSkillTreeLLMOps:
    def test_e2_crossover_with_llm_returns_structured_tree(self):
        from src.schemas import SkillTreeOperationResponse

        child = tree_from_proposal(name="child-tree", proposal="Do the child behavior.")
        agent = _FakeAgent(
            SkillTreeOperationResponse(tree=child, reasoning="combined", operator="e2")
        )

        result = asyncio.run(
            e2_crossover_with_llm(agent, _sample_tree(), child, failures="failed on units")
        )

        assert result.tree == child
        assert result.operator == "e2"
        assert result.cost_usd == pytest.approx(0.25)
        assert "LLM-guided crossover" in agent.queries[0]

    def test_m1_reflection_update_with_llm(self):
        from src.schemas import SkillTreeOperationResponse

        updated = tree_from_proposal(name="updated-tree", proposal="Always verify percent units.")
        agent = _FakeAgent(
            SkillTreeOperationResponse(tree=updated, reasoning="reflected", operator="m1")
        )

        result = asyncio.run(
            m1_reflection_update_with_llm(agent, _sample_tree(), failures="percent missed")
        )

        assert result.tree.name == "updated-tree"
        assert "reflection update" in agent.queries[0]

    def test_m2_random_semantic_mutation_with_llm(self):
        from src.schemas import SkillTreeOperationResponse

        mutated = tree_from_proposal(name="mutated-tree", proposal="Generalize unit checks.")
        agent = _FakeAgent(
            SkillTreeOperationResponse(tree=mutated, reasoning="mutated", operator="m2")
        )

        result = asyncio.run(
            m2_random_semantic_mutation_with_llm(agent, _sample_tree(), focus="units")
        )

        assert result.operator == "m2"
        assert "random semantic mutation" in agent.queries[0]

    def test_maintain_with_llm_falls_back_when_output_missing(self):
        agent = _FakeAgent(output=None)

        result = asyncio.run(maintain_with_llm(agent, _sample_tree(), max_depth=2))

        assert result.operator == "maintain"
        assert result.tree.max_depth() <= 2
