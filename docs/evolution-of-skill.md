# Evolution of Skill

郑执-CIAM

## 研究现状

Opus 4.7 等高级模型能够较好地理解和执行复杂 skill，但 Qwen-3-14B、GPT-4o-mini 等小规模模型在使用复杂 skill 时容易出现理解偏差、执行不稳定或忽略关键步骤的问题。这会导致现有 Trace2Skill、SkillOpt 等方法在小模型 setting 下失效，甚至出现 WebArena 上使用 skill 的表现不如裸跑的情况。

这说明当前 skill 的主要瓶颈不只是“能否从 trace 中总结出经验”，还包括总结出的 skill 是否能被目标模型稳定读取、路由和执行。对于小模型而言，长文本、隐式条件、交叉依赖和冗余描述都会显著增加 skill 的使用难度。

## 核心思路

将 skill 从自然语言说明进一步结构化，建模为一种可执行的决策树。每个 instance 在使用 skill 时，本质上是在树上做 routing：根据输入、状态、页面信息、任务类型或历史错误，依次判断应该进入哪些分支，并触发对应的注意事项、操作策略或答案格式约束。

一个简单 skill 例如：

1. 对于数据 A 多加关注。
2. 对于数据 B 使用某种回答格式。

可以转写为树结构：

```text
root
├── if A
│   └── action: 多加关注
└── if B
    └── action: 使用指定回答格式
```

这种表达方式的优势是：

- 条件和动作分离，减少小模型一次性理解长段 skill 的负担。
- instance 只需要经过相关分支，避免读入大量无关规则。
- 后续交叉、变异、裁剪都可以发生在子树层面。
- 树搜索可以帮助维护 routing 结构，减少交叉变异后的冗余和冲突。

借鉴 EoH 中维护种群并持续演化 heuristic 的经验，可以将每个 skill 表示为一棵树，将多个 skill tree 作为种群，在 benchmark 上迭代选择、交叉、变异和淘汰。

## Skill Tree 表示

每棵 skill tree 可以由以下节点组成：

- `condition`：进入该分支的判断条件，例如任务类型、数据形态、页面状态、错误模式。
- `action`：模型进入该分支后应该执行的策略、检查项或输出约束。
- `reflection`：从失败 trace 或评估反馈中总结出的修正说明。
- `metadata`：节点来源、命中次数、贡献分数、最近一次更新迭代等辅助信息。

一个可落地的数据结构可以是：

```yaml
name: table_numeric_answering
root:
  type: condition
  rule: "任务是否涉及表格中的数值比较或计算"
  children:
    - type: condition
      rule: "是否存在数据 A"
      children:
        - type: action
          instruction: "优先定位数据 A 的行列位置，并在计算前复核单位。"
    - type: condition
      rule: "是否需要按照数据 B 的格式回答"
      children:
        - type: action
          instruction: "最终答案使用指定格式，不添加额外解释。"
```

实际执行时，可以将整棵树直接提供给 agent，也可以先由一个 lightweight router 将相关路径抽取出来，再把路径上的 action 压缩成当前 instance 的 skill context。

## 交叉与变异

树结构的特殊性在于，交叉和变异可以在子树层面进行。相比直接编辑长文本 skill，树结构更容易控制冗余、冲突和局部退化。

拟采用以下算子：

- `e1`：交叉变异。无需大模型，从两个 skill tree 中选择兼容子树进行交换、合并或挂载。
- `e2`：生成不太一样的交叉变异。需要大模型，先给出两个父代 skill tree 和目标失败模式，再生成结构上更有差异的新子树。
- `m1`：reflection 更新当前 skill。需要大模型，基于失败 trace 和评估反馈，对命中路径上的 condition/action/reflection 进行局部修正。
- `m2`：随机语义修改。需要大模型，对某些节点进行改写、泛化、特化或重新组织。
- `m3`：删除部分 branch。无需大模型，基于低命中率、负贡献或冗余检测裁剪子树。

其中 `e1` 和 `m3` 是低成本结构操作，适合高频使用；`e2`、`m1` 和 `m2` 依赖大模型，适合在关键迭代或停滞时触发。

## 树结构维护

交叉和变异后，需要保证 skill tree 仍然是可路由、可读、低冗余的结构。树维护可以由大模型辅助完成，主要包括：

- 合并语义重复的 condition。
- 删除互相矛盾或覆盖关系不清的分支。
- 将过长 action 拆成更小的步骤。
- 将隐式条件显式化。
- 为每个 action 补充适用范围，避免泛化过度。
- 将多个子树整理为统一的 routing 顺序。

维护目标不是让 skill 更长，而是让小模型更容易在具体 instance 上命中正确路径并执行正确动作。

## 进化流程

整体流程如下：

1. 初始化一个 skill tree 种群，可以来自人工 seed、Trace2Skill、SkillOpt 或 EvoSkill 当前生成的自然语言 skill。
2. 在训练集上运行 agent，记录失败 trace、命中路径、输出错误类型和评估反馈。
3. 基于验证集表现为每棵 skill tree 打分，同时记录节点级别的命中率和贡献。
4. 选择父代 skill tree，应用 `e1/e2/m1/m2/m3` 生成候选子代。
5. 对候选子代进行树结构维护，保证 routing 合法且低冗余。
6. 在 held-out validation set 上评估候选子代。
7. 保留性能提升或具有多样性价值的子代，更新种群与 frontier。
8. 重复迭代，直到达到预算、性能收敛或连续多轮无提升。

## 实验设计

实验目标是验证结构化 skill tree 是否能让小模型也稳定受益，而不是只在强模型上有效。

### Benchmark

优先考虑 SkillBench 等新方法的 setting，尽量和现有工作保持可比。可以额外加入 WebArena 或类似网页任务，重点观察复杂 skill 在小模型上的退化问题是否被缓解。

### 模型

- 小模型主测：Qwen-3-14B、GPT-4o-mini。
- 强模型参考：Opus 4.7 等能够较好处理复杂 skill 的模型。
- 可选扩展：加入不同开源模型，观察 skill tree 对模型规模和指令遵循能力的敏感性。

### Baseline

- 裸跑 agent，不使用 skill。
- Trace2Skill。
- SkillOpt。
- EvoSkill 原始自然语言 skill/prompt mutation。
- 结构化 skill tree，但不做进化。
- 结构化 skill tree，加完整进化算子。

### Metrics

- 任务成功率或 benchmark score。
- 相比裸跑的提升幅度。
- skill 使用后的负迁移比例。
- 小模型上的 invalid routing / invalid execution 频率。
- skill token 长度和实际命中路径长度。
- 单轮进化成本，包括大模型调用次数和 token cost。
- 节点级别贡献，例如分支命中率、命中后成功率、裁剪前后性能变化。

### Ablation

- 去掉 `e1`，观察无交叉的影响。
- 去掉 `e2`，观察无大模型交叉生成的影响。
- 去掉 `m1`，观察无 reflection 局部更新的影响。
- 去掉 `m2`，观察随机语义修改是否必要。
- 去掉 `m3`，观察树裁剪对冗余和性能的影响。
- 不做树维护，观察交叉变异后结构退化的程度。
- 只提供完整树 vs 只提供命中路径，观察 context 压缩对小模型的影响。

## 与当前 EvoSkill 仓库的关系

当前 EvoSkill 已经具备 agent 运行、失败收集、skill/prompt 生成、验证集评估和 frontier 维护能力。Evolution of Skill 可以作为一个新的 skill representation 与 mutation backend 接入现有 loop：

- 在 registry 中新增 skill tree program 的存储和版本管理。
- 在 proposer/generator 阶段增加 tree-aware proposer 和 tree maintainer。
- 在 evaluation 阶段记录节点命中和节点贡献。
- 在 loop 中加入种群选择与树结构交叉变异算子。
- 在 harness 层增加 skill tree 到 agent-readable context 的渲染方式。

最小可行实现可以先从 YAML skill tree 开始：将自然语言 skill 转换为树，支持 `e1` 子树交叉、`m3` 子树裁剪、`m1` reflection 更新，然后在 SkillBench 小模型 setting 上与自然语言 skill 做对比。

## 当前仓库中的 MVP 实现

当前仓库已加入 skill tree 基础模块，位置在 `src/skill_tree/`：

- `models.py`：定义 `SkillTree` 和 `SkillNode`，约束 root 必须是 condition，action/reflection 必须是叶子节点。
- `renderer.py`：将 skill tree 渲染成当前 harness 能读取的 `SKILL.md` Markdown 格式。
- `fitness.py`：实现 `fitness = validation_score - complexity_penalty`，当前复杂度惩罚主要包括节点数、深度和可选 token 长度。
- `operators.py`：实现低成本 `e1` 子树交叉和 `m3` 基于 metadata 的 branch 裁剪。
- `llm_ops.py`：实现需要大模型参与的 `e2`、`m1`、`m2` 和 LLM tree maintenance 调用。
- `maintainer.py`：实现无需大模型的去重、限深、限节点数维护。
- `storage.py`：将 YAML skill tree 存储到 `.evoskill/skill_trees/`，并渲染到 `.claude/skills/<name>/SKILL.md`。

当前 `SelfImprovingLoop` 已支持新的 `evolution_mode = "skill_tree"`。默认 `skill_only` 和 `prompt_only` 行为不变；只有配置为 `skill_tree` 时，loop 才会使用 skill tree operator agent 生成/维护结构化 skill tree，并用 fitness 作为 frontier score。

可在 `.evoskill/config.toml` 中启用：

```toml
[evolution]
mode = "skill_tree"
```

## 预期贡献

该方向的核心贡献是把 skill 从“长文本经验总结”推进到“可路由、可组合、可进化的结构化程序”。如果实验成立，它可以解释并缓解小模型无法稳定使用复杂 skill 的问题，同时为 EvoSkill 提供更适合小模型和多轮进化的 skill 表示方式。
