# AGENT_SPEC — uni-thesis-engine 系统铁律

所有参与本仓库开发的人类与 Agent 必须遵守以下三条铁律。任何违反铁律的提交都不得合入 `main`。

## 铁律 1：引擎与具体课题解耦
- `src/` 内的代码只实现通用能力（文献检索、数据加载、实验调度、统计检验、排版渲染）。
- **禁止**在 `src/` 中硬编码任何特定课题的实验常量（采样率、因子水平、阈值、文件路径、方法名等）。
- 课题相关的一切配置必须写在 `profiles/*.yaml`（由 `TopicProfile` 校验），课题专属算法放在 `plugins/`。

## 铁律 2：零幻觉数据溯源
- **严禁**伪造、估算或"美化"任何实验结果。所有数值必须由实际运行产生，并写入带 SHA-256 指纹的 `ExperimentResultBundle`（含 `git_commit` 与 `dataset_sha256`）。
- 文献条目必须经过 DOI 核验后才能标记 `verified=True`；未核验条目不得进入正文引用。
- `.tex` 正文中**严禁手写实验裸数字**；所有结果数字必须通过模板宏/自动生成的表格从结果包注入。

## 铁律 3：严禁占位实现
- **严禁**使用 `pass`、`TODO`、`FIXME`、`...` 或空函数体作为占位。
- 每个函数要么完整实现并有测试覆盖，要么不存在。

## 目录约定
| 路径 | 用途 |
| --- | --- |
| `src/uni_thesis/` | 通用引擎代码（`schema.py` 为跨模块数据契约） |
| `profiles/` | 课题配置 YAML（`TopicProfile`） |
| `plugins/` | 课题专属算法插件 |
| `results/` | 实验结果包（受版本控制的仅 `.gitkeep`） |
| `templates/thesis_tex/` | LaTeX 论文模板 |
| `scripts/` | 运维/批处理脚本 |
| `tests/` | 单元测试 |

## 自动执行（Step 4 起）
- `uni-thesis build-tex results/<profile_id>/metrics.json` 生成 `templates/thesis_tex/generated/auto_metrics.tex` 与 `auto_tables.tex`（带 `metrics_sha256` 溯源头）。
- `python scripts/audit_anti_hallucination.py --tex-dir templates/thesis_tex --bib templates/thesis_tex/refs.bib --metrics results/<profile_id>/metrics.json` 执行三道门禁（裸数字 / 引用真实性 / 溯源哈希），非零退出即禁止合入。
