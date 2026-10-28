# uni-thesis-engine

[![ci](https://github.com/Vocaliodream/uni-thesis-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Vocaliodream/uni-thesis-engine/actions/workflows/ci.yml)

通用、与课题解耦的毕业论文自动化引擎：**一份 YAML 课题配置 → DOI 核验文献 → 可复现实验 → 自动路由统计检验 → 600 DPI 图表 → LaTeX 论文 → 反幻觉审计 → PDF**。
论文正文中的每一个实验数值都来自签名结果包生成的宏，每一条引用都经过 Crossref DOI 核验；手写数字、伪造引用或与结果包不匹配的生成文件都会被审计拦截。

内置两个完全不同领域的示范课题：

| Profile | 领域 | 内容 |
| --- | --- | --- |
| `profiles/demo_biosignal.yaml` | 信号处理 | 多模态 ECG/PPG（250 Hz），滑动平均 / Savitzky–Golay / 四阶零相位 Butterworth 对比，ΔSNR、RMSE、Pearson r、Bland–Altman |
| `profiles/demo_rsm_opt.yaml` | 工艺参数优化 | 三因子中心复合设计响应面、ANOVA 与失拟检验、四阶段累加式消融 |

---

## 五层架构

```
┌───────────────────────────────────────────────────────────────────────────┐
│ ⑤ 论文与审计层  tex_bridge.py · audit.py · templates/thesis_tex(_zh)        │
│    结果宏 / 三线表 / 图表 / 文献 → LaTeX；三道反幻觉闸门；pdflatex/xelatex   │
├───────────────────────────────────────────────────────────────────────────┤
│ ④ 统计与制图层  stat_router.py · fig_foundry.py                             │
│    正态/方差齐性诊断 → t / Welch / Wilcoxon / Mann–Whitney / ANOVA+Tukey /   │
│    Kruskal–Wallis；二阶响应面 + ANOVA；600 DPI PNG + 矢量 PDF               │
├───────────────────────────────────────────────────────────────────────────┤
│ ③ 实验执行层  exp_harness.py · plugins/                                     │
│    信号模拟器与滤波器（SOS Butterworth）、CCD 设计、消融、可注册目标函数；   │
│    结果包含 git 提交号、输入 SHA-256、随机种子与逐次原始数据                 │
├───────────────────────────────────────────────────────────────────────────┤
│ ② 文献层  lit_engine.py                                                    │
│    OpenAlex + Crossref polite pool，DOI 逐条核验，签名缓存，离线回退         │
├───────────────────────────────────────────────────────────────────────────┤
│ ① 课题配置层  schema.py · profiles/*.yaml                                   │
│    Pydantic 严格校验的 TopicProfile：领域、文献检索式、数据/因子、方法、指标 │
└───────────────────────────────────────────────────────────────────────────┘
        编排：pipeline.py（run / audit / pdf） · 入口：cli.py（uni-thesis）
```

运行产物位于 `results/<profile_id>/`：`metrics.json`（签名结果包）、`raw_series.npz`、`lit_cache.json`、`literature_matrix.{csv,md}`、`figures/`、`thesis/`、`run_manifest.json`、`thesis.pdf`。

## 快速开始

```bash
git clone https://github.com/Vocaliodream/uni-thesis-engine.git
cd uni-thesis-engine
make init                                        # pip install -e . 并注册 pre-commit 门禁
make run PROFILE=profiles/demo_biosignal.yaml    # 全流水线
make pdf PROFILE=profiles/demo_biosignal.yaml    # 需要 TeX Live（pdflatex + bibtex）
make all                                         # 两个内置课题 + 测试（覆盖率 ≥ 90%）+ 反幻觉审计
```

常用命令：

| 命令 | 作用 |
| --- | --- |
| `make test` | `pytest --cov=src/uni_thesis --cov-fail-under=90`（警告即错误） |
| `make audit` | 审计 `results/` 下所有已生成论文 |
| `make audit-fresh` | 用当前代码与模板离线重建每个课题后再审计（pre-commit 使用此模式） |
| `uni-thesis run --profile P [--offline] [--template-dir templates/thesis_tex_zh]` | 运行流水线；`--offline` 只用签名缓存；中文模板走 xelatex |
| `uni-thesis audit --profile P [--strict-integers]` | 单课题审计；默认放行普通整数，`--strict-integers` 恢复严格模式 |
| `uni-thesis validate-profile P` | 只校验 YAML |

中文论文：`uni-thesis run --profile P --template-dir templates/thesis_tex_zh && uni-thesis pdf --profile P`（ctex + TeX Live 自带开源 Fandol 字体，需 `texlive-xetex texlive-lang-chinese`）。

## 教程：用不到 50 行 YAML 接入一个全新毕设课题

以“再生骨料混凝土配合比响应面优化”为例，完整文件见 [`profiles/examples/tutorial_concrete_mix.yaml`](profiles/examples/tutorial_concrete_mix.yaml)（30 行）。

**第 1 步：写课题配置**

```yaml
profile_id: tutorial_concrete_mix            # 结果目录名：字母数字、下划线或连字符
title: Response Surface Optimisation of Recycled-Aggregate Concrete Mix Design
title_zh: 再生骨料混凝土配合比响应面优化研究     # 中文模板使用
domain_type: parameter_optimization          # 或 signal_processing
literature_query:                            # 文献层检索式
  keywords: [recycled aggregate concrete, compressive strength, response surface methodology, mix design]
  start_year: 2000
  max_papers: 6
data_config:
  factor_ranges:                             # 因子及上下限 → 自动生成 CCD
    water_cement_ratio: {low: 0.35, high: 0.55}
    recycled_pct: {low: 0, high: 60}
  simulator:
    name: analytic_response_surface          # 已注册的目标函数（可换成自己的插件）
    params:
      metric_name: strength_mpa
      intercept: 20.0
      linear: {water_cement_ratio: 120.0, recycled_pct: 0.05}
      quadratic: {water_cement_ratio: -160.0, recycled_pct: -0.002}
      interaction: {"water_cement_ratio*recycled_pct": -0.1}
      noise_std: 0.6
design: {kind: ccd, center_points: 5}
baselines: [center]
metrics_target: [strength_mpa]
metrics_direction: {strength_mpa: max}
random_seed: 42
n_repeats: 3
```

信号类课题改用 `domain_type: signal_processing`，在 `data_config` 中给出 `sampling_rate_hz` 与 `simulator`（`ecg` / `ppg` / `tof` / `multichannel`），列出 `baselines`、`proposed_method`、`method_params` 与指标即可，参照 `profiles/demo_biosignal.yaml`。需要消融时加入 `ablation: {modules: [...], mode: incremental|removal}`。

**第 2 步（可选）：接入真实模型或数据**

在 `plugins/` 下放一个 Python 文件，用装饰器注册，YAML 中 `simulator.name` 写同名即可：

```python
# plugins/my_lab_model.py
from uni_thesis.exp_harness import register_objective

@register_objective("my_lab_model")
def my_lab_model(config, params, rng):
    x = config["factors"]                     # 当前设计点，如 {"water_cement_ratio": 0.45, ...}
    y = ...                                   # 调用你的仿真 / 读取实测数据
    return {"strength_mpa": float(y)}
```

**第 3 步：运行、审计、出 PDF**

```bash
uni-thesis validate-profile profiles/examples/tutorial_concrete_mix.yaml
make run PROFILE=profiles/examples/tutorial_concrete_mix.yaml
make pdf PROFILE=profiles/examples/tutorial_concrete_mix.yaml
```

模板与课题无关：章节只引用角色宏（`\MetricPrimaryProposedMean`、`\StatRsmPrimaryOptPoint` 等），并由 `\ifAutoHasComparison / Ablation / Rsm` 自动开关对应小节。写作时只需在 `templates/thesis_tex/chapters/` 中补充文字，**不要手写任何实验数字**。

## 防幻觉机制

1. **数值溯源闸门（raw-number gate）**：方法、实验、结论章节中出现的小数、百分比、p 值或科学计数法一律拦截，必须改用自动生成的 `\Metric… / \Stat… / \Diff…` 宏；普通整数（被试数、折数）默认放行，`--strict-integers` 可收紧。
2. **引用闸门（citation gate）**：每个 `\cite{}` 键必须存在于 `refs.bib`，带合法 DOI，且出现在经 Crossref 核验的签名文献缓存中。
3. **结果包绑定闸门（provenance gate）**：生成的 `.tex` 头部记录 `metrics.json` 的 SHA-256，审计时重新计算比对；正文使用的每个结果宏都必须在生成文件中定义。结果包本身携带 git 提交号、输入数据哈希与随机种子。

## 双重门禁：本地 pre-commit + GitHub Actions

- `bash scripts/install_hooks.sh`（`make init` 会调用）生成 `.git/hooks/pre-commit`：每次 `git commit` 依次运行 `pytest`（覆盖率 ≥ 90%）与 `scripts/audit_anti_hallucination.py`（用当前代码和全部模板离线重建每个课题再审计），任一失败即以非零状态码阻断提交。
- `.github/workflows/ci.yml` 在 push 到 `main`、推送 `v*` 标签以及 Pull Request 时触发：
  - `test-and-audit`：Python 3.12，测试 + 覆盖率门槛，完整运行两个内置课题（文献检索失败时回退签名缓存）并执行反幻觉审计；
  - `build-thesis-pdf`：安装轻量 TeX Live，`pdflatex + bibtex` 编译 `thesis_biosignal.pdf`、`thesis_rsm.pdf`，连同 600 DPI 图表包上传为 Artifact；`v*` 标签自动发布到 GitHub Releases。

## 开发约定

见 [`AGENT_SPEC.md`](AGENT_SPEC.md)。测试：`python -m pytest -v -W error`。OpenAlex 无 API key 时经常返回 429，系统会自动降级到 Crossref。
