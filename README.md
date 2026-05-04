# Closure-Delay Suffix Validation

本仓库用于验证新的 **controllable closure-delay suffix** 方案：通过可控强度的后缀，让模型在推理过程中延后进入“准备收束/给出最终答案”的状态，同时保留可量化的长度和控制关系。

当前 README 只描述 closure-delay 新方案。早期实验主线已经从代码和默认入口中删除。

## 项目目标

目标是构建和验证一组可控强度后缀，使它们能够按预期改变模型的 closure 行为：

- 后缀强度越高，模型越晚进入 closure 状态。
- 输出长度增长应与后缀强度和 closure-delay 效果保持可解释关系。
- 不同强度条件下应呈现基本单调趋势，而不是随机波动。
- 评估重点放在 closure risk/margin、长度比例和控制误差上。

默认验证模型是：

```text
/data/LLM/Qwen2.5-1.5B-Instruct
```

默认设备是：

```text
cuda:2
```

## 主指标

closure-delay 新方案的主指标包括：

- `closure risk`: 在给定推理位置上，模型进入 closure/最终回答状态的风险。
- `closure margin`: closure 相关候选与 continuation 相关候选之间的 margin，用于衡量模型离收束状态的距离。
- `length ratio`: attacked/generated length 与 clean baseline length 的比例。
- `control error`: 目标控制强度与实际 closure-delay/长度变化之间的偏差。
- `monotonicity`: 后缀强度增加时，closure-delay 效果和长度比例是否保持单调或近似单调。

本项目不再使用旧的 token-level uncertainty 指标作为主线目标。

## 主要入口

主入口是：

```text
scripts/validate_closure.py
```

它会加载模型、生成 clean baseline、在多个 suffix condition 下评估 closure risk/margin 曲线，并输出逐样本和逐条件汇总。

## GPU2 运行示例

使用默认 Qwen2.5-1.5B-Instruct 和 GPU2：

```bash
CUDA_VISIBLE_DEVICES=2 /home/ssn/.conda/envs/new_env/bin/python scripts/validate_closure.py \
  --model-path /data/LLM/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --output-dir outputs/closure_validation/qwen25_15b \
  --n-questions 30 \
  --max-new-tokens 512
```

快速 smoke run 可以缩小样本和生成长度：

```bash
CUDA_VISIBLE_DEVICES=2 /home/ssn/.conda/envs/new_env/bin/python scripts/validate_closure.py \
  --model-path /data/LLM/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --output-dir outputs/closure_validation/smoke_qwen25_15b \
  --n-questions 2 \
  --max-new-tokens 96 \
  --min-baseline-tokens 20 \
  --continuation-tokens 8 \
  --closure-tokens 8 \
  --fractions 0.2 0.4 0.6
```

常用参数：

- `--suffix-bank-path data/suffix_bank.json`: 额外 suffix bank。
- `--fractions`: 评估 closure curve 的推理位置比例。
- `--continuation-tokens`: continuation 参考 token 数。
- `--closure-tokens`: closure 参考 token 数。
- `--no-verbosity`: 不评估内置 verbosity suffix 条件。
- `--no-suffix-bank`: 不评估 suffix bank。
- `--no-viz`: 不生成图。

## 输出文件

默认输出目录：

```text
outputs/closure_validation/qwen25_15b
```

主要输出：

- `summary.json`: 完整配置、baseline reference 质量、closure curve、calibration 和所有 condition 结果。
- `example_metrics.csv`: 逐样本指标，包括 `baseline_length`、`attacked_length`、`length_ratio`、`mean_delta_risk`、`mean_delta_margin`、正确性和速度信息。
- `condition_summary.csv`: 逐条件汇总，包括样本数、length ratio 统计和 mean delta risk。

启用可视化时，还会在输出目录下生成 plots，用于查看 closure risk curve 以及 closure shift 与 length ratio 的关系。

## 代码结构

- `scripts/validate_closure.py`: closure-delay 验证主入口。
- `closure_delay/closure.py`: closure trajectory、risk/margin 打分和 length ratio 工具。
- `closure_delay/closure_experiments.py`: closure validation 实验编排与输出写入。
- `closure_delay/viz.py`: closure 相关图表。
- `data/suffix_bank.json`: 可选后缀库。
- `outputs/closure_validation/`: closure-delay 实验输出目录。

## 说明

本项目是受控本地验证 harness，不是线上攻击工具。当前主线只围绕 controllable closure-delay suffix 展开。
