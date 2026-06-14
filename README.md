# Exit-Hazard Proxy Harness

本仓库用于验证一个在线推理行为代理：`exit_hazard`。它把模型从“仍在有效推理”转向“准备收束或进入长尾漂移”的局部风险建模为一个 hazard process，并用累积 hazard 得到在线、近似单调的进程信号。

## 项目目标

我们要找到并验证一个可在线计算的代理指标，用它量化推理进程中的退出风险：

- `exit_hazard`: 当前 prefix 附近发生 reasoning-exit transition 的局部风险。
- `exit_hazard_cumprob`: 沿生成过程累积后的退出状态概率。
- `exit_hazard_cumlogit`: `exit_hazard_cumprob` 的 logit 形式，是当前默认主代理。

这个代理后续会作为 prompt 优化目标的一部分，用来定向增强或抑制某类推理行为，例如过度推理。prompt 优化入口先按 GCD, greedy coordinate descent, 设计。

## Pipeline

先用 decode-gate 生成较长的推理轨迹，再训练 exit-hazard 代理：

```bash
/home/ssn/.conda/envs/new_env/bin/python scripts/extract_exit_hazard_features.py \
  outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged \
  --model-path /data/LLM/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --layers 24 \
  --eval-subdir exit_hazard_features
```

```bash
/home/ssn/.conda/envs/new_env/bin/python scripts/score_exit_hazard_logits.py \
  outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged \
  --model-path /data/LLM/Qwen2.5-1.5B-Instruct \
  --device cuda:0 \
  --eval-subdir exit_hazard_logits
```

```bash
/home/ssn/.conda/envs/new_env/bin/python scripts/train_exit_hazard_proxy.py \
  outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged \
  --feature-dir outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged/exit_hazard_features \
  --logit-points-csv outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged/exit_hazard_logits/exit_hazard_logit_points.csv \
  --layer 24 \
  --feature-mode static-delta-logit \
  --eval-subdir exit_hazard_proxy
```

```bash
/home/ssn/.conda/envs/new_env/bin/python scripts/evaluate_exit_hazard_proxy.py \
  outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged \
  --hazard-points-csv outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged/exit_hazard_proxy/exit_hazard_points.csv \
  --eval-subdir exit_hazard_eval
```

可选信息量评估：

```bash
/home/ssn/.conda/envs/new_env/bin/python scripts/evaluate_exit_hazard_incremental_ce.py \
  outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged \
  --hazard-points-csv outputs/exit_hazard/qwen25_15b_decode_gate_2p4_merged/exit_hazard_proxy/exit_hazard_points.csv \
  --eval-subdir exit_hazard_incremental_ce
```

## 主要输出

- `exit_hazard_feature_rows.csv`: 每个生成 token 对应的事件标签和轨迹位置。
- `hidden_layer_*.npy`: 对应 token 的内部 hidden state。
- `exit_hazard_logit_points.csv`: 低成本 logit 辅助特征。
- `exit_hazard_points.csv`: 每个 token 的 hazard 分数。
- `exit_hazard_eval_summary.csv`: 单调性、及时性、跃升对齐、长度耦合等验收结果。
- `exit_hazard_incremental_ce_summary.csv`: 在长度和 marker 控制之后，hazard 额外贡献的信息量。

## 代码结构

- `closure_delay/exit_hazard.py`: exit-hazard 事件定义、probe phrase、轨迹加载和通用数值工具。
- `scripts/probe_decode_gate.py`: 生成长推理轨迹。
- `scripts/merge_decode_gate_shards.py`: 合并多 shard 生成结果。
- `scripts/extract_exit_hazard_features.py`: 抽取 hidden-state token 特征。
- `scripts/score_exit_hazard_logits.py`: 抽取 logit 辅助特征。
- `scripts/train_exit_hazard_proxy.py`: 训练 OOF transition hazard 模型并输出累积 hazard。
- `scripts/evaluate_exit_hazard_proxy.py`: hazard-only 在线代理验收。
- `scripts/evaluate_exit_hazard_incremental_ce.py`: hazard-only incremental CE 评估。
- `scripts/optimize_suffix.py`: 后续 GCD prompt 优化入口。
- `scripts/evaluate_suffix_control_bank.py`: 精简 A-F suffix-control 主实验；`--full-control-bank` 可跑附录控制组。
- `scripts/evaluate_closure_dynamics.py`: 对已保存 generations 复算 jump、plateau、multi-step、local reset 等机制指标。
- `scripts/run_rag_suffix_benchmark.py`: fixed-retrieval / generator-only RAG suffix benchmark。
- `scripts/run_closure_aware_defense.py`: 基于逐 token 过程信号的 closure-aware defense replay。
- `docs/wise2026_experiment_result_matrix.md`: WISE 论文用的 4 个主实验 + appendix 结果矩阵。

## 说明

`outputs/` 下的 CSV、JSON、NPY 和图像是实验产物，不属于源码主线。
