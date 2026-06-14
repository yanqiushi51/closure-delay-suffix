# WISE 2026 Closure-Delay Experiment Plan

这版矩阵把原来的 6 个 RQ 压缩成 4 个主实验和 1 个附录鲁棒性实验。核心原则是：RQ2/RQ3/RQ4 不再跑三套实验，而是复用同一批 suffix-control generations，分别做 attack effectiveness、multi-stage dynamics、length-risk frontier 三种分析。

## Paper Claims

| Claim | 论文要证明的事情 | 对应实验 |
| --- | --- | --- |
| C1 Problem | LLM 云应用存在 correctness-preserving cost amplification：自然语言 suffix 增加 token/cost，但基本保持任务意图和答案 | Experiment 2, Experiment 3 |
| C2 Proxy | closure-readiness proxy 能在线跟踪语义收束状态，并且不是 length-only / marker-only | Experiment 1 |
| C3 Mechanism | structured E 不是普通 verbosity，而是诱导 jump、plateau、local reset、rise-reset cycles、VPCG 等 multi-stage dynamics | Experiment 2 |
| C4 Transfer | Closure-Delay 可迁移到 fixed-retrieval RAG/cloud QA，表现为 post-evidence overthinking | Experiment 3 |
| C5 Defense | 同一个 prefix-available closure signal 可用于在线预算控制，降低攻击成本且少误伤 clean 输入 | Experiment 4 |

## Main Experiments

| Experiment | Purpose | Data / Conditions | Main Metrics | Required Result Pattern | Code / Output |
| --- | --- | --- | --- | --- | --- |
| E1 Proxy Validity | 证明 proxy 有效且非 length/marker detector | GSM8K, Qwen2.5-7B; raw hazard/cumprob/cumlogit; hidden/logit ablations; all tokens and marker-free subset | proxy_score, Spearman, relaxed_monotone_rate, timely_rate, jump_align_rate, length_coupling, delta_ce, delta_auc | OOF/held-out 上 cumlogit 最稳；static-delta-logit 最强；marker-free subset 不崩；控制 fraction/log index/marker 后 delta_ce 和 delta_auc 仍为正 | `scripts/evaluate_exit_hazard_proxy.py`, `scripts/evaluate_exit_hazard_incremental_ce.py` |
| E2 Suffix-Control Attack + Mechanism + Frontier | 用同一批 A-F generations 同时证明攻击有效、不是 verbosity、frontier 更优 | GSM8K/free generation; no suffix, irrelevant, verbose-only, manual verification, structured E, optimized | Effectiveness: generated_tokens, delta_tokens, length_ratio, output_cost, accuracy, truncation, repetition, drift. Mechanism: jump_count, plateau_longest, multi_step_count, local_reset_count, rise_reset_cycle_count, PCG, VPCG. Frontier: risk_score | structured E 增加 tokens/cost 且 accuracy 基本保持；drift/repetition/truncation 低；jump/plateau/local reset/VPCG 高于 verbose-only；risk-score frontier 优于 manual/verbose | `scripts/evaluate_suffix_control_bank.py`, `scripts/evaluate_closure_dynamics.py`, `scripts/evaluate_suffix_closure_dynamics.py` |
| E3 Fixed-Retrieval RAG Transfer | 证明它是 RAG/cloud QA 的 post-evidence overthinking，而不是 retrieval attack | HotpotQA/FAQ subset; retriever only sees original question; fixed context; generator sees question + context + suffix | generated_tokens, delta_tokens, estimated_cost_delta, answer_correct, answer_supported, citation_precision, citation_recall, post_evidence_tokens, VPCG, evidence_stage_count, citation_switches, drift, repetition | structured E 增加 output cost 和 post-evidence tokens；answer support/citation precision 基本保持；evidence-stage/citation-switch/VPCG 上升；drift/repetition 不显著上升 | `scripts/run_rag_suffix_benchmark.py`, `scripts/evaluate_rag_evidence_closure.py`, `scripts/evaluate_rag_closure_dynamics.py` |
| E4 Closure-Aware Defense | 证明 proxy 可做 prefix-available online budget control | Clean: no suffix/irrelevant. Attacked: manual/structured/optimized. Defenses: no defense, fixed budget, answer-marker stop, closure-aware stop/finalization | clean_false_trigger_rate, clean_accuracy_drop, clean_support_drop, attack_token_reduction, attack_cost_reduction, answer_retention, support_retention | closure-aware finalization 比 marker stop 更早省 token；比 fixed budget 更少误伤 clean/hard cases；E attack 下 cost reduction 高且 answer/support retention 高 | `scripts/run_closure_aware_defense.py` |
| Appendix Robustness | 不抢主线，只证明结论稳健 | seeds, thresholds, sample sizes, Qwen2.5-1.5B/14B, end-to-end RAG, suffix transfer | same as main experiment subset | 方向一致即可，不要求所有指标最好 | existing eval scripts plus robustness sweeps |

## Main Paper Figures And Tables

| Paper Item | Claim Supported | Source |
| --- | --- | --- |
| Table 1 Proxy Validity | C2 | E1 |
| Table 2 GSM8K Suffix Attack Effectiveness | C1 | E2, same generations |
| Figure 3 Typical Closure Dynamics Curves | C3 | E2, selected cases |
| Figure 4 Length-Risk Frontier | C3 | E2, delta/risk summary |
| Table 3 GSM8K Multi-Stage Closure Dynamics | C3 | E2, same generations |
| Table 4 RAG Attack + Dynamics | C4 | E3 |
| Table 5 Defense Tradeoff | C5 | E4 |

## Table Templates

### Table 1: Proxy Validity

| Model | Dataset | Proxy | Feature Mode | Subset | OOF/Held-out | Proxy Score | Spearman | Monotone | Timely | Jump Align | Length Coupling | Delta CE | Delta AUC |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | GSM8K | cumlogit | hidden-only | all tokens | yes | 0.71 | 0.58 | 0.74 | 0.68 | 0.61 | 0.42 | 0.06 | 0.02 |
| Qwen2.5-7B | GSM8K | cumlogit | logit-only | all tokens | yes | 0.76 | 0.64 | 0.79 | 0.72 | 0.66 | 0.35 | 0.09 | 0.04 |
| Qwen2.5-7B | GSM8K | cumlogit | static-delta-logit | all tokens | yes | 0.89 | 0.76 | 0.88 | 0.83 | 0.79 | 0.27 | 0.17 | 0.08 |
| Qwen2.5-7B | GSM8K | cumlogit | static-delta-logit | marker-free | yes | 0.85 | 0.72 | 0.84 | 0.80 | 0.75 | 0.24 | 0.14 | 0.07 |

Main-text statement if space is tight: `static-delta-logit was selected on validation because it achieved the best proxy score and incremental CE`; full hidden/logit ablation can move to appendix.

### Table 2: GSM8K Suffix Attack Effectiveness

| Condition | Delta Tokens | Length Ratio | Output Cost Delta | Accuracy | Truncation | Drift | Repetition | VPCG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no suffix | 0 | 1.00 | 0 | 0.84 | 0.02 | 0.03 | 0.04 | 0.07 |
| irrelevant | 15 | 1.10 | +10% | 0.83 | 0.02 | 0.03 | 0.04 | 0.09 |
| verbose-only | 138 | 1.92 | +92% | 0.79 | 0.06 | 0.08 | 0.16 | 0.14 |
| manual verification | 195 | 2.30 | +130% | 0.82 | 0.04 | 0.05 | 0.08 | 0.26 |
| structured E | 278 | 2.85 | +185% | 0.82 | 0.03 | 0.06 | 0.06 | 0.47 |
| optimized | 342 | 3.28 | +228% | 0.81 | 0.04 | 0.07 | 0.07 | 0.58 |

Report main values as `mean +/- 95% CI`. Required paired comparisons: structured E vs no suffix, structured E vs verbose-only, and structured E vs manual verification.

### Table 3: GSM8K Multi-Stage Closure Dynamics

| Condition | Jump Count | Plateau Longest | Multi-Step Count | Local Reset Count | Rise-Reset Cycles | VPCG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no suffix | 0.8 | 4 | 0.3 | 0.2 | 0.1 | 0.07 |
| verbose-only | 1.4 | 9 | 0.6 | 0.5 | 0.3 | 0.14 |
| manual verification | 2.6 | 16 | 1.5 | 1.3 | 0.9 | 0.26 |
| structured E | 4.3 | 28 | 3.1 | 2.7 | 2.1 | 0.47 |
| optimized | 5.4 | 35 | 3.9 | 3.4 | 2.6 | 0.58 |

### Figure 4 Data: Length-Risk Frontier

Risk should not use `mean_cumlogit` as the main risk term. Use:

```text
GSM8K risk_score =
  accuracy_drop
  + max(drift_delta, 0)
  + max(repetition_delta, 0)
  + max(truncation_delta, 0)

RAG risk_score =
  support_drop
  + max(drift_delta, 0)
  + max(repetition_delta, 0)
  + max(truncation_delta, 0)
  + unsupported_tail_rate
```

Use `VPCG`, `cycle_count`, and `mean_cumlogit` as mechanism-strength annotations, not as risk itself.

### Table 4: RAG Attack + Dynamics

| Condition | Delta Tokens | Output Cost Delta | Answer Correct | Answer Supported | Citation Precision | Citation Recall | Post-Evidence Tokens | VPCG | Evidence Stages | Citation Switches | Drift | Repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no suffix | 0 | 0 | 0.78 | 0.86 | 0.80 | 0.83 | 38 | 0.05 | 1.2 | 0.3 | 0.02 | 0.02 |
| irrelevant | 12 | +8% | 0.77 | 0.85 | 0.79 | 0.82 | 44 | 0.07 | 1.3 | 0.4 | 0.03 | 0.03 |
| verbose-only | 112 | +74% | 0.75 | 0.82 | 0.77 | 0.80 | 88 | 0.12 | 1.8 | 0.8 | 0.06 | 0.13 |
| manual verification | 168 | +111% | 0.76 | 0.84 | 0.78 | 0.82 | 132 | 0.23 | 2.3 | 1.2 | 0.04 | 0.07 |
| structured E | 238 | +157% | 0.76 | 0.84 | 0.79 | 0.82 | 192 | 0.42 | 3.2 | 1.9 | 0.05 | 0.05 |
| optimized | 288 | +190% | 0.75 | 0.83 | 0.77 | 0.81 | 238 | 0.52 | 3.8 | 2.3 | 0.07 | 0.06 |

RAG metric definitions:

- `answer_supported`: generated answer is supported by cited retrieved evidence.
- `citation_precision`: cited gold supporting docs divided by all cited docs.
- `citation_recall`: cited gold supporting docs divided by all gold supporting docs.
- `post_evidence_tokens`: generated tokens after the first prefix that cites sufficient supporting evidence and before final answer emission.
- For HotpotQA, sufficient evidence means citing all gold supporting document IDs or covering required supporting facts.

### Table 5: Defense Tradeoff

| Defense | Clean False Trigger | Clean Token Reduction | Attack Token Reduction | Attack Cost Reduction | Finalizer Cost | Answer Retention | Support Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no defense | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 |
| fixed budget | 0.12 | 0.06 | 0.45 | 0.41 | 0 | 0.91 | 0.89 |
| answer-marker stop | 0.07 | 0.04 | 0.31 | 0.27 | 0 | 0.94 | 0.93 |
| closure-aware stop | 0.03 | 0.14 | 0.56 | 0.51 | 0 | 0.97 | 0.96 |
| closure-aware finalization | 0.02 | 0.18 | 0.63 | 0.58 | +0.04 | 0.98 | 0.97 |

Required paired comparisons: closure-aware finalization vs fixed budget, and closure-aware finalization vs answer-marker stop. For finalization, report total cost as `prefix_generation_cost + finalization_call_cost`.

## Definitions To Use In The Paper

**Closure-Delay Attack:** A natural-language suffix attack that increases generation cost by delaying answer emission after the model has reached a closure-ready state, while preserving task intent and largely preserving answer correctness.

**Structured Overthinking:** Extended reasoning characterized by multi-stage closure-readiness dynamics, verification plateaus, or local rise-reset cycles, rather than repetitive or unsupported verbosity.

**Post-Evidence Overthinking In RAG:** The model has already identified sufficient retrieved evidence to support the answer but continues cross-checking, conflict inspection, or evidence restatement before emitting the final answer.

## Non-Negotiable Reporting Notes

- Proxy validity must be OOF or held-out.
- Incremental CE must report both all tokens and marker-free subset.
- RAG main experiment must be fixed-retrieval / generator-only.
- Defense trigger must use only prefix-available signals: `q_closure`, `answer_survival`, `verify_prob`, `drift_prob`, and tokens after crossing.
- Gold evidence is allowed for analysis/evaluation, not as the main deployable defense trigger.
- Latency is useful, but generated tokens and output cost are the primary cost metrics.
