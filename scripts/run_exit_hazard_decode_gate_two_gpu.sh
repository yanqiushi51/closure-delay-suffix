#!/usr/bin/env bash
set -euo pipefail

PY="/home/ssn/.conda/envs/new_env/bin/python"
SCRIPT="/data/ssn/scripts/probe_decode_gate.py"
MODEL="/data/LLM/Qwen2.5-1.5B-Instruct"
OUT_BASE="/data/ssn/outputs/exit_hazard"

SEEDS=(42 43 44 45 46 47 48 49 50 51)
GPUS=(0 1)
N_PER_SHARD=100

run_shard() {
  local seed="$1"
  local gpu="$2"
  local shard_idx="$3"
  local out_dir="${OUT_BASE}/qwen25_15b_decode_gate_2p4_shard${shard_idx}_seed${seed}"
  local log_file="${out_dir}.log"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${SCRIPT}" \
    --model-path "${MODEL}" \
    --device cuda:0 \
    --output-dir "${out_dir}" \
    --n-questions "${N_PER_SHARD}" \
    --max-new-tokens 1024 \
    --target-ratios 2.4 \
    --max-baseline-tokens 360 \
    --phrase-set base \
    --hard-block \
    --post-gate-boost 4 \
    --boost-eos \
    --post-gate-continuation-hard-block \
    --continuation-block-from-start \
    --force-min-new-tokens \
    --post-gate-slack-tokens 16 \
    --no-repeat-ngram-size 8 \
    --seed "${seed}" > "${log_file}" 2>&1
}

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$((i % 2))]}"
  run_shard "${seed}" "${gpu}" "${i}" &
  pids+=("$!")
  echo "launched shard=${i} seed=${seed} gpu=${gpu} pid=${pids[-1]}"
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "decode-gate shards completed"
