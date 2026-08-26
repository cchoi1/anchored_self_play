#!/bin/bash

set -e
mkdir -p logs
set -euo pipefail


MODEL="gpt-5.2"
MODE="solver-attacker-style"
WORKERS=2
WORKERS_EVAL=10

declare -A DATASETS=(
    ["human"]="cchoi1/bugs_human_edited_lm_eval"
    ["qwen"]="cchoi1/bugs_lm_qwen7b_eval"
    ["adv"]="cchoi1/bugbench_adversarial_new"
    ["oss"]="cchoi1/bugs_lm_gpt_oss_20b_eval"
)

for DS in human qwen adv oss; do
    INPUT="${DATASETS[$DS]}"
    OUTPUT="${MODEL}_solver_attacker_${DS}.json"
    EVAL_OUTPUT="${MODEL}_solver_attacker_${DS}_eval.json"

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 || true

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 \
        --continue-from "$OUTPUT" || true

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --eval "$OUTPUT" --output "$EVAL_OUTPUT" --workers $WORKERS_EVAL || true
done
