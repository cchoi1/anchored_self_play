#!/bin/bash

set -e
mkdir -p logs
set -euo pipefail


MODEL="claude-sonnet-4-5-20250929"
MODE="solver-diff"
WORKERS=2
WORKERS_EVAL=10

declare -A DATASETS=(
    ["human_authored"]="cchoi1/bugs_human_authored_eval"
    ["human_edited_lm"]="cchoi1/bugs_human_edited_lm_eval"
    ["qwen"]="cchoi1/bugs_lm_qwen7b_eval"
    ["oss"]="cchoi1/bugs_lm_gpt_oss_20b_eval"
)
for DS in "${!DATASETS[@]}"; do
    INPUT="${DATASETS[$DS]}"
    OUTPUT="${MODEL}_differ_${DS}.json"
    EVAL_OUTPUT="${MODEL}_differ_${DS}_eval.json"

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 || true

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 \
        --continue-from "$OUTPUT" || true

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --eval "$OUTPUT" --output "$EVAL_OUTPUT" --workers $WORKERS_EVAL || true
done
