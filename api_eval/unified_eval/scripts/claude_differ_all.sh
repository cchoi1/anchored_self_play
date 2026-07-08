#!/bin/bash

set -e
mkdir -p logs
set -euo pipefail


MODEL="claude-sonnet-4-5-20250929"
MODE="solver-diff"
WORKERS=2
WORKERS_EVAL=10

for DS in human qwen adv oss; do
    INPUT="cchoi1/bugbench_${DS}"
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
