#!/bin/bash

set -e
mkdir -p logs
set -euo pipefail


INPUT="cchoi1/bugbench_og"
MODEL="claude-sonnet-4-5-20250929"
WORKERS=2
WORKERS_EVAL=18

for MODE in solver-attacker-style solver-diff solver-test-cases; do
    case $MODE in
        solver-attacker-style) MODE_SHORT="solver_attacker" ;;
        solver-diff) MODE_SHORT="differ" ;;
        solver-test-cases) MODE_SHORT="testcases" ;;
    esac
    
    OUTPUT="${MODEL}_${MODE_SHORT}_og.json"
    EVAL_OUTPUT="${MODEL}_${MODE_SHORT}_og_eval.json"

    echo "Running $MODEL - $MODE"

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 || true

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 \
        --continue-from "$OUTPUT" || true

    python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
        --model "$MODEL" --eval "$OUTPUT" --output "$EVAL_OUTPUT" --workers $WORKERS_EVAL || true
done

echo "Done: $MODEL"
