#!/bin/bash

set -e
mkdir -p logs
set -euo pipefail


INPUT="cchoi1/bugbench_og"
WORKERS=2
WORKERS_EVAL=18

# Models to evaluate
MODELS=("gpt-5.2" "o4-mini" "claude-sonnet-4-5-20250929")

# Modes: solver-attacker-style, solver-diff, solver-test-cases
declare -A MODE_NAMES=(
    ["solver-attacker-style"]="solver_attacker"
    ["solver-diff"]="differ"
    ["solver-test-cases"]="testcases"
)

for MODEL in "${MODELS[@]}"; do
    for MODE in solver-attacker-style solver-diff solver-test-cases; do
        MODE_SHORT="${MODE_NAMES[$MODE]}"
        OUTPUT="${MODEL}_${MODE_SHORT}_og.json"
        EVAL_OUTPUT="${MODEL}_${MODE_SHORT}_og_eval.json"

        echo "=============================================="
        echo "Running $MODEL - $MODE on bugbench_og"
        echo "=============================================="

        # Inference
        python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
            --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 || true

        # Retry pass
        python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
            --model "$MODEL" --output "$OUTPUT" --inference-only --workers $WORKERS --max-new-tokens 10000 \
            --continue-from "$OUTPUT" || true

        # Eval
        python -m unified_eval.run_eval --input "$INPUT" --mode $MODE --mutation-col "buggy_solution" \
            --model "$MODEL" --eval "$OUTPUT" --output "$EVAL_OUTPUT" --workers $WORKERS_EVAL || true
    done
done

echo "All evaluations complete!"
