#!/bin/bash
# =============================================================================
# Run one API model over every BugSourceBench bug source, for one eval mode.
#
# Replaces the nine per-model scripts that used to live here (claude / gpt-5.2 /
# o4-mini x differ / testcases / solver-attacker); they were identical apart
# from the model id, the mode, and the eval worker count.
#
# Usage:
#   bash run_eval_all.sh <mode> <model>
#
#   mode   solver-attacker-style | solver-diff | solver-test-cases
#   model  any id run_eval.py accepts, e.g. gpt-5.2, o4-mini,
#          claude-sonnet-4-5-20250929, gemini-2.5-pro
#
# Examples:
#   bash run_eval_all.sh solver-attacker-style gpt-5.2
#   bash run_eval_all.sh solver-diff claude-sonnet-4-5-20250929
#
# Set INCLUDE_ADVERSARIAL=true to add cchoi1/bugbench_adversarial_new, which is
# not one of the four BugSourceBench sources.
# =============================================================================
set -euo pipefail

MODE="${1:-solver-attacker-style}"
MODEL="${2:-gpt-5.2}"

case "$MODE" in
  solver-attacker-style) TAG="solver_attacker"; WORKERS_EVAL=10 ;;
  solver-diff)           TAG="differ";          WORKERS_EVAL=10 ;;
  solver-test-cases)     TAG="testcases";       WORKERS_EVAL=18 ;;
  *) echo "unknown mode: $MODE (expected solver-attacker-style|solver-diff|solver-test-cases)" >&2; exit 2 ;;
esac

WORKERS="${WORKERS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-logs}"
mkdir -p "$OUTPUT_DIR"

# Keys become output filenames, so they name the actual bug source.
declare -A DATASETS=(
    ["human_authored"]="cchoi1/bugs_human_authored_eval"
    ["human_edited_lm"]="cchoi1/bugs_human_edited_lm_eval"
    ["qwen7b"]="cchoi1/bugs_lm_qwen7b_eval"
    ["gpt_oss_20b"]="cchoi1/bugs_lm_gpt_oss_20b_eval"
)
if [[ "${INCLUDE_ADVERSARIAL:-false}" == "true" ]]; then
    DATASETS["adversarial"]="cchoi1/bugbench_adversarial_new"
fi

for DS in "${!DATASETS[@]}"; do
    INPUT="${DATASETS[$DS]}"
    OUTPUT="$OUTPUT_DIR/${MODEL}_${TAG}_${DS}.json"
    EVAL_OUTPUT="$OUTPUT_DIR/${MODEL}_${TAG}_${DS}_eval.json"

    echo "=== $MODEL | $MODE | $DS ($INPUT)"

    python -m unified_eval.run_eval --input "$INPUT" --mode "$MODE" --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers "$WORKERS" \
        --max-new-tokens "$MAX_NEW_TOKENS"

    # Second pass picks up anything the first pass dropped.
    python -m unified_eval.run_eval --input "$INPUT" --mode "$MODE" --mutation-col "buggy_solution" \
        --model "$MODEL" --output "$OUTPUT" --inference-only --workers "$WORKERS" \
        --max-new-tokens "$MAX_NEW_TOKENS" --continue-from "$OUTPUT"

    python -m unified_eval.run_eval --input "$INPUT" --mode "$MODE" --mutation-col "buggy_solution" \
        --model "$MODEL" --eval "$OUTPUT" --output "$EVAL_OUTPUT" --workers "$WORKERS_EVAL"
done
