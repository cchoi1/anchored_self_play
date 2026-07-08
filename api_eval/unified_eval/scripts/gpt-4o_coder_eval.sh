#!/bin/bash

set -e
mkdir -p logs

set -euo pipefail

MODEL="gpt-4o"
WORKERS_EVAL=10
INPUT="cchoi1/adv_bugbench"
CODER_OUTPUT="${MODEL}_coder.json"
CODER_EVAL_OUTPUT="${MODEL}_coder_eval.json"


## CODER EVAL
python -m unified_eval.run_eval \
    --input "$INPUT" \
    --mode coder-complete \
    --mutation-col "response" \
    --model "$MODEL" \
    --eval "$CODER_OUTPUT" \
    --output "$CODER_EVAL_OUTPUT" \
    --workers "$WORKERS_EVAL"

