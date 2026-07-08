#!/bin/bash

set -e
mkdir -p logs

set -euo pipefail

MODEL="o4-mini"
WORKERS=2
INPUT="cchoi1/adv_bugbench"
CODER_OUTPUT="${MODEL}_coder.json"


## CODER INFERENCE
python -m unified_eval.run_eval \
    --input "$INPUT" \
    --mode coder-complete \
    --mutation-col "response" \
    --model "$MODEL" \
    --output "$CODER_OUTPUT" \
    --inference-only \
    --workers "$WORKERS" \
    --max-new-tokens 10000

## CODER INFERENCE (retry pass)
python -m unified_eval.run_eval \
    --input "$INPUT" \
    --mode coder-complete \
    --mutation-col "response" \
    --model "$MODEL" \
    --output "$CODER_OUTPUT" \
    --inference-only \
    --workers "$WORKERS" \
    --max-new-tokens 10000 \
    --continue-from "$CODER_OUTPUT"

## Submit eval job
bash o4-mini_coder_eval.sh
