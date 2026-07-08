#!/bin/bash

set -e
mkdir -p logs

set -euo pipefail

MODEL="claude-sonnet-4-5-20250929"
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
    --workers "$WORKERS"

## CODER INFERENCE (retry pass)
python -m unified_eval.run_eval \
    --input "$INPUT" \
    --mode coder-complete \
    --mutation-col "response" \
    --model "$MODEL" \
    --output "$CODER_OUTPUT" \
    --inference-only \
    --workers "$WORKERS" \
    --continue-from "$CODER_OUTPUT"

## Submit eval job
bash claude-sonnet_coder_eval.sh
