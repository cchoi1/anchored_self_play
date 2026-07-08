#!/bin/bash

set -e
mkdir -p logs

set -euo pipefail

MODEL="gemini-2.5-pro"
WORKERS=22
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
bash gemini-2.5-pro_coder_eval.sh
