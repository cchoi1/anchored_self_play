#!/bin/bash
# Example commands for running generator_fixer_flow evaluation.
#
# These are independent examples, not a single pipeline -- run the one you
# want rather than executing the whole file. Each expects an OpenAI-compatible
# server for --base_url, and datasets registered via:
#     python -m examples.asp.prepare_data

python examples/asp/run_generator_fixer_flow.py \
    --val_datasets bugs_human_authored:test bugs_human_edited_lm:test bugs_lm_qwen7b:test bugs_lm_gpt_oss_20b:test bugbench_adversarial:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --n_parallel 128 \
    --eval_pregenerated_only \
    --evaluate_codegen \
    --include_failed_test_output \
    --fixer_attempts_val 1 \
    --save_results \
    --output_dir logs

python examples/asp/run_generator_fixer_flow.py \
    --val_datasets bugs_lm_qwen7b:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --n_parallel 64 \
    --eval_pregenerated_only \
    --include_failed_test_output \
    --fixer_attempts_val 1

python examples/asp/run_generator_fixer_flow.py \
    --val_datasets bugs_lm_qwen7b:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --n_parallel 64 \
    --eval_pregenerated_only \
    --evaluate_codegen \
    --include_failed_test_output \
    --fixer_attempts_val 1 \
    --save_results \
    --output_dir logs

# Embedding-similarity ("anchoring") reward reported alongside the fix rate.
# Build the reference pool once, then reuse it across runs.
python examples/asp/run_generator_fixer_flow.py \
    --dataset bugs_human_authored \
    --split test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --n_parallel 32 \
    --use_code_embedding_similarity \
    --code_embedding_reward_weight 0.2 \
    --code_embedding_model_name voyage-code-3 \
    --reference_bug_dataset bugs_human_edited_lm \
    --reference_bug_split train \
    --evaluate_codegen \
    --include_failed_test_output \
    --fixer_attempts_val 1 \
    --save_results \
    --output_dir logs
