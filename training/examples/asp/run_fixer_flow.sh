#!/bin/bash
# Example commands for running fixer_flow evaluation

python examples/asp/run_fixer_flow.py \
    --val_datasets bugs_human_authored:test bugs_human_edited_lm:test bugs_lm_qwen7b:test bugs_lm_gpt_oss_20b:test bugbench_adversarial:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --synthesizer_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --synthesizer_base_url http://localhost:30001/v1 \
    --n_parallel 32 \
    --only_train_on_failures \
    --eval_pregenerated_only \
    --include_failed_test_output \
    --save_results \
    --output_dir logs

python examples/asp/run_fixer_flow.py \
    --val_datasets bugs_human_authored:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --synthesizer_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --synthesizer_base_url http://localhost:30001/v1 \
    --n_parallel 64 \
    --only_train_on_failures \
    --eval_pregenerated_only \
    --include_failed_test_output \
    --save_results \
    --output_dir logs

python examples/asp/run_fixer_flow.py \
    --val_datasets bugs_lm_qwen7b:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --synthesizer_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --synthesizer_base_url http://localhost:30001/v1 \
    --n_parallel 64 \
    --only_train_on_failures \
    --eval_pregenerated_only \
    --include_failed_test_output \
    --save_results \
    --output_dir logs

# One-shot mode (always run fixer, don't check synthesizer failures)
python examples/asp/run_fixer_flow.py \
    --dataset bugs_human_authored \
    --split test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --synthesizer_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --synthesizer_base_url http://localhost:30001/v1 \
    --n_parallel 32 \
    --include_failed_test_output \
    --save_results \
    --output_dir logs

# Failure-only mode (only train on synthesizer failures)
python examples/asp/run_fixer_flow.py \
    --dataset bugs_human_authored \
    --split test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --synthesizer_model gpt-4o-mini \
    --synthesizer_base_url https://api.openai.com/v1 \
    --n_parallel 32 \
    --only_train_on_failures \
    --include_failed_test_output \
    --save_results \
    --output_dir logs
