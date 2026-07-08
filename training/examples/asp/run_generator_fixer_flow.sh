#!/bin/bash
# Example commands for running generator_fixer_flow evaluation

python examples/asp/run_generator_fixer_flow.py \
    --val_datasets bugbench:test bugbench_human:test bugbench_qwen7b_sampled:test bugbench_gpt-oss-20b_sampled:test bugbench_adversarial:test \
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
    --val_datasets bugbench_qwen7b_sampled:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --n_parallel 64 \
    --eval_pregenerated_only \
    --include_failed_test_output \
    --fixer_attempts_val 1

python examples/asp/run_generator_fixer_flow.py \
    --val_datasets bugbench_qwen7b_sampled:test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --n_parallel 64 \
    --eval_pregenerated_only \
    --evaluate_codegen \
    --include_failed_test_output \
    --fixer_attempts_val 1 \
    --save_results \
    --output_dir logs

# With LLM-as-judge bug similarity
python examples/asp/run_generator_fixer_flow.py \
    --dataset bugbench \
    --split test \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --base_url http://localhost:30000/v1 \
    --n_parallel 32 \
    --use_bug_similarity_judge \
    --bug_similarity_reward_weight 0.5 \
    --bug_similarity_n_targets 3 \
    --reference_bug_dataset bugbench \
    --reference_bug_split test \
    --evaluate_codegen \
    --include_failed_test_output \
    --fixer_attempts_val 1 \
    --save_results \
    --output_dir logs
