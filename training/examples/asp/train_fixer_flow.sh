#!/bin/bash
# =============================================================================
# Fixer-only baseline: train the fixer against a *frozen* bug generator.
#
# Instead of learning to generate bugs, buggy programs come from a static
# "synthesizer" model (any OpenAI-compatible endpoint, e.g. a vLLM server).
# The policy is trained only to repair them. This is the non-self-play baseline.
#
# Usage:
#   SYNTHESIZER_BASE_URL=http://localhost:32000/v1 \
#   bash examples/asp/train_fixer_flow.sh
#
# Environment variables: MODEL_PATH, NUM_GPUS, RUN_DIR, WANDB_PROJECT/WANDB_NAME,
#   SYNTHESIZER_MODEL, SYNTHESIZER_BASE_URL.
# =============================================================================
set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1

RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
cd "$RLLM_DIR"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_NAME="${RUN_NAME:-fixer-baseline}"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
RUN_DIR="${RUN_DIR:-$RLLM_DIR/runs/${RUN_NAME}_${TIMESTAMP}}"
mkdir -p "$RUN_DIR"

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  LOGGER="['console','wandb']"; export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"; export WANDB_DIR="$RUN_DIR"
else
  LOGGER="['console']"
fi

# Frozen synthesizer (static bug generator). Point at any OpenAI-compatible
# endpoint; leave SYNTHESIZER_MODEL empty to reuse the fixer's own model.
SYNTHESIZER_MODEL="${SYNTHESIZER_MODEL:-openai/gpt-oss-20b}"
SYNTHESIZER_BASE_URL="${SYNTHESIZER_BASE_URL:-http://localhost:32000/v1}"
SYNTHESIZER_TEMPERATURE=0.6
SYNTHESIZER_TOP_P=0.95

# Fixer training options
ONLY_TRAIN_ON_FAILURES=true
REWARD_PM1=false
INCLUDE_FAILED_TEST_OUTPUT=true
MAX_FAILED_TEST_OUTPUT_CHARS=4000

python3 -m examples.asp.train_fixer_flow \
    hydra.run.dir="$RUN_DIR" \
    +trainer.save_dir="$RUN_DIR/checkpoints/" \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=per_step \
    rllm.stepwise_advantage.normalize_by_steps=True \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=128 \
    ++rllm.workflow.workflow_args.dataset_name=bigcodebench \
    +rllm.workflow.workflow_args.synthesizer_model="$SYNTHESIZER_MODEL" \
    +rllm.workflow.workflow_args.synthesizer_base_url="$SYNTHESIZER_BASE_URL" \
    +rllm.workflow.workflow_args.synthesizer_temperature=$SYNTHESIZER_TEMPERATURE \
    +rllm.workflow.workflow_args.synthesizer_top_p=$SYNTHESIZER_TOP_P \
    +rllm.workflow.workflow_args.only_train_on_failures=$ONLY_TRAIN_ON_FAILURES \
    +rllm.workflow.workflow_args.reward_pm1=$REWARD_PM1 \
    +rllm.workflow.workflow_args.include_failed_test_output=$INCLUDE_FAILED_TEST_OUTPUT \
    +rllm.workflow.workflow_args.max_failed_test_output_chars=$MAX_FAILED_TEST_OUTPUT_CHARS \
    +rllm.workflow.workflow_args.val_datasets='[bugs_human_edited_lm:test,bugs_lm_qwen7b:test,bugs_lm_gpt_oss_20b:test]' \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name="${WANDB_PROJECT:-anchored-self-play}" \
    trainer.experiment_name=$RUN_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10
