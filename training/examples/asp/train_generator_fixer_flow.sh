#!/bin/bash
# =============================================================================
# Generator--Fixer self-play training for code repair.
#
# A single policy is trained to (1) generate bugs and (2) fix them. This script
# covers all methods from the paper via the toggles below:
#
#   * Standard self-play : USE_CODE_EMBED_SIM=false, USE_PREGEN_BUGS_TRAIN=false
#   * + Embedding reward  : USE_CODE_EMBED_SIM=true   (anchors bug *generation*)
#   * + Reference mixing  : USE_PREGEN_BUGS_TRAIN=true (anchors bug *fixing*)
#   * Anchored (ours)     : both of the above enabled  <-- default
#
# Usage:
#   bash examples/asp/train_generator_fixer_flow.sh
#
# Configure the machine / model / logging with environment variables:
#   MODEL_PATH   HuggingFace model id or local path (default Qwen2.5-Coder-7B-Instruct)
#   NUM_GPUS     GPUs on this node (default 8)
#   RUN_DIR      Output/run directory (default ./runs/<run_name>_<timestamp>)
#   WANDB_PROJECT / WANDB_NAME  Set to enable Weights & Biases logging.
# =============================================================================
set -euo pipefail

# --- vLLM / torch runtime env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HYDRA_FULL_ERROR=1

# --- Run from the repo root (directory that contains the `rllm` package)
RLLM_DIR=$(python3 -c "import rllm, os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
cd "$RLLM_DIR"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_NAME="${RUN_NAME:-generator-fixer-selfplay}"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
RUN_DIR="${RUN_DIR:-$RLLM_DIR/runs/${RUN_NAME}_${TIMESTAMP}}"
mkdir -p "$RUN_DIR"

# Logger: console by default; add wandb automatically if WANDB_PROJECT is set.
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  LOGGER="['console','wandb']"
  export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
  export WANDB_DIR="$RUN_DIR"
else
  LOGGER="['console']"
fi

# -----------------------------
# Datasets
# Base code-generation tasks to synthesize bugs from, plus the reference
# ("target") bug source that anchoring pulls toward. Replace the dataset names
# with the released BugSourceBench splits.
# -----------------------------
DATASET_NAME="${DATASET_NAME:-bigcodebench}"
TARGET_DATASET_NAME="${TARGET_DATASET_NAME:-bugs_human_edited_lm}"
TARGET_DATASET_SPLIT="${TARGET_DATASET_SPLIT:-train}"

# -----------------------------
# Difficulty / reward shaping (generator + fixer)
# -----------------------------
FIXER_ATTEMPTS_TRAIN=4
FIXER_ATTEMPTS_VAL=1
GEN_REWARD_MODE="band"          # band|smooth|binary
SOLVE_RATE_BAND_LOW=0.25
SOLVE_RATE_BAND_HIGH=0.75
GEN_ALPHA_EXTREME=0.2
GEN_INVALID_BUG_REWARD=-1.0
FIXER_REWARD_PM1=false          # true => {-1,+1}, false => {0,1}
USE_PREGEN_BUGS_VAL=true
EPISODE_SUCCESS_MODE="bugfix"   # bugfix|codegen
INCLUDE_FAILED_TEST_OUTPUT=true

# Role-conditioned advantage normalization
FREEZE_GENERATOR=false
FREEZE_FIXER=false
USE_ROLE_ADVNORM=true

# -----------------------------
# Anchoring (i): embedding-similarity reward on the *generator*
# -----------------------------
USE_CODE_EMBED_SIM=true
EMBED_REWARD_WEIGHT=0.20
EMBED_MODEL_NAME="voyage-code-3"   # requires VOYAGE_API_KEY
EMBED_TOPK=5
EMBED_INCLUDE_PROBLEM=false
EMBED_MODE="diff"                  # diff|buggy
USE_MARGIN=true
MARGIN_TEMPERATURE=5.0
# Pre-computed embedding pools (optional but recommended). Build with:
#   python -m examples.asp.run_generator_fixer_flow --save_code_embedding_target_pool <path>
TARGET_POOL_PATH="${TARGET_POOL_PATH:-}"
NEG_POOL_PATH="${NEG_POOL_PATH:-}"

# -----------------------------
# Anchoring (ii): mix reference bugs into *fixer* training
# -----------------------------
USE_PREGEN_BUGS_TRAIN=true
PREGEN_BUG_TRAIN_PROB=0.5

# --- Assemble optional args
EXTRA_ARGS=()
[[ -n "$TARGET_DATASET_NAME" ]] && EXTRA_ARGS+=(
  ++rllm.workflow.workflow_args.target_dataset_name="$TARGET_DATASET_NAME"
  ++rllm.workflow.workflow_args.target_dataset_split="$TARGET_DATASET_SPLIT")
if [[ "$USE_CODE_EMBED_SIM" == "true" ]]; then
  EXTRA_ARGS+=(
    +rllm.workflow.workflow_args.use_code_embedding_similarity=$USE_CODE_EMBED_SIM
    +rllm.workflow.workflow_args.code_embedding_reward_weight=$EMBED_REWARD_WEIGHT
    +rllm.workflow.workflow_args.code_embedding_model_name=$EMBED_MODEL_NAME
    +rllm.workflow.workflow_args.code_embedding_top_k=$EMBED_TOPK
    +rllm.workflow.workflow_args.code_embedding_include_problem=$EMBED_INCLUDE_PROBLEM
    +rllm.workflow.workflow_args.code_embedding_embed_mode=$EMBED_MODE
    +rllm.workflow.workflow_args.code_embedding_use_margin=$USE_MARGIN
    +rllm.workflow.workflow_args.code_embedding_margin_temperature=$MARGIN_TEMPERATURE)
  [[ -n "$TARGET_POOL_PATH" ]] && EXTRA_ARGS+=(+rllm.workflow.workflow_args.code_embedding_target_pool_path="$TARGET_POOL_PATH")
  [[ -n "$NEG_POOL_PATH" ]] && EXTRA_ARGS+=(+rllm.workflow.workflow_args.code_embedding_negative_pool_path="$NEG_POOL_PATH")
fi
if [[ "$USE_PREGEN_BUGS_TRAIN" == "true" ]]; then
  EXTRA_ARGS+=(
    +rllm.workflow.workflow_args.use_pregenerated_bugs_in_training=$USE_PREGEN_BUGS_TRAIN
    +rllm.workflow.workflow_args.pregenerated_bug_train_probability=$PREGEN_BUG_TRAIN_PROB)
fi

python3 -m examples.asp.train_generator_fixer_flow \
    hydra.run.dir="$RUN_DIR" \
    +trainer.save_dir="$RUN_DIR/checkpoints/" \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=per_step \
    rllm.stepwise_advantage.normalize_by_steps=True \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=128 \
    ++rllm.workflow.workflow_args.dataset_name=$DATASET_NAME \
    +rllm.workflow.workflow_args.fixer_attempts_train=$FIXER_ATTEMPTS_TRAIN \
    +rllm.workflow.workflow_args.fixer_attempts_val=$FIXER_ATTEMPTS_VAL \
    +rllm.workflow.workflow_args.generator_reward_mode=$GEN_REWARD_MODE \
    +rllm.workflow.workflow_args.solve_rate_band_low=$SOLVE_RATE_BAND_LOW \
    +rllm.workflow.workflow_args.solve_rate_band_high=$SOLVE_RATE_BAND_HIGH \
    +rllm.workflow.workflow_args.gen_alpha_extreme=$GEN_ALPHA_EXTREME \
    +rllm.workflow.workflow_args.gen_invalid_bug_reward=$GEN_INVALID_BUG_REWARD \
    +rllm.workflow.workflow_args.fixer_reward_pm1=$FIXER_REWARD_PM1 \
    +rllm.workflow.workflow_args.use_pregenerated_bugs_in_validation=$USE_PREGEN_BUGS_VAL \
    +rllm.workflow.workflow_args.episode_success_mode=$EPISODE_SUCCESS_MODE \
    +rllm.workflow.workflow_args.include_failed_test_output=$INCLUDE_FAILED_TEST_OUTPUT \
    +rllm.workflow.workflow_args.freeze_generator=$FREEZE_GENERATOR \
    +rllm.workflow.workflow_args.freeze_fixer=$FREEZE_FIXER \
    ++rllm.workflow.workflow_args.use_role_advnorm=$USE_ROLE_ADVNORM \
    +rllm.workflow.workflow_args.val_datasets='[bugs_human_edited_lm:test,bugs_human_authored:test,bugs_lm_qwen7b:test]' \
    "${EXTRA_ARGS[@]}" \
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
    trainer.total_epochs=20
