"""Training script for FrozenGeneratorFixerWorkflow (asp)."""
import time
from typing import Any

import hydra
from omegaconf import OmegaConf

from examples.asp.frozen_generator_fixer_flow import FrozenGeneratorFixerWorkflow
from examples.asp.data_utils import parse_val_dataset_specs
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    start_time = time.time()
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Enable workflow training.
    if not hasattr(config.rllm, "workflow"):
        config.rllm.workflow = OmegaConf.create({})
    config.rllm.workflow.use_workflow = True

    workflow_args_cfg = None
    if hasattr(config.rllm, "workflow") and hasattr(config.rllm.workflow, "workflow_args"):
        workflow_args_cfg = config.rllm.workflow.workflow_args

    # -----------------------------
    # Datasets
    # Options: bigcodebench, kodcode, deepcoder, bugbench, bugbench_human, bugbench_adversarial
    # -----------------------------
    dataset_name = getattr(workflow_args_cfg, "dataset_name", "bigcodebench") if workflow_args_cfg else "bigcodebench"
    train_split = getattr(workflow_args_cfg, "train_split", "train") if workflow_args_cfg else "train"
    train_dataset = DatasetRegistry.load_dataset(dataset_name, train_split)

    # Validation datasets (optional): evaluate separately (no concatenation).
    val_specs = parse_val_dataset_specs(workflow_args_cfg)
    if val_specs:
        val_datasets: dict[str, Any] = {}
        for alias, (ds_name, split) in val_specs.items():
            ds = DatasetRegistry.load_dataset(ds_name, split)
            if ds is None:
                raise ValueError(f"val dataset not found: {ds_name!r} split={split!r} (alias={alias!r})")
            val_datasets[alias] = ds
        val_dataset = None
        print(f"Using multiple val datasets: {list(val_datasets.keys())}")
    else:
        val_dataset = DatasetRegistry.load_dataset("bugbench_human", "test")
        val_datasets = None

    # -----------------------------
    # Frozen generator knobs (static bug generator)
    # -----------------------------
    generator_model = getattr(workflow_args_cfg, "generator_model", None) if workflow_args_cfg else None
    generator_base_url = getattr(workflow_args_cfg, "generator_base_url", None) if workflow_args_cfg else None
    generator_api_key = getattr(workflow_args_cfg, "generator_api_key", None) if workflow_args_cfg else None
    generator_temperature = getattr(workflow_args_cfg, "generator_temperature", 0.6) if workflow_args_cfg else 0.6
    generator_top_p = getattr(workflow_args_cfg, "generator_top_p", 0.95) if workflow_args_cfg else 0.95
    generator_system_prompt = getattr(workflow_args_cfg, "generator_system_prompt", None) if workflow_args_cfg else None

    # -----------------------------
    # Fixer knobs
    # -----------------------------
    fixer_system_prompt = getattr(workflow_args_cfg, "fixer_system_prompt", None) if workflow_args_cfg else None
    fixer_reward_pm1 = bool(getattr(workflow_args_cfg, "fixer_reward_pm1", False) if workflow_args_cfg else False)
    include_failed_test_output = bool(getattr(workflow_args_cfg, "include_failed_test_output", True) if workflow_args_cfg else True)
    use_pregenerated_bugs_in_validation = bool(getattr(workflow_args_cfg, "use_pregenerated_bugs_in_validation", True) if workflow_args_cfg else True)
    evaluate_codegen = bool(getattr(workflow_args_cfg, "evaluate_codegen", True) if workflow_args_cfg else True)

    trainer = AgentTrainer(
        workflow_class=FrozenGeneratorFixerWorkflow,
        workflow_args={
            "reward_function": code_reward_fn,
            "generator_model": generator_model,
            "generator_base_url": generator_base_url,
            "generator_api_key": generator_api_key,
            "generator_temperature": generator_temperature,
            "generator_top_p": generator_top_p,
            "generator_system_prompt": generator_system_prompt,
            "fixer_system_prompt": fixer_system_prompt,
            "fixer_reward_pm1": fixer_reward_pm1,
            "include_failed_test_output": include_failed_test_output,
            "use_pregenerated_bugs_in_validation": use_pregenerated_bugs_in_validation,
            "evaluate_codegen": evaluate_codegen,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_datasets=val_datasets,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total run time: {format_time(total_time)} ({total_time:.2f} seconds)")
        print("=" * 60)


if __name__ == "__main__":
    main()