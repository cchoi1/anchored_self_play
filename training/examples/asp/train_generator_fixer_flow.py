"""Training script for GeneratorFixerWorkflow (asp)."""
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import OmegaConf

from examples.asp.generator_fixer_flow import GeneratorFixerWorkflow
from examples.asp.data_utils import load_data, parse_val_dataset_specs, parse_target_dataset_specs
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def parse_target_dataset_specs_str(spec_str: str) -> List[Tuple[str, str]]:
    """
    Parse dataset specification string into list of (dataset_name, split) tuples.
    
    Formats supported:
      - "dataset1:split1,dataset2:split2"
      - "dataset1,dataset2" (uses 'train' as default split)
      - "[dataset1:split1,dataset2:split2]" (with brackets)
    """
    if not spec_str:
        return []
    
    # Remove brackets if present
    spec_str = spec_str.strip()
    if spec_str.startswith("[") and spec_str.endswith("]"):
        spec_str = spec_str[1:-1]
    
    specs: List[Tuple[str, str]] = []
    for item in spec_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = item.split(":", 1)
            specs.append((parts[0].strip(), parts[1].strip()))
        else:
            specs.append((item, "train"))
    
    return specs


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    start_time = time.time()
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure workflow config exists and enable workflow mode
    if not hasattr(config.rllm, "workflow"):
        config.rllm.workflow = OmegaConf.create({})
    config.rllm.workflow.use_workflow = True

    # Resolve workflow_args safely
    workflow_args_cfg = None
    if hasattr(config.rllm, "workflow") and hasattr(config.rllm.workflow, "workflow_args"):
        workflow_args_cfg = config.rllm.workflow.workflow_args

    # -----------------------------
    # Datasets
    # Options: bigcodebench, kodcode, deepcoder, bugbench, bugbench_human, bugbench_adversarial
    # train_split: "train" or "train_large" (train_large excludes fewer test examples)
    # -----------------------------
    dataset_name = getattr(workflow_args_cfg, "dataset_name", "bigcodebench") if workflow_args_cfg else "bigcodebench"
    train_split = str(getattr(workflow_args_cfg, "train_split", "train")) if workflow_args_cfg else "train"
    train_dataset = DatasetRegistry.load_dataset(dataset_name, train_split)
    print(f"Using dataset: {dataset_name}/{train_split}")

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

    if train_dataset is None:
        print(f"Failed to load {dataset_name} train dataset. Exiting.")
        print("Available datasets:", DatasetRegistry.get_dataset_names())
        return

    # -----------------------------
    # Optional: Limit training data size
    # Set n_train_tasks to limit the number of training examples (useful for ablations)
    # Set train_data_seed for reproducible subsampling
    # -----------------------------
    n_train_tasks: Optional[int] = (
        int(getattr(workflow_args_cfg, "n_train_tasks", 0)) if workflow_args_cfg else 0
    ) or None
    train_data_seed: int = int(getattr(workflow_args_cfg, "train_data_seed", 42)) if workflow_args_cfg else 42

    if n_train_tasks is not None:
        original_size = len(train_dataset)
        if n_train_tasks < original_size:
            subset_name = f"{dataset_name}_subset_{n_train_tasks}_seed{train_data_seed}"
            # Check if the subsampled dataset already exists
            existing_subset = DatasetRegistry.load_dataset(subset_name, "train")
            if existing_subset is not None:
                train_dataset = existing_subset
                print(f"Loaded existing subsampled dataset: {subset_name}/train ({len(train_dataset)} examples)")
            else:
                # Subsample training data with deterministic shuffling
                all_data = train_dataset.get_data()
                rng = random.Random(train_data_seed)
                indices = list(range(original_size))
                rng.shuffle(indices)
                subsampled_data = [all_data[i] for i in indices[:n_train_tasks]]
                # Register the subsampled dataset so it gets a verl data path
                train_dataset = DatasetRegistry.register_dataset(subset_name, subsampled_data, "train")
                print(f"Subsampled training data: {original_size} -> {n_train_tasks} examples (seed={train_data_seed})")
                print(f"Registered as: {subset_name}/train")
        else:
            print(f"n_train_tasks={n_train_tasks} >= dataset size {original_size}, using full dataset")

    # -----------------------------
    # Optional: concatenate target dataset(s) into training
    # Supports multiple datasets via target_datasets='[ds1:split1,ds2:split2]'
    # or single dataset via target_dataset_name (backward compat)
    # -----------------------------
    target_specs = parse_target_dataset_specs(workflow_args_cfg)
    combined_train_dataset_name: Optional[str] = (
        str(getattr(workflow_args_cfg, "combined_train_dataset_name", "")).strip() if workflow_args_cfg else ""
    ) or None

    # Optional: limit target dataset size after shuffling (uniform sampling across all target datasets)
    n_target_tasks: Optional[int] = (
        int(getattr(workflow_args_cfg, "n_target_tasks", 0)) if workflow_args_cfg else 0
    ) or None
    target_data_seed: int = int(getattr(workflow_args_cfg, "target_data_seed", 42)) if workflow_args_cfg else 42

    if target_specs:
        main_tasks = train_dataset.get_data()
        combined_tasks: List[Dict[str, Any]] = list(main_tasks)
        target_names_for_label: List[str] = []

        # Load all target datasets first
        target_datasets_loaded: List[Tuple[str, str, List[Dict[str, Any]]]] = []
        total_target_size = 0
        for target_ds_name, target_split in target_specs:
            target_ds = DatasetRegistry.load_dataset(target_ds_name, target_split)
            if target_ds is None:
                raise ValueError(
                    f"target dataset not found: {target_ds_name!r} split={target_split!r}"
                )
            target_tasks = list(target_ds.get_data())
            target_datasets_loaded.append((target_ds_name, target_split, target_tasks))
            total_target_size += len(target_tasks)
            target_names_for_label.append(f"{target_ds_name}/{target_split}")
            print(f"  + {target_ds_name}/{target_split}: {len(target_tasks)} examples")

        # Sample from each dataset proportionally if n_target_tasks is specified
        all_target_tasks: List[Dict[str, Any]] = []
        if n_target_tasks is not None and n_target_tasks < total_target_size:
            rng = random.Random(target_data_seed)
            n_datasets = len(target_datasets_loaded)
            per_dataset_base = n_target_tasks // n_datasets
            remainder = n_target_tasks % n_datasets

            for i, (ds_name, ds_split, tasks) in enumerate(target_datasets_loaded):
                # Distribute remainder to first few datasets
                n_samples = per_dataset_base + (1 if i < remainder else 0)
                n_samples = min(n_samples, len(tasks))  # Don't sample more than available

                # Shuffle and sample from this dataset
                indices = list(range(len(tasks)))
                rng.shuffle(indices)
                sampled_tasks = [tasks[idx] for idx in indices[:n_samples]]
                all_target_tasks.extend(sampled_tasks)
                print(f"    Sampled {n_samples} from {ds_name}/{ds_split}")

            print(f"Subsampled target data: {total_target_size} -> {len(all_target_tasks)} examples "
                  f"({n_target_tasks} requested, {per_dataset_base}+remainder per dataset, seed={target_data_seed})")
        else:
            # Use all target tasks
            for ds_name, ds_split, tasks in target_datasets_loaded:
                all_target_tasks.extend(tasks)
            if n_target_tasks is not None:
                print(f"n_target_tasks={n_target_tasks} >= target size {total_target_size}, using full target dataset")

        combined_tasks.extend(all_target_tasks)

        # Build combined dataset name
        if combined_train_dataset_name:
            name = combined_train_dataset_name
        else:
            target_suffix = "_plus_".join(ds_name for ds_name, _ in target_specs)
            name = f"{dataset_name}_plus_{target_suffix}"
            if n_target_tasks is not None and n_target_tasks < total_target_size:
                name = f"{name}_n{n_target_tasks}"

        print(
            f"Combining training data: {dataset_name} ({len(main_tasks)}) + "
            f"{', '.join(target_names_for_label)} ({len(all_target_tasks)} after subsampling) "
            f"= {len(combined_tasks)} total -> registering '{name}'"
        )
        train_dataset = DatasetRegistry.register_dataset(name, combined_tasks, "train")

    # -----------------------------
    # Prompts / eval flags
    # -----------------------------
    generator_system_prompt = getattr(workflow_args_cfg, "generator_system_prompt", None) if workflow_args_cfg else None
    fixer_system_prompt = getattr(workflow_args_cfg, "fixer_system_prompt", None) if workflow_args_cfg else None
    evaluate_codegen = bool(getattr(workflow_args_cfg, "evaluate_codegen", True)) if workflow_args_cfg else True

    # -----------------------------
    # SSR-like self-play knobs
    # -----------------------------
    fixer_attempts_train = int(getattr(workflow_args_cfg, "fixer_attempts_train", 8)) if workflow_args_cfg else 8
    fixer_attempts_val = int(getattr(workflow_args_cfg, "fixer_attempts_val", 1)) if workflow_args_cfg else 1

    generator_reward_mode = str(getattr(workflow_args_cfg, "generator_reward_mode", "band")) if workflow_args_cfg else "band"
    solve_rate_band_low = float(getattr(workflow_args_cfg, "solve_rate_band_low", 0.05)) if workflow_args_cfg else 0.05
    solve_rate_band_high = float(getattr(workflow_args_cfg, "solve_rate_band_high", 0.25)) if workflow_args_cfg else 0.25
    gen_alpha_extreme = float(getattr(workflow_args_cfg, "gen_alpha_extreme", 0.2)) if workflow_args_cfg else 0.2
    gen_invalid_bug_reward = float(getattr(workflow_args_cfg, "gen_invalid_bug_reward", -1.0)) if workflow_args_cfg else -1.0

    fixer_reward_pm1 = bool(getattr(workflow_args_cfg, "fixer_reward_pm1", False)) if workflow_args_cfg else False

    use_pregenerated_bugs_in_validation = bool(
        getattr(workflow_args_cfg, "use_pregenerated_bugs_in_validation", True)
    ) if workflow_args_cfg else True

    use_pregenerated_bugs_in_training = bool(
        getattr(workflow_args_cfg, "use_pregenerated_bugs_in_training", bool(target_specs))
    ) if workflow_args_cfg else bool(target_specs)
    pregenerated_bug_train_probability = float(
        getattr(workflow_args_cfg, "pregenerated_bug_train_probability", 1.0)
    ) if workflow_args_cfg else 1.0

    episode_success_mode = str(getattr(workflow_args_cfg, "episode_success_mode", "bugfix")) if workflow_args_cfg else "bugfix"

    # Include failed test output in fixer prompts
    include_failed_test_output = bool(
        getattr(workflow_args_cfg, "include_failed_test_output", True)
    ) if workflow_args_cfg else True

    # Train generator on reference/pregenerated-bug episodes
    train_generator_on_reference_episodes = bool(
        getattr(workflow_args_cfg, "train_generator_on_reference_episodes", False)
    ) if workflow_args_cfg else False

    # -----------------------------
    # Generator example bugs (few-shot prompting)
    # -----------------------------
    generator_n_example_bugs = int(
        getattr(workflow_args_cfg, "generator_n_example_bugs", 3)
    ) if workflow_args_cfg else 3
    
    generator_example_bugs_dataset = (
        str(getattr(workflow_args_cfg, "generator_example_bugs_dataset", "")).strip()
        if workflow_args_cfg else ""
    ) or None
    generator_example_bugs_split = str(
        getattr(workflow_args_cfg, "generator_example_bugs_split", "train")
    ) if workflow_args_cfg else "train"
    
    # Load example bugs from dataset if specified
    generator_example_bugs_from_tasks = None
    if generator_example_bugs_dataset:
        example_ds = DatasetRegistry.load_dataset(generator_example_bugs_dataset, generator_example_bugs_split)
        if example_ds is not None:
            generator_example_bugs_from_tasks = list(example_ds.get_data())
            print(
                f"Loaded {len(generator_example_bugs_from_tasks)} tasks from "
                f"{generator_example_bugs_dataset}:{generator_example_bugs_split} for generator example bugs"
            )
        else:
            print(
                f"WARNING: Could not load generator_example_bugs_dataset="
                f"{generator_example_bugs_dataset}:{generator_example_bugs_split}"
            )

    # -----------------------------
    # Role-conditioned advantage options
    # -----------------------------
    freeze_generator = bool(getattr(workflow_args_cfg, "freeze_generator", False)) if workflow_args_cfg else False
    freeze_fixer = bool(getattr(workflow_args_cfg, "freeze_fixer", False)) if workflow_args_cfg else False
    use_role_advnorm = bool(getattr(workflow_args_cfg, "use_role_advnorm", False)) if workflow_args_cfg else False

    frozen_roles = []
    if freeze_generator:
        frozen_roles.append("generator")
    if freeze_fixer:
        # NOTE: bug_fixer* trajectories map to "solver" role in agent_workflow_engine.py
        # So we use "solver" here to match the role mapping
        frozen_roles.append("solver")

    # -----------------------------
    # Code embedding similarity (optional)
    # -----------------------------
    use_code_embedding_similarity = bool(getattr(workflow_args_cfg, "use_code_embedding_similarity", False)) if workflow_args_cfg else False
    code_embedding_reward_weight = float(getattr(workflow_args_cfg, "code_embedding_reward_weight", 0.3)) if workflow_args_cfg else 0.3
    code_embedding_model_name = str(getattr(workflow_args_cfg, "code_embedding_model_name", "voyage-code-3")) if workflow_args_cfg else "voyage-code-3"
    code_embedding_embed_mode = str(getattr(workflow_args_cfg, "code_embedding_embed_mode", "buggy")) if workflow_args_cfg else "buggy"
    code_embedding_include_problem = bool(getattr(workflow_args_cfg, "code_embedding_include_problem", False)) if workflow_args_cfg else False
    code_embedding_top_k = int(getattr(workflow_args_cfg, "code_embedding_top_k", 5)) if workflow_args_cfg else 5
    code_embedding_use_margin = bool(getattr(workflow_args_cfg, "code_embedding_use_margin", True)) if workflow_args_cfg else True
    code_embedding_margin_temperature = float(getattr(workflow_args_cfg, "code_embedding_margin_temperature", 10.0)) if workflow_args_cfg else 10.0
    
    # Pool paths (if pre-computed)
    code_embedding_target_pool_path = (
        str(getattr(workflow_args_cfg, "code_embedding_target_pool_path", "")).strip() if workflow_args_cfg else ""
    ) or None
    code_embedding_negative_pool_path = (
        str(getattr(workflow_args_cfg, "code_embedding_negative_pool_path", "")).strip() if workflow_args_cfg else ""
    ) or None
    
    # Dataset specs for building pools (if no pool paths provided)
    # Format: "dataset1:split1,dataset2:split2" or just use target_datasets
    code_embedding_target_datasets = (
        str(getattr(workflow_args_cfg, "code_embedding_target_datasets", "")).strip() if workflow_args_cfg else ""
    ) or None
    code_embedding_negative_datasets = (
        str(getattr(workflow_args_cfg, "code_embedding_negative_datasets", "")).strip() if workflow_args_cfg else ""
    ) or None
    
    # Build reference bugs from datasets if needed
    code_embedding_reference_bugs: Optional[List[Dict[str, Any]]] = None
    code_embedding_negative_bugs: Optional[List[Dict[str, Any]]] = None
    
    if use_code_embedding_similarity:
        # Build target pool reference bugs
        if not code_embedding_target_pool_path:
            # Use code_embedding_target_datasets if specified, otherwise use target_specs
            if code_embedding_target_datasets:
                target_ds_specs = parse_target_dataset_specs_str(code_embedding_target_datasets)
            elif target_specs:
                # Reuse the target datasets already loaded for training
                target_ds_specs = target_specs
            else:
                target_ds_specs = []
            
            if target_ds_specs:
                code_embedding_reference_bugs = []
                for ds_name, ds_split in target_ds_specs:
                    ds = DatasetRegistry.load_dataset(ds_name, ds_split)
                    if ds is not None:
                        tasks = list(ds.get_data())
                        code_embedding_reference_bugs.extend(tasks)
                        print(f"  [CodeEmbed] Loaded {len(tasks)} tasks from {ds_name}:{ds_split} for target pool")
                if code_embedding_reference_bugs:
                    print(f"  [CodeEmbed] Total target pool reference bugs: {len(code_embedding_reference_bugs)}")
                else:
                    print("  [CodeEmbed] WARNING: No reference bugs loaded for target pool")
        
        # Build negative pool reference bugs
        if not code_embedding_negative_pool_path and code_embedding_negative_datasets:
            neg_ds_specs = parse_target_dataset_specs_str(code_embedding_negative_datasets)
            if neg_ds_specs:
                code_embedding_negative_bugs = []
                for ds_name, ds_split in neg_ds_specs:
                    ds = DatasetRegistry.load_dataset(ds_name, ds_split)
                    if ds is not None:
                        tasks = list(ds.get_data())
                        code_embedding_negative_bugs.extend(tasks)
                        print(f"  [CodeEmbed] Loaded {len(tasks)} tasks from {ds_name}:{ds_split} for negative pool")
                if code_embedding_negative_bugs:
                    print(f"  [CodeEmbed] Total negative pool reference bugs: {len(code_embedding_negative_bugs)}")

    trainer = AgentTrainer(
        workflow_class=GeneratorFixerWorkflow,
        workflow_args={
            "reward_function": code_reward_fn,
            "generator_system_prompt": generator_system_prompt,
            "fixer_system_prompt": fixer_system_prompt,
            "evaluate_codegen": evaluate_codegen,
            "fixer_attempts_train": fixer_attempts_train,
            "fixer_attempts_val": fixer_attempts_val,
            "generator_reward_mode": generator_reward_mode,
            "solve_rate_band_low": solve_rate_band_low,
            "solve_rate_band_high": solve_rate_band_high,
            "gen_alpha_extreme": gen_alpha_extreme,
            "gen_invalid_bug_reward": gen_invalid_bug_reward,
            "fixer_reward_pm1": fixer_reward_pm1,
            "use_pregenerated_bugs_in_validation": use_pregenerated_bugs_in_validation,
            "use_pregenerated_bugs_in_training": use_pregenerated_bugs_in_training,
            "pregenerated_bug_train_probability": pregenerated_bug_train_probability,
            "episode_success_mode": episode_success_mode,
            "include_failed_test_output": include_failed_test_output,
            "train_generator_on_reference_episodes": train_generator_on_reference_episodes,
            # Generator example bugs (few-shot prompting)
            "generator_n_example_bugs": generator_n_example_bugs,
            "generator_example_bugs_from_tasks": generator_example_bugs_from_tasks,
            # Role-conditioned advantage options
            "freeze_cm": bool(frozen_roles),
            "cm_roles": frozen_roles,
            "use_role_advnorm": use_role_advnorm,
            # Code embedding similarity
            "use_code_embedding_similarity": use_code_embedding_similarity,
            "code_embedding_reward_weight": code_embedding_reward_weight,
            "code_embedding_model_name": code_embedding_model_name,
            "code_embedding_embed_mode": code_embedding_embed_mode,
            "code_embedding_include_problem": code_embedding_include_problem,
            "code_embedding_top_k": code_embedding_top_k,
            "code_embedding_use_margin": code_embedding_use_margin,
            "code_embedding_margin_temperature": code_embedding_margin_temperature,
            "code_embedding_target_pool_path": code_embedding_target_pool_path,
            "code_embedding_negative_pool_path": code_embedding_negative_pool_path,
            "code_embedding_reference_bugs": code_embedding_reference_bugs,
            "code_embedding_negative_bugs": code_embedding_negative_bugs,
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