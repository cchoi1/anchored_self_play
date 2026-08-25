from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Dict, Optional, List

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import code_reward_fn


def process_task(example, idx):
    """Process example into the expected format for bug generator workflows."""
    return example

def load_data(
    dataset_name: str,
    split: str,
    n: int = 1,
) -> List[Dict]:
    """Load dataset using the Dataset interface."""
    
    if dataset_name.lower() == "deepcoder_chunked":
        dataset = DatasetRegistry.load_dataset("deepcoder_chunked", split)
        tasks = dataset.get_data()
        return tasks
    
    elif dataset_name.lower() == "deepcoder_bugs":
        dataset = DatasetRegistry.load_dataset("deepcoder_bugs", split)
        tasks = dataset.get_data()
        return tasks
    
    elif dataset_name.lower() == "deepcoder":
        dataset = DatasetRegistry.load_dataset("deepcoder", split)
        tasks = dataset.get_data()
        return tasks
    
    elif dataset_name.lower() == "bigcodebench":
        dataset = DatasetRegistry.load_dataset("bigcodebench", split)
        tasks = dataset.get_data()
        return tasks
    
    elif dataset_name.lower() == "bugs_human_authored":
        dataset = DatasetRegistry.load_dataset("bugs_human_authored", split)
        tasks = dataset.get_data()
        return tasks

    elif dataset_name.lower() == "bugs_human_edited_lm":
        dataset = DatasetRegistry.load_dataset("bugs_human_edited_lm", split)
        tasks = dataset.get_data()
        return tasks
    
    elif dataset_name.lower() == "bugbench_adversarial":
        dataset = DatasetRegistry.load_dataset("bugbench_adversarial", split)
        tasks = dataset.get_data()
        return tasks

    elif dataset_name.lower() == "bugs_lm_gpt_oss_20b":
        dataset = DatasetRegistry.load_dataset("bugs_lm_gpt_oss_20b", split)
        tasks = dataset.get_data()
        return tasks

    elif dataset_name.lower() == "bugs_lm_gpt_oss_120b":
        dataset = DatasetRegistry.load_dataset("bugs_lm_gpt_oss_120b", split)
        tasks = dataset.get_data()
        return tasks
    
    elif dataset_name.lower() == "bugs_lm_qwen7b":
        dataset = DatasetRegistry.load_dataset("bugs_lm_qwen7b", split)
        tasks = dataset.get_data()
        return tasks
    
    elif dataset_name.lower() == "lcb_bugbench":
        dataset = DatasetRegistry.load_dataset("lcb_bugbench", split)
        tasks = dataset.get_data()
        return tasks
    
    else:
        dataset = DatasetRegistry.load_dataset(dataset_name, split)
        tasks = dataset.get_data()
        return tasks
    

def register_deepcoder_chunked_dataset(return_test: bool = False):
    """
    Compatibility wrapper used by training scripts.

    The current bugs workflows use the registered dataset name `deepcoder_bugs`.
    This function preserves the historical API expected by
    `examples/bugs/train_generator_flow.py` and `examples/bugs/train_generator_solver_flow.py`.
    """
    try:
        train_dataset = DatasetRegistry.load_dataset("deepcoder_bugs", "train")
    except Exception as e:
        print(f"Failed to load dataset deepcoder_bugs/train: {e}")
        print("Available datasets:", DatasetRegistry.get_dataset_names())
        return None if not return_test else (None, None)

    if not return_test:
        return train_dataset

    try:
        test_dataset = DatasetRegistry.load_dataset("deepcoder_bugs", "test")
    except Exception as e:
        print(f"Failed to load dataset deepcoder_bugs/test: {e}")
        test_dataset = None

    return train_dataset, test_dataset


def register_bigcodebench_dataset():
    """
    Compatibility wrapper used by training scripts.

    Returns:
        train_dataset (or None if unavailable)
    """
    try:
        return DatasetRegistry.load_dataset("bigcodebench", "train")
    except Exception as e:
        print(f"Warning: Failed to load BigCodeBench dataset: {e}")
        return None


def parse_dataset_specs(specs) -> list[tuple[str, str]]:
    """
    Parse dataset specs from a list or string format.

    Supported formats:
      - "[dataset1:split1,dataset2:split2]" (string, brackets optional)
      - ["dataset1:split1", "dataset2:split2"] (list)
      - "dataset:split" (single dataset string)

    Returns:
      list[(dataset_name, split)]
    """
    if specs is None:
        return []

    parsed: list[tuple[str, str]] = []

    # Handle string format: "[ds1:split1,ds2:split2]" or "ds1:split1,ds2:split2"
    if isinstance(specs, str):
        s = specs.strip()
        if not s:
            return []
        # Remove brackets if present
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        # Split by comma
        for item in s.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                ds_name, split = item.split(":", 1)
                parsed.append((ds_name.strip(), split.strip()))
            else:
                parsed.append((item, "train"))
        return parsed

    # Handle list format
    try:
        for raw in list(specs):
            if raw is None:
                continue
            s = str(raw).strip()
            if not s:
                continue
            if ":" in s:
                ds_name, split = s.split(":", 1)
                parsed.append((ds_name.strip(), split.strip()))
            else:
                parsed.append((s, "train"))
    except Exception:
        return []

    return parsed


def parse_target_dataset_specs(workflow_args_cfg) -> list[tuple[str, str]]:
    """
    Parse target dataset specs from workflow_args.

    Checks for:
      1. workflow_args.target_datasets (list or string): "[ds1:split1,ds2:split2]"
      2. workflow_args.target_dataset_name (single, for backward compat)

    Returns:
      list[(dataset_name, split)]
    """
    if workflow_args_cfg is None:
        return []

    # First check for multi-dataset format
    target_datasets = getattr(workflow_args_cfg, "target_datasets", None)
    if target_datasets:
        return parse_dataset_specs(target_datasets)

    # Fall back to single dataset (backward compatibility)
    target_dataset_name = getattr(workflow_args_cfg, "target_dataset_name", None)
    if target_dataset_name:
        name = str(target_dataset_name).strip()
        if name:
            split = str(getattr(workflow_args_cfg, "target_dataset_split", "train")).strip() or "train"
            return [(name, split)]

    return []


def parse_val_dataset_specs(workflow_args_cfg) -> dict[str, tuple[str, str]]:
    """
    Shared helper for training scripts to parse multi-val specs from `workflow_args.val_datasets`.

    Supported formats:
      - workflow_args.val_datasets = ["bugs_human_authored:test", "alias=bigcodebench:test", "deepcoder_bugs:test"]
      - workflow_args.val_datasets = {"bugs_human_authored": "test", "bcb": "bigcodebench:test"}

    Returns:
      dict[alias -> (dataset_name, split)]
    """
    if workflow_args_cfg is None:
        return {}

    specs = getattr(workflow_args_cfg, "val_datasets", None)
    if specs is None:
        return {}

    parsed: dict[str, tuple[str, str]] = {}

    # Dict-like (OmegaConf as well)
    if isinstance(specs, dict) or hasattr(specs, "items"):
        for alias, val in dict(specs).items():
            if val is None:
                continue
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    continue
                if ":" in s:
                    ds_name, split = s.split(":", 1)
                    parsed[str(alias)] = (ds_name.strip(), split.strip())
                else:
                    parsed[str(alias)] = (s, "test")
            else:
                try:
                    ds_name = str(val.get("dataset", "")).strip()
                    split = str(val.get("split", "test")).strip()
                    if ds_name:
                        parsed[str(alias)] = (ds_name, split)
                except Exception:
                    continue
        return parsed

    # List-like: ["alias=dataset:split", "dataset:split", "dataset"]
    try:
        for raw in list(specs):
            if raw is None:
                continue
            s = str(raw).strip()
            if not s:
                continue

            alias = None
            if "=" in s:
                alias, s = s.split("=", 1)
                alias = alias.strip() or None
                s = s.strip()

            if ":" in s:
                ds_name, split = s.split(":", 1)
                ds_name = ds_name.strip()
                split = split.strip()
            else:
                ds_name, split = s, "test"

            if not alias:
                alias = f"{ds_name}_{split}"
            parsed[alias] = (ds_name, split)
    except Exception:
        return {}

    return parsed
