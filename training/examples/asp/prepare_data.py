"""Register the datasets the ASP training / eval scripts expect.

The training and eval launchers load their data through `DatasetRegistry`, which
reads parquet files from `RLLM_DATASET_DIR` (default `rllm/data/datasets`). This
script populates that registry from the released BugSourceBench splits on
HuggingFace.

    python -m examples.asp.prepare_data                       # everything
    python -m examples.asp.prepare_data --datasets bugs_human_edited_lm bigcodebench

Registered names / splits (these are exactly the names referenced by
`train_*.sh` and `run_*.sh`):

    bugs_human_edited_lm                  train / test / test_all
    bugs_human_authored                        train / test / test_all
    bugs_lm_qwen7b         train / test / test_all
    bugs_lm_gpt_oss_20b    train / test / test_all
    bugbench_adversarial            train / test / test_all
    bigcodebench                    train / test / test_all

`bigcodebench` holds the *base* code-generation tasks the generator mutates. It
is derived from the bug-source splits above: rows are pooled, deduplicated by
`uid` (the BigCodeBench task id), and the `buggy_solution` column is dropped, so
only the problem statement, reference solution and unit tests remain. Deriving
it this way keeps the prompt formatting identical to the bug splits and inherits
the same train/test partition, so no held-out benchmark task leaks into training.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Iterable, List

# Bug-source datasets: local registry name -> HuggingFace repo id.
#
# The registry names describe the bug source. The HuggingFace repo ids are the
# published ones and do NOT line up with them -- notably `cchoi1/bugbench` holds
# the human-authored bugs while `cchoi1/bugbench_human` holds the human-edited-LM
# bugs. Verified by matching bug text against bugsourcebench.csv (127/127 exact on
# every test row). Keep this mapping intact if you edit either side.
BUG_SOURCES: Dict[str, str] = {
    "bugs_human_authored": "cchoi1/bugbench",
    "bugs_human_edited_lm": "cchoi1/bugbench_human",
    "bugs_lm_qwen7b": "cchoi1/bugbench_qwen7b_sampled",
    "bugs_lm_gpt_oss_20b": "cchoi1/bugbench_gpt-oss-20b_sampled",
    # Provenance unconfirmed (generator optimized against another model), so it
    # keeps its published name rather than being given a descriptive one.
    "bugbench_adversarial": "cchoi1/bugbench_adversarial",
}

# Base code-generation tasks, derived from BUG_SOURCES (see module docstring).
BASE_DATASET = "bigcodebench"

ALL_DATASETS: List[str] = list(BUG_SOURCES) + [BASE_DATASET]

SPLITS = ("train", "test", "test_all")


def _load_hf_split(repo_id: str, split: str) -> List[Dict[str, Any]]:
    """Download one split of a HuggingFace dataset as a list of dicts."""
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(repo_id, split=split)]


def derive_base_tasks(rows_by_source: Iterable[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Pool bug-source rows into base code-generation tasks.

    Deduplicates by `uid`, drops the buggy program, and rewrites `data_source`
    so the reward function routes these through the BigCodeBench test runner.
    """
    seen: set = set()
    base: List[Dict[str, Any]] = []
    for rows in rows_by_source:
        for row in rows:
            uid = row.get("uid")
            if uid is None or uid in seen:
                continue
            seen.add(uid)
            task = {k: v for k, v in row.items() if k != "buggy_solution"}
            task["data_source"] = BASE_DATASET
            base.append(task)
    base.sort(key=lambda r: str(r.get("uid", "")))
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Which datasets to register (default: all).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Which splits to register (default: train test test_all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite splits that are already registered.",
    )
    args = parser.parse_args()

    from rllm.data.dataset import DatasetRegistry

    requested = list(args.datasets)
    # `bigcodebench` is derived from the bug sources, so they must be fetched too.
    needed_sources = set(name for name in requested if name in BUG_SOURCES)
    if BASE_DATASET in requested:
        needed_sources |= set(BUG_SOURCES)

    print(f"Dataset directory: {DatasetRegistry._DATASET_DIR}")

    fetched: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name in sorted(needed_sources):
        repo_id = BUG_SOURCES[name]
        fetched[name] = {}
        for split in args.splits:
            print(f"  downloading {repo_id}:{split} ...", flush=True)
            fetched[name][split] = _load_hf_split(repo_id, split)

    registered = 0
    for name in requested:
        if name == BASE_DATASET:
            continue
        for split in args.splits:
            if not args.force and DatasetRegistry.dataset_exists(name, split):
                print(f"  skip   {name}:{split} (already registered; use --force to overwrite)")
                continue
            rows = fetched[name][split]
            for row in rows:
                row["data_source"] = name
            DatasetRegistry.register_dataset(name, rows, split)
            print(f"  ok     {name}:{split}  n={len(rows)}")
            registered += 1

    if BASE_DATASET in requested:
        for split in args.splits:
            if not args.force and DatasetRegistry.dataset_exists(BASE_DATASET, split):
                print(f"  skip   {BASE_DATASET}:{split} (already registered; use --force to overwrite)")
                continue
            rows = derive_base_tasks(fetched[src][split] for src in sorted(BUG_SOURCES))
            DatasetRegistry.register_dataset(BASE_DATASET, rows, split)
            print(f"  ok     {BASE_DATASET}:{split}  n={len(rows)}  (derived, deduplicated by uid)")
            registered += 1

    print(f"\nRegistered {registered} split(s).")
    print("Available datasets:", ", ".join(sorted(DatasetRegistry.get_dataset_names())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
