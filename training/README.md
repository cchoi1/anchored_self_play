# Anchored Self-Play — Training

Generator–fixer self-play training for code repair, built on
[rLLM](https://github.com/rllm-org/rllm) + [verl](https://github.com/volcengine/verl).

A single policy is trained with RL to play two roles:

- **Generator** — edits a correct program into a buggy one.
- **Fixer** — repairs the buggy program so it passes the unit tests.

The generator is rewarded for producing valid, appropriately-difficult bugs; the
fixer is rewarded for repairs that pass the tests. **Anchored Self-Play (ours)**
adds two mechanisms that keep the self-generated bugs realistic:

1. **Embedding-similarity reward** on the generator, pulling generated bugs toward
   a small reference set of real bugs.
2. **Reference-bug mixing** into fixer training, so the fixer keeps seeing real bugs.

All of the paper's training code lives in [`examples/asp/`](examples/asp).

## Installation

```bash
# Clone with submodules (pulls in verl)
git clone --recurse-submodules https://github.com/cchoi1/anchored_self_play.git
cd anchored_self_play/training

conda create -n asp python=3.10 -y && conda activate asp

# 1. Install verl (inference frameworks, flash-attn, etc.). Run this FIRST -- it
#    pins torch 2.6.0 / numpy<2 to match the vLLM and flash-attn wheels.
bash scripts/install_verl.sh

# 2. Install this package
pip install -e .

# 3. Install the libraries the unit tests import (see note below)
pip install -r requirements-bigcodebench.txt
```

A CUDA GPU environment is required for training.

**Why step 3 matters.** The reward function runs each task's real unit tests, and
BigCodeBench tests import `matplotlib`, `sklearn`, `seaborn`, `scipy` and friends.
If those are missing the tests fail on import, so correct programs score as
failures and the training signal is silently wrong rather than loudly broken.

The embedding-similarity reward calls Voyage AI, so `VOYAGE_API_KEY` must be set
for anchored self-play (the default). Set `USE_CODE_EMBED_SIM=false` in the
launcher to train without it.

## Data

The launchers load data through `DatasetRegistry`, which reads parquet from
`RLLM_DATASET_DIR` (default `rllm/data/datasets`). Populate it once:

```bash
python -m examples.asp.prepare_data
```

That downloads the released BugSourceBench splits from HuggingFace and registers
the names the training and eval scripts reference:

| Registered name | train / test | Bug source | HuggingFace repo | CSV column |
|---|---|---|---|---|
| `bigcodebench` | 899 / 228 | *(no bug — base tasks)* | *(derived)* | — |
| `bugs_human_authored` | 894 / 227 | Human-authored | `cchoi1/bugs_human_authored` | `buggy_Human` |
| `bugs_human_edited_lm` | 765 / 195 | Human edits of buggy LM code | `cchoi1/bugs_human_edited_lm` | `buggy_Human-Edited_LM` |
| `bugs_lm_qwen7b` | 712 / 195 | Errors from a weaker code LM | `cchoi1/bugs_lm_qwen7b` | `buggy_LM_Errors_Qwen-7B` |
| `bugs_lm_gpt_oss_20b` | 617 / 172 | Errors from a stronger code LM | `cchoi1/bugs_lm_gpt_oss_20b` | `buggy_LM_Errors_gpt-oss-20b` |
| `bugbench_adversarial` | 637 / 174 | Generator optimized against another model | `cchoi1/bugbench_adversarial` | *(not in the CSV)* |

Registry names and HuggingFace repo names now match. Both were verified against
`bugsourcebench.csv` — 127/127 exact on every test row. The repos were previously
named `bugbench`, `bugbench_human`, `bugbench_*_sampled`, where the names did not
describe the contents; those old ids still redirect on the Hub. The mapping lives
in `BUG_SOURCES` in [`prepare_data.py`](examples/asp/prepare_data.py).

`bigcodebench` holds the *base* tasks (problem, reference solution, unit tests --
no bug). It is derived from the bug splits above: rows are pooled, deduplicated
by BigCodeBench task id, and stripped of `buggy_solution`. That keeps prompt
formatting identical to the bug splits and inherits the same train/test
partition, so no held-out benchmark task leaks into training.

Pass `--datasets` / `--splits` to register a subset, and `--force` to re-download.

## Training

One launcher covers every method in the paper via toggles at the top of the file:

```bash
# Anchored self-play (ours): embedding reward + reference mixing (both on by default)
bash examples/asp/train_generator_fixer_flow.sh

# Standard self-play: set USE_CODE_EMBED_SIM=false and USE_PREGEN_BUGS_TRAIN=false
# + embedding reward only: USE_CODE_EMBED_SIM=true,  USE_PREGEN_BUGS_TRAIN=false
# + reference mixing only:  USE_CODE_EMBED_SIM=false, USE_PREGEN_BUGS_TRAIN=true

# Fixer-only baseline (frozen bug generator):
SYNTHESIZER_BASE_URL=http://localhost:32000/v1 \
  bash examples/asp/train_fixer_flow.sh
```

Configure the run with environment variables: `MODEL_PATH`, `NUM_GPUS`, `RUN_DIR`,
and `WANDB_PROJECT`/`WANDB_NAME` (logging falls back to console if unset). The
embedding reward uses Voyage embeddings by default — set `VOYAGE_API_KEY`.

## Evaluation

Evaluate a checkpoint (or base model) served on an OpenAI-compatible endpoint:

```bash
# Start a server for your model first, e.g. vLLM on localhost:30000, then:
bash examples/asp/run_generator_fixer_flow.sh   # fix-rate across bug sources
bash examples/asp/run_fixer_flow.sh             # fixer-only evaluation
```

Both files hold several independent example invocations rather than one pipeline —
copy the one you want instead of executing the whole file.

For evaluating **API models** on BugSourceBench, see [`../api_eval`](../api_eval).

## Layout

```
training/
├── rllm/                       # rLLM framework (trimmed to what self-play needs)
│   ├── agents/ engine/ workflows/ trainer/   # rollout + GRPO training loop
│   ├── rewards/                # code-repair rewards + unit-test execution
│   └── data/ parser/ tools/    # datasets, chat parsing, code tools
├── examples/
│   └── asp/          # generator–fixer self-play (paper code)
│       ├── generator_fixer_flow.py     # the self-play workflow
│       ├── fixer_flow.py / generator_flow.py / frozen_generator_fixer_flow.py
│       ├── components.py, utils.py
│       ├── prompts.py, code_embedding.py, data_utils.py   # prompts, embedding reward, data
│       ├── prepare_data.py             # registers the datasets (run this first)
│       ├── train_*.py / run_*.py       # trainers + eval runners
│       └── train_*.sh / run_*.sh       # launchers
├── scripts/install_verl.sh
├── requirements-bigcodebench.txt   # libraries the unit tests import
├── verl/                       # submodule: volcengine/verl
└── pyproject.toml
```

This training code is a trimmed fork of rLLM; unrelated agents, environments, and
examples from upstream have been removed. See the upstream
[rLLM project](https://github.com/rllm-org/rllm) for the full framework.
