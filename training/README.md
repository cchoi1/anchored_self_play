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

All of the paper's training code lives in [`examples/bugs_refactor/`](examples/bugs_refactor).

## Installation

```bash
# Clone with submodules (pulls in verl)
git clone --recurse-submodules https://github.com/cchoi1/anchored_self_play.git
cd anchored_self_play/training

conda create -n asp python=3.10 -y && conda activate asp

# Install verl (inference frameworks, flash-attn, etc.)
bash scripts/install_verl.sh

# Install this package
pip install -e .
```

A CUDA GPU environment is required for training.

## Data

Training draws base code-generation tasks (e.g. BigCodeBench) and a small
**reference bug set** from [BugSourceBench](../bugsourcebench.csv). Register
datasets as parquet with the `DatasetRegistry`; dataset-prep helpers are in
`rllm/data`. Set `RLLM_DATASET_DIR` to control where registered parquet files live
(defaults to `rllm/data/datasets`).

The training/eval scripts refer to these splits: `bigcodebench` (base tasks),
`bugbench_human`, `bugbench_qwen7b_sampled`, `bugbench_gpt-oss-20b_sampled`, and
`bugbench` — the BugSourceBench bug sources.

## Training

One launcher covers every method in the paper via toggles at the top of the file:

```bash
# Anchored self-play (ours): embedding reward + reference mixing (both on by default)
bash examples/bugs_refactor/train_generator_fixer_flow.sh

# Standard self-play: set USE_CODE_EMBED_SIM=false and USE_PREGEN_BUGS_TRAIN=false
# + embedding reward only: USE_CODE_EMBED_SIM=true,  USE_PREGEN_BUGS_TRAIN=false
# + reference mixing only:  USE_CODE_EMBED_SIM=false, USE_PREGEN_BUGS_TRAIN=true

# Fixer-only baseline (frozen bug generator):
SYNTHESIZER_BASE_URL=http://localhost:32000/v1 \
  bash examples/bugs_refactor/train_fixer_flow.sh
```

Configure the run with environment variables: `MODEL_PATH`, `NUM_GPUS`, `RUN_DIR`,
and `WANDB_PROJECT`/`WANDB_NAME` (logging falls back to console if unset). The
embedding reward uses Voyage embeddings by default — set `VOYAGE_API_KEY`.

## Evaluation

Evaluate a checkpoint (or base model) served on an OpenAI-compatible endpoint:

```bash
# Start a server for your model first, e.g. vLLM on localhost:30000, then:
bash examples/bugs_refactor/run_generator_fixer_flow.sh   # fix-rate across bug sources
bash examples/bugs_refactor/run_fixer_flow.sh             # fixer-only evaluation
```

For evaluating **API models** on BugSourceBench, see [`../api_eval`](../api_eval).

## Layout

```
training/
├── rllm/                       # rLLM framework (trimmed to what self-play needs)
│   ├── agents/ engine/ workflows/ trainer/   # rollout + GRPO training loop
│   ├── rewards/                # code-repair rewards + unit-test execution
│   └── data/ parser/ tools/    # datasets, chat parsing, code tools
├── examples/
│   ├── bugs_refactor/          # generator–fixer self-play (paper code)
│   │   ├── generator_fixer_flow.py     # the self-play workflow
│   │   ├── fixer_flow.py / generator_flow.py / frozen_generator_fixer_flow.py
│   │   ├── components.py, utils.py
│   │   ├── train_*.py / run_*.py       # trainers + eval runners
│   │   └── train_*.sh / run_*.sh       # launchers
│   └── bugs/                   # prompts, code-embedding reward, data utils
├── scripts/install_verl.sh
├── verl/                       # submodule: volcengine/verl
└── pyproject.toml
```

This training code is a trimmed fork of rLLM; unrelated agents, environments, and
examples from upstream have been removed. See the upstream
[rLLM project](https://github.com/rllm-org/rllm) for the full framework.
