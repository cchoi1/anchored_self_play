# Anchored Self-Play for Code Repair

Code and benchmark for **Anchored Self-Play for Code Repair** — generator–fixer
self-play RL for code repair, plus **BugSourceBench** for cross-source repair evaluation.

| Path | What | Run |
|------|------|-----|
| [`training/`](training) | Generator–fixer self-play training (rLLM + verl). | [`training/README.md`](training/README.md) |
| [`api_eval/`](api_eval) | Evaluate API models (OpenAI/Anthropic/Google) on BugSourceBench. | [`api_eval/README.md`](api_eval/README.md) |
| [`bugsourcebench.csv`](bugsourcebench.csv) | The BugSourceBench benchmark (see below). | — |

## Training

```bash
git clone --recurse-submodules https://github.com/cchoi1/anchored_self_play.git
cd anchored_self_play/training
conda create -n asp python=3.10 -y && conda activate asp
bash scripts/install_verl.sh          # verl / vllm / flash-attn (run first: pins torch/numpy)
pip install -e .
pip install -r requirements-bigcodebench.txt   # libraries the unit tests import
python -m examples.asp.prepare_data            # download + register the datasets
```

Launchers live in [`training/examples/asp/`](training/examples/asp):

```bash
# Anchored self-play (ours): embedding reward + reference mixing (on by default)
bash examples/asp/train_generator_fixer_flow.sh

# Ablations — toggle the flags at the top of that script:
#   standard self-play : USE_CODE_EMBED_SIM=false USE_PREGEN_BUGS_TRAIN=false
#   + embedding reward : USE_CODE_EMBED_SIM=true  USE_PREGEN_BUGS_TRAIN=false
#   + reference mixing : USE_CODE_EMBED_SIM=false USE_PREGEN_BUGS_TRAIN=true

# Baselines
SYNTHESIZER_BASE_URL=http://localhost:32000/v1 bash examples/asp/train_fixer_flow.sh
GENERATOR_BASE_URL=http://localhost:32000/v1 bash examples/asp/train_frozen_generator_fixer_flow.sh
```

Evaluate a trained checkpoint served on an OpenAI-compatible endpoint:

```bash
bash examples/asp/run_generator_fixer_flow.sh   # fix-rate across bug sources
bash examples/asp/run_fixer_flow.sh
```

Config via env vars (`MODEL_PATH`, `NUM_GPUS`, `RUN_DIR`, `WANDB_PROJECT`, `VOYAGE_API_KEY`);
see [`training/README.md`](training/README.md) for details and data setup.

## Evaluating API models

```bash
cd api_eval
pip install -r requirements.txt          # or: conda env create -f env.yml
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GOOGLE_API_KEY=...

# inference + scoring on a bug-source split
python -m unified_eval.run_eval --input cchoi1/bugs_human_edited_lm_eval \
    --mode solver-attacker-style --model gpt-4o --mutation-col buggy_solution \
    --output out.json --inference-only
python -m unified_eval.run_eval --input cchoi1/bugs_human_edited_lm_eval \
    --mode solver-attacker-style --model gpt-4o --eval out.json --output out_eval.json
```

To sweep one model across all four bug sources, use
[`api_eval/unified_eval/scripts/run_eval_all.sh`](api_eval/unified_eval/scripts/run_eval_all.sh):

```bash
bash run_eval_all.sh solver-attacker-style gpt-5.2   # mode, model
```

Modes are `solver-attacker-style`, `solver-diff` and `solver-test-cases`.
See [`api_eval/README.md`](api_eval/README.md) for all modes and arguments.

## BugSourceBench

`bugsourcebench.csv` holds task, spec, and unit tests fixed while varying the bug source
(one column per source):

| Column | Bug source | Training split | Full eval split |
|---|---|---|---|
| `buggy_Human` | human-authored bugs | `cchoi1/bugs_human_authored` | `cchoi1/bugs_human_authored_eval` |
| `buggy_Human-Edited_LM` | human edits of buggy LM code | `cchoi1/bugs_human_edited_lm` | `cchoi1/bugs_human_edited_lm_eval` |
| `buggy_LM_Errors_Qwen-7B` | errors from a weaker code LM | `cchoi1/bugs_lm_qwen7b` | `cchoi1/bugs_lm_qwen7b_eval` |
| `buggy_LM_Errors_gpt-oss-20b` | errors from a stronger code LM | `cchoi1/bugs_lm_gpt_oss_20b` | `cchoi1/bugs_lm_gpt_oss_20b_eval` |

Plus `canonical_solution`, `test`, prompts, and `entry_point`. Splits are on
HuggingFace under [`cchoi1`](https://huggingface.co/cchoi1). The training splits
carry the RL train/test partition; the `_eval` splits are the larger 561-task
sets used by [`api_eval`](api_eval). These repos were previously named
`bugbench*`, where the names did not describe the contents; the old ids still
redirect.

## Citation

```bibtex
@inproceedings{choi2026anchored,
  title  = {Anchored Self-Play for Code Repair},
  author = {Choi, Caroline and Kaya, Zeyneb and Wu, Shirley and
            Ma, Tengyu and Hashimoto, Tatsunori and Schmidt, Ludwig},
  year   = {2026},
}
```

## License

Apache License 2.0 (see [`LICENSE`](LICENSE)). Training code derives from
[rLLM](https://github.com/rllm-org/rllm) (Apache 2.0).
