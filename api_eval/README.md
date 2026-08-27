# API Model Evaluation

Evaluation harness for running **API code models** (OpenAI, Anthropic, Google) on
**BugSourceBench**, the multi-source code-repair benchmark from *Anchored Self-Play
for Code Repair*.

It scores a model's ability to repair buggy programs drawn from different bug
sources (human-written, human-edited LM, and LM-generated bugs), holding the task,
specification, and unit tests fixed across sources.

## Layout

```
api_eval/
├── unified_eval/          # the evaluation pipeline
│   ├── run_eval.py        # main entry point (inference + scoring)
│   ├── input.py           # prompt construction per eval mode
│   ├── output.py          # parsing + pass-rate scoring
│   ├── diff_applier.py    # applies model-produced diffs
│   ├── metrics.py         # standalone debugger-vs-coder comparison CLI
│   └── scripts/           # example run scripts, grouped by role (see below)
└── src/                   # shared library used by unified_eval
    ├── utils/api.py       # provider-agnostic get_llm_output()
    ├── utils/api_lib/     # per-provider clients (claude / gpt / gemini / hf)
    ├── utils/{parse,prompt_handler,logger_utils,extract_json_reliable}.py
    └── metrics/, utils/execute/   # unit-test execution + pass-rate metrics
```

## Setup

```bash
# Option A: conda
conda env create -f env.yml && conda activate buggen

# Option B: pip
pip install -r requirements.txt
```

Run everything from the `api_eval/` directory so `unified_eval` and `src` are importable.

## API keys

Keys are read from a git-ignored `config/` directory if present, otherwise from
environment variables:

| Provider  | `config/` file            | Env var                              |
|-----------|---------------------------|--------------------------------------|
| Anthropic | `config/claude_api_key.txt` | `ANTHROPIC_API_KEY`                |
| OpenAI    | `config/openai_api_key.txt` | `OPENAI_API_KEY`                   |
| Google    | `config/gemini_api_key.txt` | `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) |

```bash
mkdir -p config && echo "sk-..." > config/openai_api_key.txt   # or: export OPENAI_API_KEY=sk-...
```

## Running

The entry point is `unified_eval.run_eval`. A run is typically two passes:
inference (generate repairs) then scoring (`--eval` on the saved output).

```bash
# 1) Inference: generate repairs and save them
python -m unified_eval.run_eval \
    --input  <hf_dataset> \
    --mode   solver-attacker-style \
    --model  gpt-4o \
    --mutation-col buggy_solution \
    --output gpt-4o_repairs.json \
    --inference-only \
    --workers 8

# 2) Scoring: run unit tests over the saved repairs
python -m unified_eval.run_eval \
    --input  <hf_dataset> \
    --mode   solver-attacker-style \
    --model  gpt-4o \
    --eval   gpt-4o_repairs.json \
    --output gpt-4o_eval.json \
    --workers 10
```

Re-run inference with `--continue-from <output.json>` to retry only the items that
failed to generate.

### Key arguments

| Arg | Meaning |
|-----|---------|
| `--input` | HF dataset id of the bug-source split to evaluate |
| `--mode` | Prompting/eval mode (see below) |
| `--model` | Model id (e.g. `gpt-4o`, `o4-mini`, `gpt-5.2`, `claude-sonnet-4-5-20250929`, `gemini-2.5-pro`) |
| `--mutation-col` | Dataset column holding the buggy program (`buggy_solution`, `response`, …) |
| `--eval` | Score a saved inference JSON instead of generating |
| `--inference-only` | Generate only, don't score |
| `--continue-from` | Resume/retry from a previous output file |
| `--workers` | Parallel request workers |
| `--max-new-tokens`, `--temperature`, `--limit`, `--offset` | Generation controls |

### Modes

`coder-complete` / `coder-instruct` (fresh code generation),
`solver-complete` / `solver-instruct` / `solver-ambig-*` / `solver-attacker-style` /
`solver-solve-then-patch` (repair variants), `differ` (diff-style repair),
and `bigcodebench-*`.

`solver-test-cases` first runs the buggy program with the benchmark evaluator and
includes the resulting pass/fail diagnostics in the repair prompt. If a dataset
already provides `mutation_info`, `test_case_results`, or `test_results`, that
materialized feedback is used instead. Evaluator failures stop the run rather
than producing an empty feedback section.

### Example scripts (`unified_eval/scripts/`)

Reproduce the paper's runs across all models; grouped by role:

- `*_coder.sh` / `*_coder_eval.sh` — fresh code generation (coder) inference + scoring.
- `*_solver_attacker_all.sh` — repair sweep over all four bug-source datasets
  (human / qwen / adversarial / gpt-oss).
- `*_differ_all.sh` — diff-style repair over all bug sources.
- `*_testcases_all.sh` — test-case generation over all bug sources.
- `bugbench_og_*.sh` — baseline runs on the original BugSourceBench split.

> The example scripts were originally written for a SLURM cluster; the cluster-specific
> preamble has been removed and job submission (`sbatch`) replaced with `bash`, so they
> run on a single machine.

## Datasets

The scripts reference the BugSourceBench splits on HuggingFace under
[`cchoi1`](https://huggingface.co/cchoi1) (e.g. `cchoi1/bugs_human_edited_lm_eval`). The
raw benchmark also ships as `bugsourcebench.csv` at the repository root.

## Reproduction scripts

`unified_eval/scripts/run_eval_all.sh` runs one model across all four
BugSourceBench bug sources for a given mode:

```bash
bash unified_eval/scripts/run_eval_all.sh <mode> <model>

# e.g.
bash unified_eval/scripts/run_eval_all.sh solver-attacker-style gpt-5.2
bash unified_eval/scripts/run_eval_all.sh solver-diff claude-sonnet-4-5-20250929
bash unified_eval/scripts/run_eval_all.sh solver-test-cases o4-mini
```

It replaces the nine per-model scripts that used to live here, which were
identical apart from the model id and mode. Output filenames are keyed by the
bug source (`human_authored`, `human_edited_lm`, `qwen7b`, `gpt_oss_20b`).
