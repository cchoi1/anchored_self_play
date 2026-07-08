# Anchored Self-Play for Code Repair

Code and benchmark for **Anchored Self-Play for Code Repair**.

Code repair — given a buggy program and unit tests, produce a fixed program that
passes — is bottlenecked by the scarcity of realistic buggy programs. We scale
supervision by having a language model *generate* bug–fix tasks and validate them
with unit tests, in a **generator–fixer self-play** loop: one policy learns both to
introduce bugs and to fix them, forming an automatic curriculum as the fixer improves.

Unit tests certify that an edit *breaks* a program, but not that the bug is
*realistic*. Left unconstrained, self-play **drifts** — the generator invents
idiosyncratic test-failing edits, improving repair on its own synthetic bugs while
degrading on human-written ones. **Anchored Self-Play (ASP)** counters this by
anchoring to a small reference set of real bugs in two ways:

1. a **code-embedding similarity reward** that guides bug *generation* toward
   reference-like edits, and
2. **reference-bug mixing** that keeps realistic bugs in the fixer's *training*.

Across bug sources, ASP achieves the best fix rates, improving over standard
self-play on both LM-generated and human-authored bugs.

## Repository layout

| Path | What |
|------|------|
| [`training/`](training) | Generator–fixer self-play training (trimmed [rLLM](https://github.com/rllm-org/rllm) + [verl](https://github.com/volcengine/verl) fork). Standard self-play, the two anchoring mechanisms, and the fixer baseline. |
| [`api_eval/`](api_eval) | Evaluation of **API** code models (OpenAI / Anthropic / Google) on BugSourceBench. |
| [`bugsourcebench.csv`](bugsourcebench.csv) | The **BugSourceBench** benchmark (see below). |

Each subdirectory has its own README with setup and usage.

## BugSourceBench

A controlled benchmark for **cross-source repair generalization**. It holds the
programming task, specification, and unit tests fixed while varying only the
*source* of the buggy implementation, so a model must repair different kinds of
bugs for the same underlying tasks. Bug sources (columns in `bugsourcebench.csv`):

- `buggy_Human` — human-authored bugs.
- `buggy_Human-Edited_LM` — human edits of buggy LM-generated code.
- `buggy_LM_Errors_Qwen-7B` — errors from a weaker code LM.
- `buggy_LM_Errors_gpt-oss-20b` — errors from a stronger code LM.

Each row also carries the `canonical_solution`, `test`, prompts, and `entry_point`.

## Citation

```bibtex
@inproceedings{choi2026anchored,
  title     = {Anchored Self-Play for Code Repair},
  author    = {Choi, Caroline and Kaya, Zeyneb and Wu, Shirley and
               Ma, Tengyu and Hashimoto, Tatsunori and Schmidt, Ludwig},
  year      = {2026},
}
```

## License

Apache License 2.0 (see [`LICENSE`](LICENSE)). The training code is derived from
[rLLM](https://github.com/rllm-org/rllm), also Apache 2.0.
