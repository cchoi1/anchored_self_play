from __future__ import annotations

from typing import Any, Optional


def _extract_failed_test_output(metadata: Any, max_chars: int = 4000) -> Optional[str]:
    """
    Best-effort extraction of failed unit test output from RewardOutput.metadata.

    RewardCodeFn returns different shapes depending on dataset runner, but commonly includes:
      - "output" / "error"
      - "test_results" entries with "output" / "error"
      - BigCodeBench/BugBench: details dict with keys like "ALL" or per-test trace strings
      - TACO/APPS: "debug" list with "stderr" / "exec_outputs"
    """
    if not isinstance(max_chars, int) or max_chars <= 0:
        max_chars = 4000

    if metadata is None:
        return None

    parts: list[str] = []

    # Common simple keys
    if isinstance(metadata, dict):
        for k in ("error", "output", "stderr", "traceback"):
            v = metadata.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(f"{k}:\n{v.strip()}")

        # BigCodeBench/BugBench: details dict with per-test traces
        # If present, include ALL first, else include a few trace entries.
        if "ALL" in metadata and isinstance(metadata["ALL"], str) and metadata["ALL"].strip():
            parts.append(f"tests:\n{metadata['ALL'].strip()}")
        else:
            trace_items: list[str] = []
            for k, v in metadata.items():
                if k in (
                    "all_passed",
                    "passed_tests",
                    "total_tests",
                    "test_results",
                    "debug",
                    "error",
                    "output",
                    "stderr",
                    "traceback",
                ):
                    continue
                if isinstance(v, str) and v.strip():
                    trace_items.append(f"{k}:\n{v.strip()}")
            if trace_items:
                # Keep it short-ish.
                parts.append("\n\n".join(trace_items[:3]))

        # Structured test results
        tr = metadata.get("test_results")
        if isinstance(tr, list):
            # Find first failing entry with output-like fields.
            for item in tr:
                if not isinstance(item, dict):
                    continue
                passed = item.get("passed", None)
                if passed is True:
                    continue
                for k in ("output", "error", "stderr"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(f"test_result.{k}:\n{v.strip()}")
                        break
                break

        # TACO debug entries
        dbg = metadata.get("debug")
        if isinstance(dbg, list) and dbg:
            d0 = dbg[0]
            if isinstance(d0, dict):
                for k in ("stderr", "exec_outputs"):
                    v = d0.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(f"debug.{k}:\n{v.strip()}")

    # Last resort: stringify
    if not parts:
        try:
            s = str(metadata).strip()
            if s:
                parts.append(s)
        except Exception:
            return None

    text = "\n\n".join(parts).strip()
    if not text:
        return None

    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n\n...(truncated)"
    return text


def _build_bug_generator_prompt(problem: str, correct_code: str) -> str:
    """Format the prompt for the bug generator.

    We assume `problem` is the original DeepCoder prompt (natural language +
    I/O format) and `correct_code` is a correct reference solution.
    """
    return f"""You are a *bug generator* for Python solutions to competitive programming problems.

You will be given:

1. A problem description.
2. A *correct* reference implementation in Python.

Your task:

- Introduce **one or a few subtle bugs** into the code.
- The resulting code **must still be syntactically valid Python**.
- It should change the behavior so that **at least one unit test fails**.
- Do **not** drastically rewrite the code; keep the overall structure similar.
- Do **not** change the function signature, imports, or I/O format.
- Output **only** the full buggy Python code inside a single ```python``` block.

Problem:
{problem}

Correct reference implementation:
{correct_code}

Now generate the buggy version of this code. Return the entire function with the buggy code inside a ```python``` block:"""


def _build_human_bug_generator_prompt(problem: str, starter_code: str | None = None) -> str:
    """
    Prompt the model to write a plausible *human-like* first-draft solution that contains a subtle bug.

    The intent is to produce natural mistakes (off-by-one, boundary handling, wrong default, etc.)
    rather than explicit "inject a bug into this reference solution" edits.
    """
    p = (problem or "").strip()
    cp = (starter_code or "").strip("\n") if isinstance(starter_code, str) else ""

    parts = [
        "Pretend you are a human writing a first-draft solution.",
        "What kind of buggy solution might you naturally make in trying to solve this problem?",
        "Write a plausible Python solution that a human might naturally produce, but it should contain a subtle bug so it fails on some edge cases.",
        "Do NOT mention that it is buggy.",
        "Return ONLY the full Python solution inside a single ```python``` block.",
        "",
        "PROBLEM:",
        p,
    ]
    if cp.strip():
        parts += [
            "",
            "STARTER CODE (keep the same signature / entry point):",
            "```python",
            cp,
            "```",
        ]
    return "\n".join(parts).strip()

def _build_bug_fixer_prompt(
    problem: str,
    buggy_code: str,
    *,
    include_failed_test_output: bool = False,
    failed_test_output: str | None = None,
) -> str:
    """Format the prompt for the bug fixer (solver).

    Args:
        problem: Natural language problem statement.
        buggy_code: The buggy solution to fix.
        include_failed_test_output: If True, include a section with the unit test failure output.
        failed_test_output: The failure output string (stderr/traceback/test summary). Ignored unless enabled.
    """
    failed_section = ""
    if include_failed_test_output and failed_test_output and str(failed_test_output).strip():
        failed_section = f"""

Failed unit test output from running the buggy implementation:
```text
{str(failed_test_output).strip()}
```"""

    return f"""You are an expert Python debugging assistant.

You will be given:

1. A problem description.
2. A buggy Python implementation that may fail some hidden unit tests.
{failed_section}

Your task:

- Carefully read the code and identify the bug(s).
- Produce a fixed version of the code that makes all unit tests pass.
- Preserve the original function signature, imports, and I/O format.
- Keep the solution reasonably close to the given implementation.
- Output **only** the full corrected Python code inside a single ```python``` block.

Problem:
{problem}

Buggy implementation:
{buggy_code}
```

Now fix the bugs in this code. Return the entire function with the fixed code inside a ```python``` block:"""

# def _build_bug_similarity_judge_prompt(
#     generated_problem: str,
#     generated_bug: str,
#     target_problem: str,
#     target_bug: str,
#     generated_ground_truth: str = "",
#     target_ground_truth: str = "",
# ) -> str:
#     """Build prompt for LLM judge to score bug similarity."""

#     generated_gt_section = ""
#     if generated_ground_truth and generated_ground_truth.strip():
#         generated_gt_section = f"""
# Correct solution (for reference only; use to infer intended behavior / edit size):
# ```python
# {generated_ground_truth}
# ```"""

#     target_gt_section = ""
#     if target_ground_truth and target_ground_truth.strip():
#         target_gt_section = f"""
# Correct solution (for reference only; use to infer intended behavior / edit size):
# ```python
# {target_ground_truth}
# ```"""

#     return f"""You are an expert code reviewer evaluating the similarity of bug patterns between TWO buggy code solutions.

# IMPORTANT RULES:
# - The bugs may come from different problems. Ignore problem domain and variable names.
# - Focus ONLY on the type/nature of the bug and how it was introduced.
# - If correct solutions are provided, use them ONLY to infer the intended behavior and approximate bug style/type and edit locality.

# ========================
# Bug A (Generated)
# ========================
# Problem:
# {generated_problem}
# {generated_gt_section}

# Buggy code:
# ```python
# {generated_bug}
# ========================
# Bug B (Target / Reference)

# Problem:
# {target_problem}
# {target_gt_section}

# Buggy code:

# {target_bug}

# ========================
# Task

# Evaluate how similar Bug A’s bug pattern is to Bug B’s bug pattern along these dimensions:

# Bug Type

# Same category (e.g., off-by-one, wrong comparator/operator, missing edge case, boundary error, logic inversion)?

# Bug Subtlety

# Similar difficulty to detect? Similar likelihood of passing “most” cases?

# Bug Mechanism / Pattern

# Similar reasoning error or coding mistake (what exactly is wrong and why it fails)?

# Edit Size (relative to correct solution if available)

# Similar magnitude: single token/char vs small 1-line vs multi-line?

# Edit Locality

# Similar location in code structure: loop bounds/conditions, indexing, return statement, arithmetic expression, branch condition, etc.?

# ========================
# Scoring (0 to 10)

# Use this rubric:

# 0–2: Completely different bug types/patterns and clearly different edit size/locality

# 3–4: Different bug types, but some similarity in subtlety OR edit size/locality

# 5–6: Similar broad category, but different specific mechanism OR significantly different edit size/locality

# 7–8: Very similar mechanism/pattern with comparable edit size and locality; minor differences

# 9–10: Nearly identical bug pattern (same mechanism, similar subtlety, very comparable edit size and locality)

# ========================
# Output (JSON only)

# Return ONLY a JSON object (no markdown, no code fences, no extra text) in this exact schema:

# {{"score": <number 0-10>, "reasoning": "<brief explanation>"}}
# """

def _build_bug_similarity_judge_prompt(
    generated_problem: str,
    generated_bug: str,
    target_problem: str,
    target_bug: str,
    generated_ground_truth: str = "",
    target_ground_truth: str = "",
) -> str:
    """Build prompt for LLM judge to score bug *style* similarity (generator fingerprint)."""

    generated_gt_section = ""
    if generated_ground_truth and generated_ground_truth.strip():
        generated_gt_section = f"""
Correct solution (reference only; use to infer intended behavior and estimate the delta):
```python
{generated_ground_truth}
```"""

    target_gt_section = ""
    if target_ground_truth and target_ground_truth.strip():
        target_gt_section = f"""
Correct solution (reference only; use to infer intended behavior and estimate the delta):
```python
{target_ground_truth}
```"""

    return f"""You are a forensic analyst of LLM-generated code bugs. Your goal is NOT to judge whether the *bug types* match in an abstract sense, but whether the TWO buggy solutions exhibit a similar *bug-generation style / fingerprint* that could plausibly come from the same generator family.

KEY IDEA:
Different generators tend to inject bugs in characteristic ways (edit patterns, locality, aggressiveness, artifacts). You must compare those style cues.

IMPORTANT RULES:
- The bugs may come from different problems. Ignore problem domain, algorithm choice, and variable names unless they reveal generator style (e.g., consistent refactor habits).
- Do NOT reward “same broad bug category” by itself. Two off-by-one bugs can have totally different *injection styles*.
- If correct solutions are provided, use them ONLY to infer intended behavior and estimate the delta (what changed, how much, and where).
- Focus on “how the bug was introduced” and any recurring generator artifacts.

========================
Bug A
========================
Problem (context only; don’t compare domains):
{generated_problem}
{generated_gt_section}

Buggy code:
```python
{generated_bug}
========================
Bug B
Problem (context only; don’t compare domains):
{target_problem}
{target_gt_section}

Buggy code:

python
Copy code
{target_bug}
========================
TASK: Compare BUG-GENERATION STYLE (generator fingerprint)

Step 1 — Extract a compact “Style Signature” for each bug (A and B).
Describe each signature using the dimensions below (short phrases, not essays):

Injection Operator (what kind of edit/perturbation?)

e.g., operator swap (>= vs >), sign flip, constant tweak, index shift, wrong variable used,
condition inversion, missing guard, wrong loop bound, wrong return, wrong init, data-structure misuse,
premature break/continue, wrong function call, swapped arguments, silent truncation/cast, etc.

Edit Granularity + Minimality (how big / how surgical?)

single token/char vs one-line tweak vs multi-line rewrite vs structural refactor.

Is the bug “minimally invasive” or does it involve unnecessary surrounding changes?

Edit Locality Pattern (where does the generator like to touch?)

boundary checks, loop bounds, indexing, accumulator updates, return statement, branching condition,
base case, sorting/comparison, parsing, I/O, default values, etc.

Failure Profile Bias (what kinds of cases does it tend to break?)

edge cases only, corner-case boundaries, negative/zero, empty input, last element, ties, overflow-ish,
“almost always wrong”, “usually right but fails rare cases”, etc.

Does it look test-aware (breaks exactly one-ish scenario) vs blunt?

Artifacts / Generator Tells (style/formatting/structure clues)

unnecessary helper functions, redundant variables, overly generic patterns, defensive code,
oddly verbose comments, consistent formatting quirks, repeated idioms, suspicious renamings,
unused imports, placeholder logic, “LLM-ish” refactor habits, etc.

Step 2 — Compare the two signatures and judge whether they look like the same generator style.

Highlight 2–5 strongest shared style cues.

Highlight 2–5 strongest differences (especially if they suggest different generators).

========================
STYLE-SIMILARITY SCORE (0 to 10)

Interpret the score as: “Likelihood these bugs come from the same generator family/style.”

Rubric:
0–2 : Clearly different injection operator AND different minimality/locality/artifacts.
3–4 : Mostly different style, but 1 weak overlap (e.g., both boundary edits but otherwise unlike).
5–6 : Some shared fingerprint traits (similar injection operator + similar minimality OR similar artifacts),
but notable differences remain.
7–8 : Strong match in injection operator + minimality + locality pattern; artifacts/failure profile broadly align.
9–10 : Near-identical fingerprint: same perturbation strategy, similar delta size, similar touchpoints,
and similar generator tells; differences are minor.

========================
OUTPUT (JSON ONLY)

Return ONLY a JSON object (no markdown, no code fences, no extra text) in this exact schema:

{{"score": <number 0-10>, "reasoning": "<concise: A_signature; B_signature; shared_cues; differences; conclusion>"}}

Constraints for reasoning:

Keep it concise (prefer semicolon-separated phrases).

Do not include problem-domain comparisons.
"""


def _build_code_generation_prompt(task: dict) -> str:
    """Build the *solver's* regular code-generation prompt for a given task.

    This should be dataset-specific:
    - **deepcoder**: use the LiveCodeBench-formatted `task["question"]` (see
      `examples/deepcoder/prepare_deepcoder_data.py`).
    - **bigcodebench**: use BigCodeBench's native *instruct* prompt when available
      (see `examples/bugs/data_processing/prepare_bigcodebench_data.py`).
    """
    data_source = str(task.get("data_source", "")).lower()

    # BigCodeBench: prefer the native instruct prompt.
    if data_source == "bigcodebench":
        instruct_prompt = task.get("instruct_prompt")
        if isinstance(instruct_prompt, str) and instruct_prompt.strip():
            return instruct_prompt

    # Default / DeepCoder: `question` is typically already LCB-formatted.
    problem = task.get("question") or ""
    if isinstance(problem, str) and "### Answer:" in problem and "```python" in problem:
        return problem

    # Fallback: apply the same formatting used by `fetch_live_code_bench_system_prompt`.
    starter_code = task.get("starter_code")
    if not isinstance(starter_code, str):
        starter_code = None

    from rllm.system_prompts import (
        LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE,
        LCB_FORMATTING_WITHOUT_STARTER_CODE,
        LCB_SYSTEM_MESSAGE_GENERIC,
    )

    prompt = LCB_SYSTEM_MESSAGE_GENERIC + "\n\n" + str(problem)
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


if __name__ == "__main__":
    # Quick smoke-test for `_extract_failed_test_output` on a few dataset examples.
    #
    # We intentionally run *bad code* through `code_reward_fn` to force test failures
    # and inspect what metadata comes back.
    from pprint import pprint

    from rllm.data.dataset import DatasetRegistry
    from rllm.rewards.code_reward import RewardCodeFn
    from rllm.rewards.reward_types import RewardConfig

    def _normalize_task_info(task: dict) -> dict:
        task_info = task.get("extra_info", task)
        if "ground_truth" not in task_info and "test" in task_info:
            task_info = dict(task_info)
            task_info["ground_truth"] = task_info["test"]
        return task_info

    def _sample(ds_name: str, split: str, n: int = 3) -> list[dict]:
        ds = DatasetRegistry.load_dataset(ds_name, split)
        if ds is None:
            raise ValueError(f"Dataset not found: {ds_name!r} split={split!r}")
        data = ds.get_data()
        return data[: min(n, len(data))]

    # Try to force failures in a consistent way.
    # BAD_CODE = "```python\nraise RuntimeError('forced failure')\n```"
    BAD_CODE = "```python\nprint('forced failure')\n```"

    targets = [
        ("deepcoder_bugs", "test", 3),
        ("bigcodebench", "test", 3),
        ("bugbench", "test", 3),
        ("lcb_bugbench", "test", 3),
        ("bugbench_gpt_oss_20b_sampled", "test", 3),
        ("bugbench_qwen7b_sampled", "test", 3),
    ]

    # Some environments may not have the optional dependencies used by BigCodeBench's sandbox.
    try:
        import importlib

        importlib.import_module("matplotlib.pyplot")
        _HAS_MPL = True
    except Exception:
        _HAS_MPL = False

    for ds_name, split, n in targets:
        if ds_name in ("bigcodebench", "bugbench", "bugbench_gpt_oss_20b_sampled", "bugbench_qwen7b_sampled") and not _HAS_MPL:
            print("\n" + "=" * 80)
            print(f"Dataset: {ds_name}  split: {split}  n={n}")
            print("=" * 80)
            print("Skipping: BigCodeBench/BugBench runner requires `matplotlib` in this environment.")
            continue

        print("\n" + "=" * 80)
        print(f"Dataset: {ds_name}  split: {split}  n={n}")
        print("=" * 80)
        for i, task in enumerate(_sample(ds_name, split, n=n)):
            task_info = _normalize_task_info(task)
            try:
                out = RewardCodeFn(RewardConfig())(task_info=task_info, action=BAD_CODE)
                print(f"raw output: {out}")
            except Exception as e:
                print(f"[{ds_name}#{i}] reward_fn failed: {e}")
                continue

            failed = _extract_failed_test_output(out.metadata, max_chars=1200)
            print(f"\n[{ds_name}#{i}] is_correct={out.is_correct} reward={out.reward}")
            print(f"len of failed_test_output: {len(str(failed).strip())}")
            print("---- extracted failed test output ----")
            print(failed or "(none extracted)")
            print("---- metadata keys ----")
            if isinstance(out.metadata, dict):
                pprint(sorted(list(out.metadata.keys()))[:50])
            else:
                pprint(type(out.metadata))