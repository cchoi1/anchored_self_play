# examples/asp/utils.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from rllm.agents.agent import Step, Trajectory
from rllm.engine import ModelOutput

from examples.asp.prompts import _extract_failed_test_output


# =============================================================================
# Task schema helpers (shared across workflows)
# =============================================================================

def _get_problem(task: Dict[str, Any]) -> str:
    """Extract problem description across dataset schemas."""
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("question", "instruct_prompt", "complete_prompt", "prompt", "text", "problem", "description", "code_prompt"):
            val = extra_info.get(key)
            if isinstance(val, str) and val.strip():
                return val

    for key in ("question", "instruct_prompt", "complete_prompt", "prompt", "text", "problem", "description", "code_prompt"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _get_reference_solution(task: Dict[str, Any]) -> str:
    """Extract reference/canonical solution across dataset schemas."""
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("reference_solution", "canonical_solution", "solution", "code", "correct_code", "ground_truth_solution"):
            val = extra_info.get(key)
            if isinstance(val, str) and val.strip():
                return val

    for key in ("reference_solution", "canonical_solution", "solution", "code", "correct_code", "ground_truth_solution"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _get_pregenerated_bug(task: Dict[str, Any]) -> Optional[str]:
    """Extract pregenerated buggy code if present."""
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("buggy_solution", "buggy_sampled_solution", "buggy", "buggy_code", "bug"):
            val = extra_info.get(key)
            if isinstance(val, str) and val.strip():
                return val

    for key in ("buggy_solution", "buggy_sampled_solution", "buggy", "buggy_code", "bug"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _get_ground_truth(task: Dict[str, Any]) -> Any:
    """Extract tests/ground_truth in whatever format reward_fn expects."""
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        for key in ("ground_truth", "test", "test_list", "tests"):
            val = extra_info.get(key)
            if val is not None and val != "":
                return val

    for key in ("ground_truth", "test", "test_list", "tests"):
        val = task.get(key)
        if val is not None and val != "":
            return val
    return None


def _get_entry_point(task: Dict[str, Any]) -> Optional[str]:
    """Extract entry_point for BugBench/BigCodeBench execution."""
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        ep = extra_info.get("entry_point")
        if isinstance(ep, str) and ep.strip():
            return ep.strip()
        meta = extra_info.get("metadata", {})
        if isinstance(meta, dict):
            ep = meta.get("func_name") or meta.get("entry_point")
            if isinstance(ep, str) and ep.strip():
                return ep.strip()

    ep = task.get("entry_point")
    if isinstance(ep, str) and ep.strip():
        return ep.strip()

    meta = task.get("metadata", {})
    if isinstance(meta, dict):
        ep = meta.get("func_name") or meta.get("entry_point")
        if isinstance(ep, str) and ep.strip():
            return ep.strip()
    return None


def _get_data_source(task: Dict[str, Any]) -> str:
    """Infer/extract reward routing key (data_source)."""
    extra_info = task.get("extra_info", {})
    if isinstance(extra_info, dict):
        ds = extra_info.get("data_source")
        if isinstance(ds, str) and ds.strip():
            return ds.strip()

    ds = task.get("data_source")
    if isinstance(ds, str) and ds.strip():
        return ds.strip()

    # Infer BugBench/BigCodeBench
    has_entry_point = bool(task.get("entry_point") or (isinstance(extra_info, dict) and extra_info.get("entry_point")))
    has_string_test = isinstance(task.get("test"), str) or (isinstance(extra_info, dict) and isinstance(extra_info.get("test"), str))
    has_bugbench_prompts = any(
        (task.get(k) or (isinstance(extra_info, dict) and extra_info.get(k)))
        for k in ("instruct_prompt", "complete_prompt", "code_prompt")
    )
    metadata = task.get("metadata") or (extra_info.get("metadata") if isinstance(extra_info, dict) else None) or {}
    if has_entry_point or (has_bugbench_prompts and has_string_test) or (isinstance(metadata, dict) and metadata.get("func_name")):
        # Routing sentinel, not a dataset name: this is only reached for tasks
        # that carry no `data_source` at all, where the source is unknown. It
        # routes to the BigCodeBench test runner in rllm/rewards/code_reward.py.
        # Datasets registered by prepare_data.py always carry their own
        # data_source (bugs_human_authored, bugs_lm_qwen7b, ...) and never land here.
        return "bugbench"

    return "livecodebench"


def normalize_task_info(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize task into the format reward_fn expects:
      - ground_truth
      - data_source
      - entry_point (if available)
    """
    base_info = task.get("extra_info", task)
    task_info = dict(base_info) if isinstance(base_info, dict) else {}

    if "ground_truth" not in task_info or task_info.get("ground_truth") is None:
        gt = _get_ground_truth(task)
        if gt is not None:
            task_info["ground_truth"] = gt

    if "data_source" not in task_info or not task_info.get("data_source"):
        task_info["data_source"] = _get_data_source(task)

    if "entry_point" not in task_info or not task_info.get("entry_point"):
        ep = _get_entry_point(task)
        if ep:
            task_info["entry_point"] = ep

    return task_info


# =============================================================================
# Small shared eval helpers
# =============================================================================

def _pass_ratio(meta: Dict[str, Any]) -> Optional[float]:
    try:
        total = int(meta.get("total_tests", 0))
        passed = int(meta.get("passed_tests", 0))
        return passed / total if total > 0 else None
    except Exception:
        return None


def _resolve_api_key(base_url: str, explicit_api_key: Optional[str]) -> str:
    """Resolve OpenAI-compatible API key, allowing dummy keys for non-OpenAI endpoints."""
    api_key = (explicit_api_key or os.getenv("OPENAI_API_KEY", "")) or ""
    api_key = str(api_key).strip()
    if "api.openai.com" in str(base_url) and not api_key:
        raise ValueError(
            "base_url points to the OpenAI API, but OPENAI_API_KEY is missing/empty. "
            "Please export OPENAI_API_KEY or pass api_key explicitly."
        )
    return api_key or "EMPTY"


def maybe_extract_failed_test_output(
    reward_meta: Dict[str, Any],
    *,
    enabled: bool,
    max_chars: int = 4000,
) -> Optional[str]:
    if not enabled:
        return None
    return _extract_failed_test_output(reward_meta or {}, max_chars=max_chars)


def _model_text(out: ModelOutput) -> str:
    # Be robust to ModelOutput variants (some use .text, some use .content)
    return str((getattr(out, "text", None) or getattr(out, "content", None) or "")).strip()


def _make_trajectory(
    *,
    name: str,
    messages: List[Dict[str, str]],
    assistant_text: str,
    model_output: ModelOutput,
) -> Trajectory:
    chat_completions = messages + [{"role": "assistant", "content": assistant_text}]
    step = Step(
        chat_completions=chat_completions,
        action=assistant_text,
        model_output=model_output,
    )
    # Populate optional attrs if Step supports them (keeps compatibility across older Step defs)
    for k, v in (("thought", ""), ("model_response", assistant_text)):
        if hasattr(step, k):
            setattr(step, k, v)
    return Trajectory(name=name, steps=[step])


# ---------------------------
# SSR-style generator reward shaping
# ---------------------------

def _shaped_generator_reward_from_solve_rate(
    *,
    bug_valid: bool,
    solve_rate: float,
    mode: str,
    band_low: float,
    band_high: float,
    alpha_extreme: float,
    invalid_bug_reward: float,
) -> float:
    """SSR-like shaping: reward 'frontier' examples and penalize degeneracy.

    Modes:
      - "binary": old behavior: reward iff (bug_valid and solve_rate < 1.0)
      - "band":   +1 if solve_rate in [band_low, band_high],
                  -alpha if solve_rate in {0,1}, else 0
      - "smooth": peak in the middle of the band, 0 outside, -alpha at {0,1}

    Notes:
      - solve_rate in [0,1], computed from K solver attempts.
      - Penalizing solve_rate==0 prevents "unsolvable" bugs dominating.
      - Penalizing solve_rate==1 prevents "too easy" / no-op bugs dominating.
    """
    mode = (mode or "band").lower().strip()
    solve_rate = max(0.0, min(1.0, float(solve_rate)))

    if not bug_valid:
        return float(invalid_bug_reward)

    # Extremes (degenerate) get explicit penalty.
    if solve_rate <= 0.0 or solve_rate >= 1.0:
        return float(-alpha_extreme)

    if mode == "binary":
        # old-ish: reward if not trivially solvable by everyone
        return 1.0 if solve_rate < 1.0 else 0.0

    band_low = float(band_low)
    band_high = float(band_high)
    if band_low > band_high:
        band_low, band_high = band_high, band_low

    if mode == "band":
        return 1.0 if (band_low <= solve_rate <= band_high) else 0.0

    if mode == "smooth":
        # Piecewise linear bump: 0 outside band, 1 at band midpoint.
        if solve_rate < band_low or solve_rate > band_high:
            return 0.0
        mid = 0.5 * (band_low + band_high)
        half = max(1e-8, 0.5 * (band_high - band_low))
        # 1 at mid, 0 at edges
        return float(max(0.0, 1.0 - abs(solve_rate - mid) / half))

    # Fallback
    return 1.0 if (band_low <= solve_rate <= band_high) else 0.0


# =============================================================================
# Shared bug validity
# =============================================================================

def check_bug_validity(
    bug_meta: Dict[str, Any],
    bug_reward_output: Any,
    compile_errors_invalid: bool = True,
) -> Tuple[bool, bool]:
    """
    A valid bug must:
      1) Have no compilation errors (if compile_errors_invalid=True)
      2) Fail at least one unit test (passed_tests < total_tests OR all_passed=False)
    """
    total_tests = bug_meta.get("total_tests")
    passed_tests = bug_meta.get("passed_tests")
    all_passed = bug_meta.get("all_passed", False)

    test_results = bug_meta.get("test_results", [])
    has_compile_error = False

    diagnostic_records: List[Dict[str, Any]] = []
    if isinstance(test_results, list):
        diagnostic_records.extend(test for test in test_results if isinstance(test, dict))
    # BigCodeBench also exposes the same diagnostic at the metadata top level.
    diagnostic_records.append(bug_meta)

    for test in diagnostic_records:
        # Different code runners expose diagnostics under different keys.
        # In particular, the BigCodeBench runner stores its diagnostic in
        # `output`, not `error_message`.
        messages = [
            str(test.get(key, "") or "")
            for key in ("error_message", "error", "output", "stderr", "traceback")
        ]
        msg_l = "\n".join(message for message in messages if message).lower()
        if not msg_l:
            continue

        if "error during testing:" in msg_l:
            has_compile_error = True
            break

        if "wrong answer" not in msg_l:
            compile_patterns = [
                "syntax", "syntaxerror", "compilation", "compile error",
                "cannot compile", "indentation", "invalid syntax",
                "unexpected", "eof", "unterminated", "was never closed",
                "nameerror", "typeerror", "attributeerror", "import error",
                "importerror", "modulenotfounderror", "module not found",
                "indentationerror",
            ]
            if any(p in msg_l for p in compile_patterns):
                has_compile_error = True
                break

    if compile_errors_invalid and has_compile_error:
        bug_valid = False
    elif total_tests is not None and passed_tests is not None and total_tests > 0:
        bug_valid = passed_tests < total_tests
    elif all_passed is False:
        bug_valid = True
    else:
        # Fallback: use correctness signal if metadata is incomplete.
        bug_valid = not bool(getattr(bug_reward_output, "is_correct", False))
        if compile_errors_invalid:
            bug_valid = bug_valid and not has_compile_error

    return bug_valid, has_compile_error
