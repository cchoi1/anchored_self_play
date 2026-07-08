import ast
import os
import re
import sys
import textwrap
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.pass_rate import PassRateMetric
from src.utils.parse import parse_response
from unified_eval.diff_applier import apply_diff

BODY_ONLY_MODES = {"solver-complete", "solver-ambig-complete", "solver_prompt", "coder-complete"}


def _extract_diff(raw: str) -> Optional[str]:
    def clean(text):
        # filter line nums if added
        return "\n".join(re.sub(r"^([-+ ]?)(\d{4}:\s?)", r"\1", ln) for ln in text.splitlines())
    
    # fenced 
    for m in re.findall(r"```(?:diff|patch)?\s*\n(.*?)```", raw, re.DOTALL | re.I):
        if "@@" in m or re.search(r"^[-+]", m, re.M) or "*** " in m:
            return clean(m.strip())
    
    # v4a 
    if "*** Begin Patch" in raw or "*** Update File" in raw:
        lines, capture = [], False
        for ln in raw.splitlines():
            if "*** Begin Patch" in ln or "*** Update File" in ln:
                capture = True
            if capture:
                lines.append(ln)
                if "*** End Patch" in ln:
                    break
        if lines:
            return clean("\n".join(lines))
    
    # inline diff
    if "@@" in raw:
        lines, capture = [], False
        for ln in raw.splitlines():
            if ln.startswith(("@@", "---", "+++")) or ln.strip() == "@@":
                capture = True
            if capture:
                if ln.startswith("```") or ln.strip().lower().startswith("explanation:"):
                    break
                lines.append(ln)
        if lines:
            return clean("\n".join(lines))
    
    return None


def _extract_code(raw: str) -> Optional[str]:
    # collapse repeated ```python markers
    cleaned = raw
    while True:
        new_cleaned = re.sub(r'```(?:python)?\s*\n```(?:python)?\s*\n', '```python\n', cleaned)
        new_cleaned = re.sub(r'```(?:python)?\s*```(?:python)?\s*\n', '```python\n', new_cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned
    
    for pat in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        for m in re.findall(pat, cleaned, re.DOTALL):
            code = m.strip()
            
            code = textwrap.dedent(code)
            if not code.startswith(("@@", "---", "+++")):
                try:
                    ast.parse(code)
                    return code
                except SyntaxError:
                    pass
    return None


def _extract_explanation(raw: str) -> Optional[str]:
    for line in raw.splitlines():
        if line.strip().lower().startswith("explanation:"):
            return line.split(":", 1)[1].strip() or None
    return None


def _process_solve_then_patch(raw: str, mutation: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    python_result = {}
    diff_result = {}
    
    # python code block
    python_code = _extract_code(raw)
    if python_code:
        python_result["solution"] = python_code + "\n"
        python_result["apply_mode"] = "python_direct"
    else:
        python_result["solution"] = None
        python_result["apply_mode"] = "python_not_found"
    
    # extract + apply diff block
    diff = _extract_diff(raw)
    if diff:
        diff_result["edit"] = diff
        result = apply_diff(mutation, diff)
        
        if result.success and result.valid_python():
            diff_result["solution"] = result.code
            diff_result["apply_mode"] = "diff"
        elif result.success:
            diff_result["solution"] = None
            diff_result["apply_mode"] = "diff-invalid"
            diff_result["failed_patch"] = result.code
        else:
            diff_result["solution"] = None
            diff_result["apply_mode"] = "diff-failed"
            diff_result["diff_error"] = result.error
    else:
        diff_result["solution"] = None
        diff_result["apply_mode"] = "diff_not_found"
    
    return python_result, diff_result


def _process_diff(raw: str, mutation: str) -> Tuple[Optional[str], Dict]:
    extra = {}
    
    diff = _extract_diff(raw)
    if diff:
        extra["edit"] = diff
        result = apply_diff(mutation, diff)
        
        if result.success and result.valid_python():
            extra["apply_mode"] = "diff"
            return result.code, extra
        elif result.success:
            extra["apply_mode"] = "diff-invalid"
            extra["failed_patch"] = result.code
        else:
            extra["apply_mode"] = "diff-failed"
            extra["diff_error"] = result.error
    
    # fallback
    code = _extract_code(raw)
    if code:
        extra["apply_mode"] = "fullcode"
        return code + "\n", extra
    
    return None, extra


def _process_text(raw: str, mode: str) -> Optional[str]:
    try:
        solution = parse_response(raw).get("code") or raw
    except Exception:
        solution = raw
    
    if mode in BODY_ONLY_MODES and solution:
        sol = solution.strip("\n")
        
        has_toplevel_def = any(line.startswith(("def ", "import ", "from ", "class ")) for line in sol.splitlines())
        if not has_toplevel_def and sol.strip() and any(line.strip() and not line.startswith((" ", "\t")) for line in sol.splitlines()):
            solution = textwrap.indent(sol, "    ") + "\n"
    
    return solution


def process_model_response(mode: str, ctx: Dict[str, Any], raw: str) -> Dict[str, Any]:
    if mode == "solver-solve-then-patch":
        # 2 stage mode
        mutation = ctx.get("mutation", "") or ""
        python_result, diff_result = _process_solve_then_patch(raw, mutation)
        
        
        if diff_result.get("solution"):
            entry = (ctx.get("problem") or {}).get("entry_point", "")
            if entry and entry in mutation and entry not in diff_result["solution"]:
                diff_result["error"] = f"Lost entry point: {entry}"
                diff_result["solution"] = None
            elif "def " in mutation and "def " not in diff_result["solution"]:
                diff_result["error"] = "Lost function definition"
                diff_result["solution"] = None
        
        return {
            "solution": python_result.get("solution"), 
            "python_solution": python_result.get("solution"),
            "python_apply_mode": python_result.get("apply_mode"),
            "diff_solution": diff_result.get("solution"),
            "diff_apply_mode": diff_result.get("apply_mode"),
            "edit": diff_result.get("edit"),
            "diff_error": diff_result.get("diff_error"),
            "failed_patch": diff_result.get("failed_patch"),
            "explanation": _extract_explanation(raw),
            "raw_response": raw,
        }
    
    if "diff" in mode:
        mutation = ctx.get("mutation", "") or ""
        solution, extra = _process_diff(raw, mutation)
        
        if solution:
            entry = (ctx.get("problem") or {}).get("entry_point", "")
            if entry and entry in mutation and entry not in solution:
                extra["error"] = f"Lost entry point: {entry}"
                solution = None
            elif "def " in mutation and "def " not in solution:
                extra["error"] = "Lost function definition"
                solution = None
        
        return {"solution": solution, "explanation": _extract_explanation(raw), "raw_response": raw, **extra}
    
    return {"solution": _process_text(raw, mode), "explanation": _extract_explanation(raw), "raw_response": raw}


def solution_from_saved_value(mode: str, ctx: Dict[str, Any], value: str, prepend_code_prompt: bool = False) -> Dict[str, Any]:
    raw = value or ""
    if prepend_code_prompt:
        prefix = ctx.get("problem", {}).get("complete_prompt") or ctx.get("problem", {}).get("prompt_text") or ""
        raw = f"{prefix}{raw}" if prefix else raw
    return process_model_response(mode, ctx, raw)


def get_verdict(mode: str, timeout: int = 5) -> PassRateMetric:
    instruct_modes = {"complete_prompt", "solver-attacker-style", "solver-test-cases", "solver-solve-then-patch"}
    ds = "bigcodebench-instruct" if ("instruct" in mode or "diff" in mode or mode in instruct_modes) else "bigcodebench-complete"
    return PassRateMetric(dataset=ds, timeout=timeout)


def evaluate_solution(mode: str, problem: Dict[str, Any], solution: Optional[str], verdict: PassRateMetric) -> Tuple[float, Dict[str, Any]]:
    if not solution:
        return 0.0, {"error": "Empty solution"}
    passed, info = verdict(problem=problem, completion=solution)
    return (1.0 if passed else 0.0), {"passed": passed, "verdict_info": info}


def evaluate_two_stage(mode: str, problem: Dict[str, Any], proc: Dict[str, Any], verdict: PassRateMetric) -> Tuple[float, Dict[str, Any]]:
    python_sol = proc.get("python_solution")
    diff_sol = proc.get("diff_solution")
    
    python_score, python_vinfo = evaluate_solution(mode, problem, python_sol, verdict)
    diff_score, diff_vinfo = evaluate_solution(mode, problem, diff_sol, verdict)
    
    return python_score, {
        "python_score": python_score,
        "python_passed": python_vinfo.get("passed", False),
        "python_verdict_info": python_vinfo.get("verdict_info"),
        "diff_score": diff_score,
        "diff_passed": diff_vinfo.get("passed", False),
        "diff_verdict_info": diff_vinfo.get("verdict_info"),
        "python_success": (python_score or 0) > 0,
        "diff_success": (diff_score or 0) > 0,
    }
