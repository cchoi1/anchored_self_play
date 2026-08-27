#!/usr/bin/env python3

import argparse
import json
import os
import sys
import multiprocessing
import random
# import numpy as np
# import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

seed = 42
random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unified_eval.input import dataset_for_mode, load_data, prepare_prompt
from unified_eval.output import evaluate_solution, evaluate_two_stage, get_verdict, process_model_response, solution_from_saved_value



def _save(path: str, results: List[Dict], summary: Dict) -> None:
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

def _make_summary(results: List[Dict], args, evaluate: bool = True, **extra) -> Dict:
    passed = (
        sum(1 for r in results if (r.get("score") or 0) > 0)
        if evaluate
        else None
    )
    summary = {
        "total": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if passed is not None and results else None,
        "model": args.model,
        "mode": args.mode,
        "dataset": dataset_for_mode(args.mode),
        "temperature": args.temperature,
        **extra,
    }
    
    # Add two-stage specific metrics for solve-then-patch mode
    if evaluate and args.mode == "solver-solve-then-patch" and results:
        python_passed = sum(1 for r in results if r.get("python_passed") or (r.get("python_score") or 0) > 0)
        diff_passed = sum(1 for r in results if r.get("diff_passed") or (r.get("diff_score") or 0) > 0)
        summary.update({
            "python_passed": python_passed,
            "python_pass_rate": python_passed / len(results) if results else None,
            "diff_passed": diff_passed,
            "diff_pass_rate": diff_passed / len(results) if results else None,
        })
    
    return summary

def _load_continue_state(path: str) -> tuple:
    with open(path) as f:
        payload = json.load(f)
    results = payload.get("results", payload if isinstance(payload, list) else [])
    if not isinstance(results, list):
        raise ValueError(f"continue file has invalid results payload: {path}")

    by_id, incomplete = {}, set()
    for r in results:
        tid = _row_task_id(r)
        if not tid:
            raise ValueError(f"continue file contains a result without a task ID: {path}")
        if tid in by_id:
            raise ValueError(f"continue file contains duplicate task ID {tid!r}: {path}")
        by_id[tid] = r
        raw = (r.get("raw_response") or "").strip()
        if not raw or r.get("error") or (r.get("solution") is None and not raw):
            incomplete.add(tid)
    return by_id, incomplete


def _row_task_id(row: Dict) -> str:
    return str(row.get("task_id") or row.get("uid") or row.get("1", ""))


def _require_unique_task_ids(rows: List[Dict], source: str) -> List[str]:
    task_ids = [_row_task_id(row) for row in rows]
    missing = [i for i, task_id in enumerate(task_ids) if not task_id]
    if missing:
        raise ValueError(f"{source} has rows without task IDs (first index: {missing[0]})")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{source} contains duplicate task IDs")
    return task_ids


def _pending_rows(data: List[Dict], task_ids: List[str], existing: Dict, incomplete: set) -> List[Dict]:
    """Select rows absent from a checkpoint or explicitly marked incomplete."""
    return [
        row for row, task_id in zip(data, task_ids)
        if task_id not in existing or task_id in incomplete
    ]

# class VLLMRunner:
#     def __init__(self, model: str, tp: int = 1, mem: float = 0.5):
#         from vllm import LLM, SamplingParams
#         self._SamplingParams = SamplingParams
#        
#         max_batched = int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", 65536))
#         max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", 32768))
#         self.llm = LLM(
#             model,
#             tensor_parallel_size=tp,
#             gpu_memory_utilization=mem,
#             trust_remote_code=True,
#             disable_log_stats=True,
#             max_num_batched_tokens=max_batched,
#             max_model_len=max_model_len,
#         )
#
#     def __call__(self, prompt: str, temp: float = 0.7, max_tokens: int = 2048, min_tokens: int = 0) -> str:
#         out = self.llm.generate([prompt], self._SamplingParams(max_tokens=max_tokens, min_tokens=min_tokens, temperature=temp, top_p=0.95))
#         return out[0].outputs[0].text if out else ""
#
#     def batch(self, prompts: List[str], temp: float = 0.7, max_tokens: int = 2048, min_tokens: int = 0) -> List[str]:
#         out = self.llm.generate(prompts, self._SamplingParams(max_tokens=max_tokens, min_tokens=min_tokens, temperature=temp, top_p=0.95))
#         return [o.outputs[0].text if o.outputs else "" for o in out]


def _process_one(row: Dict, args, infer, verdict, evaluate: bool, max_retries: int = 3) -> Dict:
    ctx = prepare_prompt(row, args.mode, args.mutation_col, args.canonical_control, args.examples)
    
    raw = ""
    proc = {"solution": None, "raw_response": ""}
    score, vinfo = (0.0 if evaluate else None), {}
    
    for attempt in range(max_retries):
        try:
            raw = infer(ctx["rendered_prompt"])
            if raw and raw.strip():
                proc = process_model_response(args.mode, ctx, raw)
                
                if evaluate:
                    if args.mode == "solver-solve-then-patch":
                        score, vinfo = evaluate_two_stage(args.mode, ctx["problem"], proc, verdict)
                    else:
                        score, vinfo = evaluate_solution(args.mode, ctx["problem"], proc["solution"], verdict)
                break
        except Exception as e:
            proc = {"solution": None, "raw_response": "", "error": str(e)}
            if attempt == max_retries - 1:
                score, vinfo = 0.0 if evaluate else None, {}
    
    return {
        "task_id": ctx["task_id"], "mode": args.mode, "problem": ctx["problem"],
        "mutation": ctx.get("mutation"), "rendered_prompt": ctx.get("rendered_prompt"),
        "score": score, "attempts": attempt + 1, **proc, **vinfo,
    }


def _eval_one(ctx: Dict, raw: str, args, verdict) -> Dict:
    proc = process_model_response(args.mode, ctx, raw)
    
    if args.mode == "solver-solve-then-patch":
        score, vinfo = evaluate_two_stage(args.mode, ctx["problem"], proc, verdict)
    else:
        score, vinfo = evaluate_solution(args.mode, ctx["problem"], proc["solution"], verdict)
    
    return {
        "task_id": ctx["task_id"], "mode": args.mode, "problem": ctx["problem"],
        "mutation": ctx.get("mutation"), "rendered_prompt": ctx.get("rendered_prompt"),
        "score": score, "success": (score or 0) > 0, **proc, **vinfo,
    }


def run_inference(args) -> List[Dict]:
    data = load_data(args.input)

    if args.offset:
        data = data[args.offset:]
    if args.limit:
        data = data[:args.limit]
    expected_ids = _require_unique_task_ids(data, "input dataset")

    existing = {}
    if args.continue_from and not os.path.exists(args.continue_from):
        raise FileNotFoundError(f"continue file not found: {args.continue_from}")
    if args.continue_from:
        existing, incomplete = _load_continue_state(args.continue_from)
        orig = len(data)
        expected_id_set = set(expected_ids)
        unexpected = set(existing) - expected_id_set
        if unexpected:
            raise ValueError(f"continue file contains task IDs outside the requested input slice: {sorted(unexpected)[:3]}")

        # Retry both explicitly incomplete rows and rows that were never saved
        # (for example, because the first process stopped between checkpoints).
        data = _pending_rows(data, expected_ids, existing, incomplete)

        print(f"Continuing: {len(data)}/{orig} tasks need re-run")
        if not data:
            print("All complete!")
            return [existing[task_id] for task_id in expected_ids]
    
    evaluate = not args.inference_only
    
    # if args.use_vllm:
    #     new = _run_batched(args, data, evaluate)
    # else:
    new = _run_sequential(args, data, evaluate)
    
    if existing:
        for r in new:
            existing[str(r["task_id"])] = r
        missing = [task_id for task_id in expected_ids if task_id not in existing]
        if missing:
            raise RuntimeError(f"inference finished without results for {len(missing)} tasks: {missing[:3]}")
        results = [existing[task_id] for task_id in expected_ids]
        _save(args.output, results, _make_summary(results, args, evaluate, continued_from=args.continue_from))
        if evaluate:
            passed = sum(1 for r in results if (r.get("score") or 0) > 0)
            print(f"Merged: {len(results)} total, {passed} passed")
        else:
            print(f"Merged: {len(results)} total")
        return results
    return new


def _run_sequential(args, data: List[Dict], evaluate: bool) -> List[Dict]:
    verdict = get_verdict(args.mode) if evaluate else None
    from src.utils.api import get_llm_output
    infer = lambda p: get_llm_output(
        p, model=args.model, temperature=args.temperature, max_new_tokens=args.max_new_tokens
    )
    results = []
    
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(_process_one, r, args, infer, verdict, evaluate): r for r in data}
        it = tqdm(as_completed(futures), total=len(futures), desc="Inference") if tqdm else as_completed(futures)
        for i, fut in enumerate(it, 1):
            results.append(fut.result())
            if i % 10 == 0 or i == len(futures):
                _save(args.output, results, _make_summary(results, args, evaluate))
    
    if evaluate:
        passed = sum(1 for r in results if (r.get("score") or 0) > 0)
        print(f"Done: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    else:
        print(f"Saved {len(results)} responses")
    return results


# def _run_batched(args, data: List[Dict], evaluate: bool, max_retries: int = 3) -> List[Dict]:
#     print(f"Preparing {len(data)} prompts...")
#     contexts = [prepare_prompt(r, args.mode, args.mutation_col, args.canonical_control, args.examples) for r in data]
#     prompts = [c["rendered_prompt"] for c in contexts]
#     
#     print(f"Generating with vLLM...")
#     runner = VLLMRunner(args.model, args.tensor_parallel)
#     responses = runner.batch(prompts, args.temperature, args.max_new_tokens, args.min_tokens)
#     
#     # Retry empty responses up to max_retries times
#     for attempt in range(1, max_retries):
#         empty_idx = [i for i, r in enumerate(responses) if not r or not r.strip()]
#         if not empty_idx:
#             break
#         print(f"Retry {attempt}: {len(empty_idx)} empty responses...")
#         retry_prompts = [prompts[i] for i in empty_idx]
#         retry_responses = runner.batch(retry_prompts, args.temperature, args.max_new_tokens, args.min_tokens)
#         for idx, new_resp in zip(empty_idx, retry_responses):
#             if new_resp and new_resp.strip():
#                 responses[idx] = new_resp
#     
#     empty_count = sum(1 for r in responses if not r or not r.strip())
#     print(f"Done generating ({empty_count} still empty after retries)")
#     
#     # Save raw
#     raw_results = [{
#         "task_id": c["task_id"], "mode": args.mode, "problem": c["problem"],
#         "mutation": c.get("mutation"), "rendered_prompt": c.get("rendered_prompt"),
#         "raw_response": r,
#     } for c, r in zip(contexts, responses)]
#     
#     if not evaluate:
#         _save(args.output, raw_results, _make_summary(raw_results, args, False, inference_only=True))
#         print(f"Saved {len(raw_results)} raw responses")
#         return raw_results
#     
#     # Save raw backup
#     raw_path = args.output.replace(".json", "_raw.json")
#     with open(raw_path, "w") as f:
#         json.dump({"total": len(raw_results), "model": args.model, "results": raw_results}, f, indent=2)
#     print(f"Raw backup: {raw_path}")
#     
#     # Evaluate
#     print(f"Evaluating...")
#     verdict = get_verdict(args.mode)
#     results = []
#     
#     with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
#         futures = {ex.submit(_eval_one, c, r, args, verdict): i for i, (c, r) in enumerate(zip(contexts, responses))}
#         it = tqdm(as_completed(futures), total=len(futures), desc="Eval") if tqdm else as_completed(futures)
#         for i, fut in enumerate(it, 1):
#             try:
#                 results.append(fut.result())
#             except Exception as e:
#                 idx = futures[fut]
#                 results.append({
#                     "task_id": contexts[idx]["task_id"], "mode": args.mode,
#                     "problem": contexts[idx]["problem"], "score": 0.0, "error": str(e),
#                 })
#             if i % 10 == 0 or i == len(futures):
#                 _save(args.output, results, _make_summary(results, args))
#                 passed = sum(1 for r in results if (r.get("score") or 0) > 0)
#                 print(f"[{i}/{len(futures)}] {passed}/{len(results)}")
#     
#     passed = sum(1 for r in results if (r.get("score") or 0) > 0)
#     print(f"Done: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
#     return results


def _eval_json_entry(entry: Dict, rows_by_id: Dict, mode: str, mut_col: str, canon: bool, examples: bool) -> Dict:
    tid = str(entry.get("task_id", entry.get("problem", {}).get("task_id", "unknown")))
    row = rows_by_id.get(tid, {})
    m = entry.get("mode", mode)
    
    if entry.get("problem") or entry.get("mutation") or entry.get("rendered_prompt"):
        ctx = {
            "task_id": tid, "handler": None, "mode": m,
            "rendered_prompt": entry.get("rendered_prompt") or entry.get("problem", {}).get("prompt_text", ""),
            "prompt_text": entry.get("prompt_text", ""),
            "mutation": entry.get("mutation", ""),
            "problem": entry.get("problem", {}),
        }
    else:
        ctx = prepare_prompt(row, m, mut_col, canon, examples)
    
    raw = entry.get("raw_response") or entry.get("solution") or ""
    proc = process_model_response(m, ctx, raw)
    verdict = get_verdict(m)
    
    if m == "solver-solve-then-patch":
        score, vinfo = evaluate_two_stage(m, ctx["problem"], proc, verdict)
    else:
        score, vinfo = evaluate_solution(m, ctx["problem"], proc["solution"], verdict)
    
    return {"task_id": tid, "mode": m, "problem": ctx["problem"], "mutation": ctx.get("mutation"),
            **proc, **vinfo, "score": score}


def _eval_column_row(row: Dict, mode: str, mut_col: str, canon: bool, col: str, prepend: bool, examples: bool) -> Dict:
    ctx = prepare_prompt(row, mode, mut_col, canon, examples)
    proc = solution_from_saved_value(mode, ctx, row.get(col, "") or "", prepend_code_prompt=prepend)
    verdict = get_verdict(mode)
    
    if mode == "solver-solve-then-patch":
        score, vinfo = evaluate_two_stage(mode, ctx["problem"], proc, verdict)
    else:
        score, vinfo = evaluate_solution(mode, ctx["problem"], proc["solution"], verdict)
    
    return {"task_id": ctx["task_id"], "mode": mode, "problem": ctx["problem"], "mutation": ctx.get("mutation"),
            "source_column": col, **proc, **vinfo, "score": score}


def run_eval_only(args) -> List[Dict]:
    data = load_data(args.input)
    if args.offset:
        data = data[args.offset:]
    if args.limit:
        data = data[:args.limit]
    
    target = args.eval
    if target and os.path.exists(target):
        # Eval from JSON
        with open(target) as f:
            payload = json.load(f)
        entries = payload.get("results", payload)
        expected_ids = _require_unique_task_ids(data, "input dataset")
        entry_ids = _require_unique_task_ids(entries, "evaluation JSON")
        missing = set(expected_ids) - set(entry_ids)
        unexpected = set(entry_ids) - set(expected_ids)
        if missing or unexpected:
            raise ValueError(
                "evaluation JSON does not exactly cover the requested dataset slice: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        rows_by_id = {_row_task_id(r): r for r in data}
        
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_eval_json_entry, e, rows_by_id, args.mode, 
                                  args.mutation_col, args.canonical_control, args.examples)
                        for e in entries]
                it = as_completed(futs)
                results = [f.result() for f in (tqdm(it, total=len(futs), desc="eval") if tqdm else it)]
        else:
            it = (_eval_json_entry(e, rows_by_id, args.mode, args.mutation_col, args.canonical_control, args.examples)
                  for e in entries)
            results = list(tqdm(it, total=len(entries), desc="eval") if tqdm else it)
    else:
        # Eval from column
        if not data or target not in data[0]:
            raise FileNotFoundError(
                f"--eval must be an existing JSON file or a dataset column; not found: {target!r}"
            )
        prepend = "instruct" not in args.mode and target in {"canonical_solution", "buggy", "mutation", "response"}
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_eval_column_row, r, args.mode, target, 
                                  args.canonical_control, target, prepend, args.examples)
                        for r in data]
                it = as_completed(futs)
                results = [f.result() for f in (tqdm(it, total=len(futs), desc="eval") if tqdm else it)]
        else:
            it = (_eval_column_row(r, args.mode, target, args.canonical_control, target, prepend, args.examples)
                  for r in data)
            results = list(tqdm(it, total=len(data), desc="eval") if tqdm else it)
    
    passed = sum(1 for r in results if (r.get("score") or 0) > 0)
    _save(args.output, results, {
        "total": len(results), "passed": passed,
        "pass_rate": passed / len(results) if results else 0,
        "mode": args.mode, "dataset": dataset_for_mode(args.mode),
        "model": args.model, "eval_only": True, "source": target,
    })
    print(f"Eval done: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Unified eval runner")
    p.add_argument("--input", default="cchoi1/adversarial_bugbench_bcb")
    p.add_argument("--output", default="unified_results.json")
    p.add_argument("--mode", default="solver-ambig-instruct")
    p.add_argument("--model", default="gpt-4o")
    # p.add_argument("--use-vllm", action="store_true")
    # p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--min-tokens", type=int, default=0, help="Minimum tokens to generate (prevents early EOS)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--mutation-col", default="buggy")
    p.add_argument("--canonical-control", action="store_true")
    p.add_argument("--eval", default=None, help="Eval from JSON or column")
    p.add_argument("--inference-only", action="store_true")
    p.add_argument("--progress-interval", type=int, default=100)
    p.add_argument("--no-examples", action="store_true")
    p.add_argument("--continue-from", default=None, help="Resume from previous output")
    
    args = p.parse_args()
    if args.eval and args.inference_only:
        p.error("--eval and --inference-only conflict")
    args.examples = not args.no_examples
    return args


def main():
    args = parse_args()
    if args.eval:
        run_eval_only(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
