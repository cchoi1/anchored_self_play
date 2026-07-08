import argparse
import asyncio
import json
import os
from datetime import datetime

from transformers import AutoTokenizer

from examples.asp.data_utils import load_data
from examples.asp.fixer_flow import FixerWorkflow
from examples.asp.utils import _get_pregenerated_bug
from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import code_reward_fn


def evaluate_results(results):
    """Evaluate results and compute summary metrics."""
    total_episodes = len(results)
    if total_episodes == 0:
        print("No results to evaluate.")
        return

    synthesizer_pass = 0
    synthesizer_evaluated = 0  # Count of episodes where synthesizer was actually evaluated
    fix_called = 0
    fix_pass = 0
    fixer_reward = 0

    for ep in results:
        m = ep.metrics or {}
        synth_pass_val = m.get("synthesizer_pass")
        if synth_pass_val is not None:
            synthesizer_evaluated += 1
            synthesizer_pass += int(synth_pass_val)
        fix_called += int(m.get("fix_called", 0) or (1 if ep.trajectories else 0))
        fix_pass += int(float(m.get("fixer_pass", 0.0)) > 0.0)

    print("\n" + "=" * 80)
    print("📊 FIXER WORKFLOW RESULTS (frozen synthesizer -> trainable fixer)")
    print("=" * 80)
    print(f"Total episodes: {total_episodes}")
    print("\nOverall Metrics:")
    if synthesizer_evaluated > 0:
        print(f"  Synthesizer Pass Rate: {synthesizer_pass}/{synthesizer_evaluated} ({100*synthesizer_pass/synthesizer_evaluated:.1f}%) [evaluated on {synthesizer_evaluated}/{total_episodes} episodes]")
    else:
        print(f"  Synthesizer Pass Rate: (not evaluated - using pregenerated bugs)")
    print(f"  Fix Called Rate: {fix_called}/{total_episodes} ({100*fix_called/total_episodes:.1f}%)")
    if fix_called > 0:
        print(f"  Fix Pass Rate:   {fix_pass}/{fix_called} ({100*fix_pass/fix_called:.1f}%)")
    print("=" * 80)


def exclude_token_ids(data):
    """Recursively remove prompt_ids and completion_ids from the data structure."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key not in ["prompt_ids", "completion_ids"]:
                result[key] = exclude_token_ids(value)
        return result
    if isinstance(data, list):
        return [exclude_token_ids(item) for item in data]
    return data


def print_sample_episodes(results, n_samples=3):
    """Print some sample episodes with synthesizer code + fixer output."""
    print("\n" + "=" * 80)
    print(f"📝 SAMPLE EPISODES (showing first {min(n_samples, len(results))})")
    print("=" * 80)

    for idx, ep in enumerate(results[:n_samples]):
        print(f"\n--- Episode {idx + 1} ---")
        print(f"Task ID: {ep.id}")
        print(f"Question: {str(ep.task.get('question', ''))[:200]}...")

        info = ep.info or {}
        synthesizer_code = info.get("buggy_code", "")
        print("\n🧊 Frozen Synthesizer Code (first 400 chars):")
        print(synthesizer_code[:400] + "..." if len(synthesizer_code) > 400 else synthesizer_code)

        fixer_traj = ep.trajectories[0] if ep.trajectories else None
        if fixer_traj and fixer_traj.steps:
            fixed = fixer_traj.steps[0].action or ""
            print("\n🔧 Fixer Output (first 400 chars):")
            print(fixed[:400] + "..." if len(fixed) > 400 else fixed)
            print(f"Reward: {fixer_traj.steps[0].reward}")

        print(f"\nMetrics: {ep.metrics}")
        print(f"Episode Correct: {ep.is_correct}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fixer workflow (repair frozen synthesizer failures) on tasks")
    parser.add_argument("--n_tasks", type=int, default=-1, help="Number of tasks to process (default: -1 = all)")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of times to repeat each task (default: 1)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (default: test)")
    parser.add_argument("--dataset", type=str, default="bigcodebench", help="Dataset: deepcoder, bigcodebench, bugbench, etc.")

    # Fixer (trainable model, but here just for evaluation)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Fixer model name")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1", help="Fixer base URL")
    parser.add_argument("--api_key", type=str, default="None", help="Fixer API key (default: None)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Fixer sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Fixer sampling top_p")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--max_response_length", type=int, default=2048, help="Max response length")
    parser.add_argument("--fixer_system_prompt", type=str, default=None, help="Optional system prompt for fixer")

    # Frozen synthesizer (OpenAI-compatible; chat endpoint, no tokenizer required)
    parser.add_argument("--synthesizer_model", type=str, required=True, help="Frozen synthesizer model name")
    parser.add_argument("--synthesizer_base_url", type=str, default="http://localhost:30001/v1", help="Frozen synthesizer base URL")
    parser.add_argument("--synthesizer_api_key", type=str, default=None, help="Frozen synthesizer API key (default: OPENAI_API_KEY or dummy)")
    parser.add_argument("--synthesizer_temperature", type=float, default=0.6, help="Frozen synthesizer sampling temperature")
    parser.add_argument("--synthesizer_top_p", type=float, default=0.95, help="Frozen synthesizer sampling top_p")
    parser.add_argument("--synthesizer_system_prompt", type=str, default=None, help="Optional system prompt for frozen synthesizer")

    # Training behavior (still affects evaluation workflow semantics)
    parser.add_argument("--only_train_on_failures", action="store_true", default=True, help="If set, synthesizer passes yield empty trajectories")
    parser.add_argument("--reward_pm1", action="store_true", default=False, help="Use {-1,+1} reward instead of {0,1}")
    parser.add_argument("--include_failed_test_output", action="store_true", default=True, help="Include unit test failure output in fixer prompt")
    parser.add_argument("--max_failed_test_output_chars", type=int, default=4000, help="Max chars of failure output to include in prompt")

    # Execution / logging
    parser.add_argument("--eval_pregenerated_only", action="store_true",
                        help="Only evaluate on tasks with pregenerated bugs (skip tasks without bugs)")
    parser.add_argument("--val_datasets", nargs="+", default=None,
                        help="Multiple validation datasets in format 'alias=dataset:split' or 'dataset:split' (e.g., 'bugbench:test' 'bcb=bigcodebench:test')")
    parser.add_argument("--n_parallel", type=int, default=32, help="Number of parallel tasks")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to save results")
    parser.add_argument("--print_samples", type=int, default=3, help="Number of sample episodes to print")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    print(f"Loading tokenizer for fixer model: {args.model}")
    fixer_tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Initializing fixer rollout engine...")
    fixer_engine = OpenAIEngine(
        model=args.model,
        tokenizer=fixer_tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        base_url=args.base_url,
        api_key=args.api_key,
        sampling_params={
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
        },
    )
    
    # IMPORTANT: Set validate mode for eval_pregenerated_only (workflow uses pregenerated bugs in validation mode)
    if args.eval_pregenerated_only:
        setattr(fixer_engine, "validate", True)
        print(f"  Set rollout_engine.validate = True (eval mode)")

    api_key = args.synthesizer_api_key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "") or ""
    api_key = str(api_key).strip()
    is_openai_api = "api.openai.com" in str(args.synthesizer_base_url)
    if is_openai_api and not api_key:
        raise ValueError("synthesizer_base_url points to OpenAI API but OPENAI_API_KEY is missing/empty.")
    if not api_key:
        api_key = "EMPTY"

    print("Initializing frozen synthesizer engine...")
    synthesizer_engine = OpenAIEngine(
        model=args.synthesizer_model,
        tokenizer=None,  # frozen synthesizer; chat-completions endpoint
        base_url=args.synthesizer_base_url,
        api_key=api_key,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        sampling_params={
            "temperature": float(args.synthesizer_temperature),
            "top_p": float(args.synthesizer_top_p),
        },
        verbose=False,
    )

    print("Creating workflow engine...")
    engine = AgentWorkflowEngine(
        workflow_cls=FixerWorkflow,
        workflow_args={
            "reward_function": code_reward_fn,
            "synthesizer_rollout_engine": synthesizer_engine,
            "synthesizer_system_prompt": args.synthesizer_system_prompt,
            "fixer_system_prompt": args.fixer_system_prompt,
            "only_train_on_failures": bool(args.only_train_on_failures),
            "reward_pm1": bool(args.reward_pm1),
            "include_failed_test_output": bool(args.include_failed_test_output),
            "max_failed_test_output_chars": int(args.max_failed_test_output_chars),
        },
        rollout_engine=fixer_engine,
        config=None,
        n_parallel_tasks=args.n_parallel,
        retry_limit=1,
    )

    # Parse val_datasets if provided
    val_dataset_specs = {}
    if args.val_datasets:
        for spec in args.val_datasets:
            alias = None
            if "=" in spec:
                alias, spec = spec.split("=", 1)
                alias = alias.strip() or None
                spec = spec.strip()
            
            if ":" in spec:
                ds_name, split = spec.split(":", 1)
                ds_name = ds_name.strip()
                split = split.strip()
            else:
                ds_name, split = spec, "test"
            
            if not alias:
                alias = f"{ds_name}_{split}"
            val_dataset_specs[alias] = (ds_name, split)
    
    # Load tasks - either single dataset or multiple val datasets
    if val_dataset_specs:
        # Multiple validation datasets mode
        print(f"\n📊 Loading {len(val_dataset_specs)} validation datasets...")
        all_results = {}
        
        for alias, (ds_name, split) in val_dataset_specs.items():
            print(f"\n{'='*80}")
            print(f"Evaluating on: {alias} ({ds_name}:{split})")
            print(f"{'='*80}")
            
            tasks = load_data(dataset_name=ds_name, split=split, n=args.n_repeats)
            if not tasks:
                print(f"  WARNING: No tasks loaded from {ds_name}:{split}. Skipping.")
                continue
            
            # Filter to only pregenerated bugs if eval mode is enabled
            if args.eval_pregenerated_only:
                original_count = len(tasks)
                tasks = [task for task in tasks if _get_pregenerated_bug(task) is not None]
                filtered_count = len(tasks)
                print(f"  Filtered to tasks with pregenerated bugs: {filtered_count}/{original_count} tasks")
                if filtered_count == 0:
                    print(f"  WARNING: No tasks with pregenerated bugs found. Skipping {alias}.")
                    continue
            
            if args.n_tasks > 0:
                tasks = tasks[: args.n_tasks]
            
            print(f"  Loaded {len(tasks)} tasks")
            print(f"  Configuration:")
            print(f"    Fixer Model: {args.model}")
            print(f"    Frozen Synthesizer Model: {args.synthesizer_model}")
            print(f"    Parallel tasks: {args.n_parallel}")
            print(f"    Eval pregenerated only: {args.eval_pregenerated_only}")
            
            print(f"\n  🚀 Executing workflow on {len(tasks)} tasks...")
            results = asyncio.run(engine.execute_tasks(tasks))
            
            print(f"\n  📊 Results for {alias}:")
            evaluate_results(results)
            
            all_results[alias] = results
            
            if args.print_samples > 0:
                print(f"\n  📝 Sample episodes for {alias}:")
                print_sample_episodes(results, n_samples=args.print_samples)
        
        # Summary across all datasets
        print("\n" + "=" * 80)
        print("📊 SUMMARY ACROSS ALL VALIDATION DATASETS")
        print("=" * 80)
        for alias, results in all_results.items():
            total = len(results)
            if total == 0:
                continue
            fixer_pass = sum(1 for ep in results if float((ep.metrics or {}).get("fixer_pass", 0.0)) > 0.0)
            # Only count synthesizer_pass when it was actually evaluated (not None)
            synthesizer_results = [(ep.metrics or {}).get("synthesizer_pass") for ep in results]
            synthesizer_evaluated = sum(1 for v in synthesizer_results if v is not None)
            synthesizer_pass = sum(1 for v in synthesizer_results if v and v > 0)
            if synthesizer_evaluated > 0:
                print(f"  {alias}: {fixer_pass}/{total} fixer pass ({100*fixer_pass/total:.1f}%), "
                      f"{synthesizer_pass}/{synthesizer_evaluated} synthesizer pass ({100*synthesizer_pass/synthesizer_evaluated:.1f}% of {synthesizer_evaluated} evaluated)")
            else:
                print(f"  {alias}: {fixer_pass}/{total} fixer pass ({100*fixer_pass/total:.1f}%), "
                      f"synthesizer pass (not evaluated)")
        
        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for alias, results in all_results.items():
                output_file = os.path.join(args.output_dir, f"fixer_flow_{alias}_{timestamp}.json")
                results_dict = [exclude_token_ids(ep.to_dict()) for ep in results]
                with open(output_file, "w") as f:
                    json.dump(results_dict, f, indent=2)
                print(f"\n💾 Results for {alias} saved to: {output_file}")
        
    else:
        # Single dataset mode (original behavior)
        print(f"\nLoading tasks from dataset '{args.dataset}' split '{args.split}'...")
        all_tasks = load_data(dataset_name=args.dataset, split=args.split, n=args.n_repeats)
        if not all_tasks:
            print("No tasks loaded. Exiting.")
            raise SystemExit(1)

        # Filter to only pregenerated bugs if eval mode is enabled
        if args.eval_pregenerated_only:
            original_count = len(all_tasks)
            all_tasks = [task for task in all_tasks if _get_pregenerated_bug(task) is not None]
            filtered_count = len(all_tasks)
            print(f"  Filtered to tasks with pregenerated bugs: {filtered_count}/{original_count} tasks")
            if filtered_count == 0:
                print("  ERROR: No tasks with pregenerated bugs found. Exiting.")
                raise SystemExit(1)

        if args.n_tasks > 0:
            all_tasks = all_tasks[: args.n_tasks]

        print(f"Loaded {len(all_tasks)} tasks")
        print("Configuration:")
        print(f"  Fixer Model: {args.model}")
        print(f"  Frozen Synthesizer Model: {args.synthesizer_model}")
        print(f"  Parallel tasks: {args.n_parallel}")
        print(f"  only_train_on_failures: {bool(args.only_train_on_failures)}")
        print(f"  include_failed_test_output: {bool(args.include_failed_test_output)}")
        print(f"  Eval pregenerated only: {args.eval_pregenerated_only}")

        print(f"\n🚀 Executing workflow on {len(all_tasks)} tasks...")
        results = asyncio.run(engine.execute_tasks(all_tasks))

        print("\n📊 Evaluating results...")
        evaluate_results(results)

        if args.print_samples > 0:
            print_sample_episodes(results, n_samples=args.print_samples)

        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.output_dir, f"fixer_flow_results_{timestamp}.json")
            results_dict = [exclude_token_ids(ep.to_dict()) for ep in results]
            with open(output_file, "w") as f:
                json.dump(results_dict, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")

    print("\n✅ Done!")
