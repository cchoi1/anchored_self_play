import argparse
import asyncio
import json
import os
from datetime import datetime

from transformers import AutoTokenizer

from examples.asp.data_utils import load_data
from examples.asp.generator_fixer_flow import GeneratorFixerWorkflow
from examples.asp.utils import _get_pregenerated_bug
from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.rewards.reward_fn import code_reward_fn


def evaluate_results(results, dataset_alias=None):
    """Evaluate the results and compute metrics.
    
    Args:
        results: List of Episode objects to evaluate.
        dataset_alias: Optional dataset name for display purposes.
    
    Returns:
        dict: Summary of key metrics.
    """
    from collections import defaultdict

    total_episodes = len(results)
    if total_episodes == 0:
        print("No results to evaluate.")
        return

    # Aggregate metrics
    bug_valid_count = 0
    generator_reward_count = 0
    fixer_reward_count = 0

    codegen_present_count = 0
    fixer_codegen_pass_count = 0

    bug_total_tests_sum = 0
    bug_passed_tests_sum = 0
    bug_test_counts = []
    
    # Code embedding similarity metrics
    code_embed_scores = []
    code_embed_count = 0

    problem_stats = defaultdict(
        lambda: {
            "total": 0,
            "bug_valid": 0,
            "generator_reward": 0,
            "fixer_reward": 0,
            "fixer_codegen_present": 0,
            "fixer_codegen_pass": 0,
        }
    )

    for episode in results:
        metrics = episode.metrics or {}
        problem = episode.task.get("question", "")[:100]  # Truncate for display

        bug_valid_count += int(float(metrics.get("bug_valid", 0.0)) > 0.0)
        # generator_reward can be negative (penalties), so check if > 0 to count positively rewarded episodes
        generator_reward_count += int(float(metrics.get("generator_reward", 0.0)) > 0.0)
        # fixer_pass is a float (solve_rate), so check if > 0 to count episodes where fixer passed at least once
        fixer_reward_count += int(float(metrics.get("fixer_pass", 0.0)) > 0.0)

        if "fixer_codegen_pass" in metrics:
            codegen_present_count += 1
            fixer_codegen_pass_count += int(metrics.get("fixer_codegen_pass", 0))

        if "bug_total_tests" in metrics:
            bug_total_tests_sum += metrics["bug_total_tests"]
        if "bug_passed_tests" in metrics:
            bug_passed_tests_sum += metrics["bug_passed_tests"]
            bug_test_counts.append(metrics["bug_passed_tests"])
        
        # Code embedding similarity metrics
        if "code_embed_score" in metrics:
            code_embed_scores.append(metrics["code_embed_score"])
            code_embed_count += 1

        # Per-problem stats
        problem_stats[problem]["total"] += 1
        problem_stats[problem]["bug_valid"] += int(float(metrics.get("bug_valid", 0.0)) > 0.0)
        problem_stats[problem]["generator_reward"] += int(float(metrics.get("generator_reward", 0.0)) > 0.0)
        problem_stats[problem]["fixer_reward"] += int(float(metrics.get("fixer_pass", 0.0)) > 0.0)
        if "fixer_codegen_pass" in metrics:
            problem_stats[problem]["fixer_codegen_present"] += 1
            problem_stats[problem]["fixer_codegen_pass"] += int(metrics.get("fixer_codegen_pass", 0))

    # Print summary statistics
    print("\n" + "=" * 80)
    header = "📊 GENERATOR-FIXER WORKFLOW RESULTS"
    if dataset_alias:
        header += f" [{dataset_alias}]"
    print(header)
    print("=" * 80)
    print(f"Total episodes: {total_episodes}")
    print("\nOverall Metrics:")
    print(f"  Generator Reward Rate: {generator_reward_count}/{total_episodes} ({100*generator_reward_count/total_episodes:.1f}%)")
    print(f"  Fixer Reward Rate (Bug Fixing): {fixer_reward_count}/{total_episodes} ({100*fixer_reward_count/total_episodes:.1f}%)")
    print(f"  Bug Valid Rate: {bug_valid_count}/{total_episodes} ({100*bug_valid_count/total_episodes:.1f}%)")

    # Always print codegen stats if present (even if pass count is 0)
    if codegen_present_count > 0:
        print(
            f"  Fixer CodeGen Pass Rate: {fixer_codegen_pass_count}/{codegen_present_count} "
            f"({100*fixer_codegen_pass_count/codegen_present_count:.1f}%)  [on episodes with codegen]"
        )
    else:
        print("  Fixer CodeGen Pass Rate: (not computed)  [did you set rollout_engine.validate=True?]")

    if bug_total_tests_sum > 0:
        avg_total_tests = bug_total_tests_sum / total_episodes
        avg_passed_tests = bug_passed_tests_sum / total_episodes
        print("\nBug Test Statistics:")
        print(f"  Average total tests per bug: {avg_total_tests:.2f}")
        print(f"  Average passed tests per bug: {avg_passed_tests:.2f}")
        print(f"  Average failed tests per bug: {avg_total_tests - avg_passed_tests:.2f}")

    # Code embedding similarity statistics
    if code_embed_count > 0:
        avg_embed = sum(code_embed_scores) / len(code_embed_scores)
        sorted_scores = sorted(code_embed_scores)
        median_embed = sorted_scores[len(sorted_scores) // 2]
        q1 = sorted_scores[len(sorted_scores) // 4] if len(sorted_scores) >= 4 else sorted_scores[0]
        q3 = sorted_scores[3 * len(sorted_scores) // 4] if len(sorted_scores) >= 4 else sorted_scores[-1]
        print("\n🔢 Code Embedding Similarity Statistics:")
        print(f"  Episodes with embedding scores: {code_embed_count}/{total_episodes}")
        print(f"  Average: {avg_embed:.3f}  |  Median: {median_embed:.3f}")
        print(f"  Min: {min(code_embed_scores):.3f}  |  Q1: {q1:.3f}  |  Q3: {q3:.3f}  |  Max: {max(code_embed_scores):.3f}")
        # Show distribution buckets
        buckets = [0, 0, 0, 0, 0]  # [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
        for score in code_embed_scores:
            bucket_idx = min(int(score * 5), 4)
            buckets[bucket_idx] += 1
        print(f"  Distribution: [0-0.2]: {buckets[0]}, [0.2-0.4]: {buckets[1]}, [0.4-0.6]: {buckets[2]}, [0.6-0.8]: {buckets[3]}, [0.8-1.0]: {buckets[4]}")

    print("=" * 80)

    # Return summary dict
    summary = {
        "dataset": dataset_alias or "unknown",
        "total_episodes": total_episodes,
        "generator_reward_rate": round(100 * generator_reward_count / total_episodes, 1) if total_episodes > 0 else 0,
        "fixer_reward_rate": round(100 * fixer_reward_count / total_episodes, 1) if total_episodes > 0 else 0,
        "bug_valid_rate": round(100 * bug_valid_count / total_episodes, 1) if total_episodes > 0 else 0,
        "fixer_codegen_pass_rate": round(100 * fixer_codegen_pass_count / codegen_present_count, 1) if codegen_present_count > 0 else None,
        "generator_reward_count": generator_reward_count,
        "fixer_reward_count": fixer_reward_count,
        "bug_valid_count": bug_valid_count,
        "fixer_codegen_pass_count": fixer_codegen_pass_count,
        "codegen_present_count": codegen_present_count,
    }
    return summary


def evaluate_results_grouped(results):
    """Evaluate results grouped by dataset alias.
    
    Args:
        results: List of Episode objects, each with task containing '_dataset_alias'.
    
    Returns:
        Tuple of (grouped_results, summaries) where:
        - grouped_results: Dict mapping dataset_alias to list of Episode objects.
        - summaries: Dict mapping dataset_alias to summary dict.
    """
    from collections import defaultdict
    
    # Group results by dataset alias
    grouped = defaultdict(list)
    for episode in results:
        alias = episode.task.get("_dataset_alias", "unknown")
        grouped[alias].append(episode)
    
    # Evaluate each group and collect summaries
    summaries = {}
    for alias, group_results in grouped.items():
        summary = evaluate_results(group_results, dataset_alias=alias)
        summaries[alias] = summary
    
    return dict(grouped), summaries


def exclude_token_ids(data):
    """Recursively remove prompt_ids and completion_ids from the data structure."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key not in ["prompt_ids", "completion_ids"]:
                result[key] = exclude_token_ids(value)
        return result
    elif isinstance(data, list):
        return [exclude_token_ids(item) for item in data]
    else:
        return data


def save_embedding_pools(target_pool=None, negative_pool=None, save_target_path=None, save_negative_path=None):
    """Save embedding pools to disk.
    
    Args:
        target_pool: Pre-built target ReferencePool object
        negative_pool: Pre-built negative ReferencePool object
        save_target_path: Path prefix for target pool (creates {path}_embeddings.npy and {path}_metadata.json)
        save_negative_path: Path prefix for negative pool (creates {path}_embeddings.npy and {path}_metadata.json)
    """
    if not save_target_path and not save_negative_path:
        return
    
    # Save target pool
    if save_target_path:
        if target_pool is not None and len(target_pool) > 0:
            try:
                os.makedirs(os.path.dirname(save_target_path) or '.', exist_ok=True)
                target_pool.save(save_target_path)
                print(f"\n💾 Saved TARGET embedding pool to:")
                print(f"   {save_target_path}_embeddings.npy")
                print(f"   {save_target_path}_metadata.json")
                print(f"   Pool size: {len(target_pool)} embeddings")
            except Exception as e:
                print(f"  Error saving target pool: {e}")
        else:
            print("  Warning: No target pool to save")
    
    # Save negative pool
    if save_negative_path:
        if negative_pool is not None and len(negative_pool) > 0:
            try:
                os.makedirs(os.path.dirname(save_negative_path) or '.', exist_ok=True)
                negative_pool.save(save_negative_path)
                print(f"\n💾 Saved NEGATIVE embedding pool to:")
                print(f"   {save_negative_path}_embeddings.npy")
                print(f"   {save_negative_path}_metadata.json")
                print(f"   Pool size: {len(negative_pool)} embeddings")
            except Exception as e:
                print(f"  Error saving negative pool: {e}")
        else:
            print("  Warning: No negative pool to save")


def print_sample_episodes(results, n_samples=3):
    """Print detailed information for a few sample episodes."""
    print("\n" + "=" * 80)
    print(f"📝 SAMPLE EPISODES (showing first {min(n_samples, len(results))})")
    print("=" * 80)

    for idx, episode in enumerate(results[:n_samples]):
        print(f"\n--- Episode {idx + 1} ---")
        print(f"Task ID: {episode.id}")
        print(f"Question: {episode.task.get('question', '')[:200]}...")

        bug_traj = None
        fixer_traj = None
        codegen_traj = None
        for traj in episode.trajectories:
            if traj.name == "bug_generator":
                bug_traj = traj
            elif traj.name.startswith("bug_fixer"):
                fixer_traj = traj  # Use first fixer trajectory
            elif traj.name == "code_generator":
                codegen_traj = traj

        if bug_traj and bug_traj.steps:
            bug_code = bug_traj.steps[0].action
            print("\n🐛 Bug Generator Output (first 300 chars):")
            print(bug_code[:300] + "..." if len(bug_code) > 300 else bug_code)
            print(f"Reward: {bug_traj.steps[0].reward}")

        if fixer_traj and fixer_traj.steps:
            fixed_code = fixer_traj.steps[0].action
            print("\n🔧 Bug Fixer Output (first 300 chars):")
            print(fixed_code[:300] + "..." if len(fixed_code) > 300 else fixed_code)
            print(f"Reward: {fixer_traj.steps[0].reward}")

        if codegen_traj and codegen_traj.steps:
            generated_code = codegen_traj.steps[0].action
            print("\n💻 Code Generator Output (first 300 chars):")
            print(generated_code[:300] + "..." if len(generated_code) > 300 else generated_code)
            codegen_pass = int((episode.metrics or {}).get("fixer_codegen_pass", 0))
            print(f"Pass: {bool(codegen_pass)}")

        # Print code embedding similarity score prominently
        metrics = episode.metrics or {}
        if "code_embed_score" in metrics:
            score = metrics["code_embed_score"]
            # Visual bar representation
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"\n🔢 Code Embedding Similarity: {score:.3f} [{bar}]")
            if "code_embed_target_sim" in metrics:
                print(f"   Target similarity: {metrics['code_embed_target_sim']:.3f}")
            if "code_embed_negative_sim" in metrics:
                print(f"   Negative similarity: {metrics['code_embed_negative_sim']:.3f}")
            if "code_embed_margin" in metrics:
                print(f"   Margin (target - negative): {metrics['code_embed_margin']:.3f}")

        print(f"\nMetrics: {metrics}")
        print(f"Episode Correct: {episode.is_correct}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generator-Fixer workflow on tasks")
    parser.add_argument("--n_tasks", type=int, default=-1, help="Number of tasks to process (default: -1)")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of times to repeat each task (default: 1)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model name for generator (default: Qwen/Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1", help="Base URL for generator API (default: http://localhost:30000/v1)")
    parser.add_argument("--api_key", type=str, default="None", help="API key for generator (default: None)")
    
    # Optional separate fixer model/server
    parser.add_argument("--fixer_model", type=str, default=None, 
                        help="Model name for fixer (default: same as --model)")
    parser.add_argument("--fixer_base_url", type=str, default=None,
                        help="Base URL for fixer API (default: same as --base_url)")
    parser.add_argument("--fixer_api_key", type=str, default=None,
                        help="API key for fixer (default: same as --api_key)")
    parser.add_argument("--fixer_temperature", type=float, default=None,
                        help="Sampling temperature for fixer (default: same as --temperature)")
    parser.add_argument("--fixer_top_p", type=float, default=None,
                        help="Sampling top_p for fixer (default: same as --top_p)")
    
    parser.add_argument("--n_parallel", type=int, default=32, help="Number of parallel tasks (default: 32)")
    parser.add_argument("--max_prompt_length", type=int, default=8192, help="Maximum prompt length (default: 8192)")
    parser.add_argument("--max_response_length", type=int, default=8192, help="Maximum response length (default: 8192)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Sampling top_p (default: 0.95)")
    parser.add_argument("--generator_system_prompt", type=str, default=None, help="Optional system prompt for bug generator")
    parser.add_argument("--fixer_system_prompt", type=str, default=None, help="Optional system prompt for bug fixer")
    parser.add_argument("--evaluate_codegen", action="store_true", help="Also evaluate fixer on regular code generation (val-mode only)")
    parser.add_argument("--include_failed_test_output", action="store_true", default=True,
                        help="Include failed test output in fixer prompts (default: True)")
    parser.add_argument("--no_failed_test_output", action="store_false", dest="include_failed_test_output",
                        help="Disable including failed test output in fixer prompts")
    
    # SSR-like self-play knobs
    parser.add_argument("--fixer_attempts_train", type=int, default=8, help="Number of fixer attempts during training (default: 8)")
    parser.add_argument("--fixer_attempts_val", type=int, default=1, help="Number of fixer attempts during validation (default: 1)")
    parser.add_argument("--generator_reward_mode", type=str, default="band", choices=["band", "smooth", "binary"],
                        help="Generator reward mode (default: band)")
    parser.add_argument("--solve_rate_band_low", type=float, default=0.05, help="Lower bound for solve rate band (default: 0.05)")
    parser.add_argument("--solve_rate_band_high", type=float, default=0.25, help="Upper bound for solve rate band (default: 0.25)")
    parser.add_argument("--gen_alpha_extreme", type=float, default=0.2, help="Penalty when solve_rate in {0,1} (default: 0.2)")
    parser.add_argument("--gen_invalid_bug_reward", type=float, default=-1.0, help="Penalty for invalid bug (default: -1.0)")
    parser.add_argument("--fixer_reward_pm1", action="store_true", help="Use {-1,+1} reward instead of {0,1} for fixer")
    
    # Code embedding reference dataset for similarity comparison
    parser.add_argument("--reference_bug_datasets", type=str, default=None,
                        help="Comma-separated list of datasets in format 'dataset:split' to load reference bugs from "
                             "(e.g., 'bugs_human_edited_lm:train,bugs_lm_qwen7b:train'). Takes precedence over --reference_bug_dataset.")
    parser.add_argument("--reference_bug_dataset", type=str, default="bugs_human_authored",
                        help="[DEPRECATED: use --reference_bug_datasets] Single dataset to load reference bugs from (default: bugs_human_authored)")
    parser.add_argument("--reference_bug_split", type=str, default="train",
                        help="[DEPRECATED: use --reference_bug_datasets] Split of reference bug dataset (default: train)")
    
    # Code-embedding similarity options
    parser.add_argument("--use_code_embedding_similarity", action="store_true", default=False,
                        help="Use code embedding similarity as auxiliary reward for generator")
    parser.add_argument("--code_embedding_reward_weight", type=float, default=0.3,
                        help="Weight for code embedding similarity reward (0-1, default: 0.3)")
    parser.add_argument("--code_embedding_model_name", type=str, default="voyage-code-3",
                        help="Embedding model to use (default: voyage-code-3)")
    parser.add_argument("--code_embedding_embed_mode", type=str, default="diff", choices=["diff", "buggy"],
                        help="Embed mode: 'diff' embeds unified diff (correct->buggy), 'buggy' embeds raw buggy code (default: diff)")
    parser.add_argument("--code_embedding_include_problem", action="store_true", default=False,
                        help="Include problem description in embedding (default: False)")
    parser.add_argument("--code_embedding_top_k", type=int, default=20,
                        help="Number of nearest neighbors to average for similarity (default: 20)")
    parser.add_argument("--code_embedding_target_pool_path", type=str, default=None,
                        help="Path prefix to load pre-computed target embedding pool (expects {path}_embeddings.npy and {path}_metadata.json)")
    parser.add_argument("--code_embedding_negative_pool_path", type=str, default=None,
                        help="Path prefix to load pre-computed negative embedding pool (expects {path}_embeddings.npy and {path}_metadata.json)")
    parser.add_argument("--negative_bug_datasets", type=str, default=None,
                        help="Comma-separated list of datasets in format 'dataset:split' to build negative pool from "
                             "(e.g., 'bugs_human_authored_llama:train,bugs_human_authored_gpt4:train'). Used when --code_embedding_negative_pool_path is not provided.")
    parser.add_argument("--save_code_embedding_target_pool", type=str, default=None,
                        help="Path prefix to save target embedding pool (creates {path}_embeddings.npy and {path}_metadata.json)")
    parser.add_argument("--save_code_embedding_negative_pool", type=str, default=None,
                        help="Path prefix to save negative embedding pool (creates {path}_embeddings.npy and {path}_metadata.json)")
    parser.add_argument("--code_embedding_use_margin", action="store_true", default=True,
                        help="Use margin-based scoring with target vs negative pools (default: True)")
    parser.add_argument("--code_embedding_margin_temperature", type=float, default=10.0,
                        help="Temperature for margin sigmoid (default: 10.0)")
    
    parser.add_argument("--eval_pregenerated_only", action="store_true",
                        help="Only evaluate on tasks with pregenerated bugs (skip tasks without bugs)")
    
    # Generator example bugs (few-shot prompting)
    parser.add_argument("--generator_example_bugs_dataset", type=str, default=None,
                        help="Dataset to load example bugs from for generator few-shot prompting (e.g., 'bugs_human_edited_lm')")
    parser.add_argument("--generator_example_bugs_split", type=str, default="train",
                        help="Split of example bugs dataset (default: train)")
    parser.add_argument("--generator_n_example_bugs", type=int, default=3,
                        help="Number of example bugs to include in generator prompt (default: 3)")
    parser.add_argument("--val_datasets", nargs="+", default=None,
                        help="Multiple validation datasets in format 'alias=dataset:split' or 'dataset:split' (e.g., 'bugs_human_authored:test' 'bcb=bigcodebench:test')")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to save results (default: logs)")
    parser.add_argument("--print_samples", type=int, default=3, help="Number of sample episodes to print in detail (default: 3)")
    parser.add_argument("--dataset", type=str, default="deepcoder", help="Dataset: deepcoder, bigcodebench, kodcode, bugs_human_authored, bugs_human_edited_lm, bugbench_adversarial, or custom (default: deepcoder)")
    parser.add_argument(
        "--generate_bugs_in_validation",
        action="store_true",
        default=False,
        help="If set, generate bugs in validation when available (default: False = use pregenerated bugs).",
    )
    parser.add_argument(
        "--generate_bugs_in_training",
        action="store_true",
        default=False,
        help="If set, generate bugs in training when available (default: False = use pregenerated bugs).",
    )

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Determine if we have separate fixer configuration
    use_separate_fixer = bool(
        args.fixer_model or args.fixer_base_url or args.fixer_api_key
    )
    
    # Generator engine setup
    print(f"Loading tokenizer for generator model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Initializing generator rollout engine...")
    generator_rollout_engine = OpenAIEngine(
        model=args.model,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        base_url=args.base_url,
        api_key=args.api_key,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )
    
    # Fixer engine setup (separate if specified, otherwise same as generator)
    fixer_rollout_engine = None
    if use_separate_fixer:
        fixer_model = args.fixer_model or args.model
        fixer_base_url = args.fixer_base_url or args.base_url
        fixer_api_key = args.fixer_api_key or args.api_key
        fixer_temperature = args.fixer_temperature if args.fixer_temperature is not None else args.temperature
        fixer_top_p = args.fixer_top_p if args.fixer_top_p is not None else args.top_p
        
        print(f"\nLoading tokenizer for fixer model: {fixer_model}")
        fixer_tokenizer = AutoTokenizer.from_pretrained(fixer_model)
        
        print("Initializing fixer rollout engine...")
        fixer_rollout_engine = OpenAIEngine(
            model=fixer_model,
            tokenizer=fixer_tokenizer,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            base_url=fixer_base_url,
            api_key=fixer_api_key,
            sampling_params={
                "temperature": fixer_temperature,
                "top_p": fixer_top_p,
            },
        )
        print(f"  Fixer model: {fixer_model}")
        print(f"  Fixer base_url: {fixer_base_url}")
        print(f"  Fixer temperature: {fixer_temperature}, top_p: {fixer_top_p}")
    
    # Use generator engine as main rollout_engine (for backward compatibility)
    rollout_engine = generator_rollout_engine

    # IMPORTANT: your workflow gates codegen (and "use pregenerated bug") on rollout_engine.validate
    # In eval_pregenerated_only mode, we must be in validation mode to use pregenerated bugs
    is_val_mode = True if args.eval_pregenerated_only else True
    setattr(rollout_engine, "validate", bool(is_val_mode))
    print(f"rollout_engine.validate = {getattr(rollout_engine, 'validate', False)}")

    # Load example bugs for generator few-shot prompting
    generator_example_bugs_from_tasks = None
    if args.generator_example_bugs_dataset:
        print(f"\nLoading example bugs from {args.generator_example_bugs_dataset}:{args.generator_example_bugs_split}...")
        example_bugs_data = load_data(dataset_name=args.generator_example_bugs_dataset, split=args.generator_example_bugs_split, n=1)
        if not example_bugs_data:
            ds = DatasetRegistry.load_dataset(args.generator_example_bugs_dataset, args.generator_example_bugs_split)
            if ds:
                example_bugs_data = list(ds.get_data())
        if example_bugs_data:
            generator_example_bugs_from_tasks = example_bugs_data
            print(f"  Loaded {len(generator_example_bugs_from_tasks)} tasks for generator example bugs")
        else:
            print(f"  WARNING: Could not load example bugs from {args.generator_example_bugs_dataset}:{args.generator_example_bugs_split}")

    # Load reference bugs for code embedding similarity if enabled
    reference_bugs = None
    if args.use_code_embedding_similarity:
        # Parse reference bug datasets - support multiple datasets
        ref_dataset_specs = []
        if args.reference_bug_datasets:
            # New format: comma-separated "dataset:split" pairs
            for spec in args.reference_bug_datasets.split(","):
                spec = spec.strip()
                if ":" in spec:
                    ds_name, split = spec.split(":", 1)
                    ref_dataset_specs.append((ds_name.strip(), split.strip()))
                else:
                    ref_dataset_specs.append((spec, "train"))
        else:
            # Fallback to deprecated single dataset args
            ref_dataset_specs.append((args.reference_bug_dataset, args.reference_bug_split))
        
        # Load and concatenate all reference bug datasets
        reference_bugs = []
        for ds_name, split in ref_dataset_specs:
            print(f"\nLoading reference bugs from {ds_name}:{split}...")
            ref_bugs_data = load_data(dataset_name=ds_name, split=split, n=1)
            if not ref_bugs_data:
                ds = DatasetRegistry.load_dataset(ds_name, split)
                if ds:
                    ref_bugs_data = list(ds.get_data())
            if ref_bugs_data:
                # Tag each task with its source dataset for debugging
                for task in ref_bugs_data:
                    task["_ref_dataset"] = f"{ds_name}:{split}"
                reference_bugs.extend(ref_bugs_data)
                print(f"  Loaded {len(ref_bugs_data)} reference bugs from {ds_name}:{split}")
            else:
                print(f"  WARNING: Could not load reference bugs from {ds_name}:{split}")
        
        if reference_bugs:
            print(f"\n📦 Total reference bugs loaded: {len(reference_bugs)} from {len(ref_dataset_specs)} dataset(s)")
        else:
            print(f"  WARNING: No reference bugs loaded from any dataset")

    # Pre-build embedding pools ONCE (to avoid rebuilding for each parallel workflow instance)
    prebuilt_target_pool = None
    prebuilt_negative_pool = None
    if args.use_code_embedding_similarity:
        try:
            from examples.asp.code_embedding import CodeEmbedder, CodeEmbeddingConfig, KNNBugSimilarity, ReferencePool
            
            # Load or build target pool
            if args.code_embedding_target_pool_path:
                print(f"\nLoading pre-computed TARGET embedding pool from {args.code_embedding_target_pool_path}...")
                prebuilt_target_pool = ReferencePool.load(args.code_embedding_target_pool_path)
                print(f"  Loaded {len(prebuilt_target_pool)} embeddings")
            elif reference_bugs:
                print(f"\nBuilding TARGET embedding pool from {len(reference_bugs)} reference bugs (once for all workers)...")
                emb_cfg = CodeEmbeddingConfig(
                    enabled=True,
                    model_name=args.code_embedding_model_name,
                    embed_mode=args.code_embedding_embed_mode,
                    include_problem=args.code_embedding_include_problem,
                    top_k=args.code_embedding_top_k,
                )
                embedder = CodeEmbedder(emb_cfg)
                knn = KNNBugSimilarity(embedder, top_k=args.code_embedding_top_k)
                knn.build_target_pool(reference_bugs)
                prebuilt_target_pool = knn.target_pool
                print(f"  Built pool with {len(prebuilt_target_pool)} embeddings")
            
            # Load or build negative pool
            if args.code_embedding_negative_pool_path:
                print(f"\nLoading pre-computed NEGATIVE embedding pool from {args.code_embedding_negative_pool_path}...")
                prebuilt_negative_pool = ReferencePool.load(args.code_embedding_negative_pool_path)
                print(f"  Loaded {len(prebuilt_negative_pool)} embeddings")
            elif args.negative_bug_datasets:
                # Build negative pool from specified datasets
                neg_dataset_specs = []
                for spec in args.negative_bug_datasets.split(","):
                    spec = spec.strip()
                    if ":" in spec:
                        ds_name, split = spec.split(":", 1)
                        neg_dataset_specs.append((ds_name.strip(), split.strip()))
                    else:
                        neg_dataset_specs.append((spec, "train"))
                
                # Load and concatenate all negative bug datasets
                negative_bugs = []
                for ds_name, split in neg_dataset_specs:
                    print(f"\nLoading negative bugs from {ds_name}:{split}...")
                    neg_bugs_data = load_data(dataset_name=ds_name, split=split, n=1)
                    if not neg_bugs_data:
                        ds = DatasetRegistry.load_dataset(ds_name, split)
                        if ds:
                            neg_bugs_data = list(ds.get_data())
                    if neg_bugs_data:
                        for task in neg_bugs_data:
                            task["_neg_dataset"] = f"{ds_name}:{split}"
                        negative_bugs.extend(neg_bugs_data)
                        print(f"  Loaded {len(neg_bugs_data)} negative bugs from {ds_name}:{split}")
                    else:
                        print(f"  WARNING: Could not load negative bugs from {ds_name}:{split}")
                
                if negative_bugs:
                    print(f"\n📦 Total negative bugs loaded: {len(negative_bugs)} from {len(neg_dataset_specs)} dataset(s)")
                    print(f"\nBuilding NEGATIVE embedding pool from {len(negative_bugs)} bugs...")
                    # Create embedder for negative pool (embedder may not exist if target pool was loaded from disk)
                    emb_cfg_neg = CodeEmbeddingConfig(
                        enabled=True,
                        model_name=args.code_embedding_model_name,
                        embed_mode=args.code_embedding_embed_mode,
                        include_problem=args.code_embedding_include_problem,
                        top_k=args.code_embedding_top_k,
                    )
                    embedder_neg = CodeEmbedder(emb_cfg_neg)
                    knn_neg = KNNBugSimilarity(embedder_neg, top_k=args.code_embedding_top_k)
                    knn_neg.build_target_pool(negative_bugs)  # build_target_pool works for any pool
                    prebuilt_negative_pool = knn_neg.target_pool
                    print(f"  Built negative pool with {len(prebuilt_negative_pool)} embeddings")
        except Exception as e:
            print(f"  WARNING: Failed to pre-build embedding pools: {e}")
            import traceback
            traceback.print_exc()

    print("Creating workflow engine...")
    # Always use pregenerated bugs in validation mode (default behavior)
    # The eval_pregenerated_only flag filters tasks to only include those with pregenerated bugs
    # use_pregenerated_bugs_in_validation = True
    # use_pregenerated_bugs_in_training = False  # Never use in training for eval scripts

    use_pregenerated_bugs_in_validation = not bool(args.generate_bugs_in_validation)
    use_pregenerated_bugs_in_training = not bool(args.generate_bugs_in_training)
    
    engine = AgentWorkflowEngine(
        workflow_cls=GeneratorFixerWorkflow,
        workflow_args={
            "reward_function": code_reward_fn,
            "generator_system_prompt": args.generator_system_prompt,
            "fixer_system_prompt": args.fixer_system_prompt,
            "evaluate_codegen": bool(args.evaluate_codegen),
            "include_failed_test_output": args.include_failed_test_output,
            "fixer_attempts_train": args.fixer_attempts_train,
            "fixer_attempts_val": args.fixer_attempts_val,
            "generator_reward_mode": args.generator_reward_mode,
            "solve_rate_band_low": args.solve_rate_band_low,
            "solve_rate_band_high": args.solve_rate_band_high,
            "gen_alpha_extreme": args.gen_alpha_extreme,
            "gen_invalid_bug_reward": args.gen_invalid_bug_reward,
            "fixer_reward_pm1": args.fixer_reward_pm1,
            "use_pregenerated_bugs_in_validation": use_pregenerated_bugs_in_validation,
            "use_pregenerated_bugs_in_training": use_pregenerated_bugs_in_training,
            # Separate rollout engines for generator and fixer (inference only)
            "generator_rollout_engine": generator_rollout_engine,
            "fixer_rollout_engine": fixer_rollout_engine,
            # Generator example bugs (few-shot prompting)
            "generator_example_bugs_from_tasks": generator_example_bugs_from_tasks,
            "generator_n_example_bugs": args.generator_n_example_bugs,
            # Code-embedding similarity options
            "use_code_embedding_similarity": args.use_code_embedding_similarity,
            "code_embedding_reward_weight": args.code_embedding_reward_weight,
            "code_embedding_model_name": args.code_embedding_model_name,
            "code_embedding_embed_mode": args.code_embedding_embed_mode,
            "code_embedding_include_problem": args.code_embedding_include_problem,
            "code_embedding_top_k": args.code_embedding_top_k,
            "code_embedding_target_pool_path": None,  # Don't load from disk, use pre-built pool
            "code_embedding_negative_pool_path": None,  # Don't load from disk, use pre-built pool
            "code_embedding_use_margin": args.code_embedding_use_margin,
            "code_embedding_margin_temperature": args.code_embedding_margin_temperature,
            # Pass pre-built pools (shared across all parallel workflow instances)
            "code_embedding_target_pool": prebuilt_target_pool,
            "code_embedding_negative_pool": prebuilt_negative_pool,
            # Don't pass reference_bugs to avoid rebuilding (we already built the pool)
            "code_embedding_reference_bugs": None,
        },
        rollout_engine=rollout_engine,
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
        # Multiple validation datasets mode - concatenate all tasks and run in one batch
        # This avoids "engine already in use" errors from reusing the WorkflowEngine
        print(f"\n📊 Loading {len(val_dataset_specs)} validation datasets...")
        all_tasks = []
        dataset_task_counts = {}
        
        for alias, (ds_name, split) in val_dataset_specs.items():
            print(f"\nLoading: {alias} ({ds_name}:{split})...")
            
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
            
            # Tag each task with dataset alias for later grouping
            for task in tasks:
                task["_dataset_alias"] = alias
            
            all_tasks.extend(tasks)
            dataset_task_counts[alias] = len(tasks)
            print(f"  Loaded {len(tasks)} tasks")
        
        if not all_tasks:
            print("No tasks loaded from any dataset. Exiting.")
            raise SystemExit(1)
        
        print(f"\n{'='*80}")
        print(f"Total tasks across all datasets: {len(all_tasks)}")
        for alias, count in dataset_task_counts.items():
            print(f"  {alias}: {count} tasks")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Generator model: {args.model}")
        if use_separate_fixer:
            fixer_model = args.fixer_model or args.model
            print(f"  Fixer model: {fixer_model}")
            print(f"  Fixer base_url: {args.fixer_base_url or args.base_url}")
        else:
            print(f"  Fixer model: (same as generator)")
        print(f"  Parallel tasks: {args.n_parallel}")
        print(f"  Temperature: {args.temperature}, Top-p: {args.top_p}")
        print(f"  Eval pregenerated only: {args.eval_pregenerated_only}")
        
        print(f"\n🚀 Executing workflow on {len(all_tasks)} tasks...")
        results = asyncio.run(engine.execute_tasks(all_tasks))
        
        # Save embedding pools (they're the same across all datasets)
        if args.use_code_embedding_similarity:
            save_embedding_pools(
                target_pool=prebuilt_target_pool,
                negative_pool=prebuilt_negative_pool,
                save_target_path=args.save_code_embedding_target_pool,
                save_negative_path=args.save_code_embedding_negative_pool
            )
        
        # Evaluate results grouped by dataset
        print("\n📊 Evaluating results per dataset...")
        all_results, all_summaries = evaluate_results_grouped(results)
        
        # Print sample episodes per dataset
        if args.print_samples > 0:
            for alias, group_results in all_results.items():
                print(f"\n📝 Sample episodes for {alias}:")
                print_sample_episodes(group_results, n_samples=args.print_samples)
        
        # Summary across all datasets
        print("\n" + "=" * 80)
        print("📊 SUMMARY ACROSS ALL VALIDATION DATASETS")
        print("=" * 80)
        for alias, group_results in all_results.items():
            total = len(group_results)
            if total == 0:
                continue
            fixer_pass = sum(1 for ep in group_results if float((ep.metrics or {}).get("fixer_pass", 0.0)) > 0.0)
            bug_valid = sum(1 for ep in group_results if float((ep.metrics or {}).get("bug_valid", 0.0)) > 0.0)
            print(f"  {alias}: {fixer_pass}/{total} fixer pass ({100*fixer_pass/total:.1f}%), "
                  f"{bug_valid}/{total} bug valid ({100*bug_valid/total:.1f}%)")
        
        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save concise summary JSON
            summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")
            with open(summary_file, "w") as f:
                json.dump(all_summaries, f, indent=2)
            print(f"\n📊 Summary saved to: {summary_file}")
            
            # Save full results per dataset
            for alias, group_results in all_results.items():
                output_file = os.path.join(args.output_dir, f"generator_fixer_flow_{alias}_{timestamp}.json")
                results_dict = [exclude_token_ids(episode.to_dict()) for episode in group_results]
                with open(output_file, "w") as f:
                    json.dump(results_dict, f, indent=2)
                print(f"💾 Results for {alias} saved to: {output_file}")
        
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
        print(f"  Generator model: {args.model}")
        if use_separate_fixer:
            fixer_model = args.fixer_model or args.model
            print(f"  Fixer model: {fixer_model}")
            print(f"  Fixer base_url: {args.fixer_base_url or args.base_url}")
        else:
            print(f"  Fixer model: (same as generator)")
        print(f"  Parallel tasks: {args.n_parallel}")
        print(f"  Temperature: {args.temperature}, Top-p: {args.top_p}")
        print(f"  Max prompt length: {args.max_prompt_length}")
        print(f"  Max response length: {args.max_response_length}")
        print(f"  Evaluate CodeGen: {bool(args.evaluate_codegen)}")
        print(f"  Fixer attempts (train/val): {args.fixer_attempts_train}/{args.fixer_attempts_val}")
        print(f"  Generator reward mode: {args.generator_reward_mode}")
        print(f"  Eval pregenerated only: {args.eval_pregenerated_only}")
        if args.use_code_embedding_similarity:
            print(f"  Code Embedding Similarity: ENABLED (weight={args.code_embedding_reward_weight}, model={args.code_embedding_model_name})")
            if args.code_embedding_target_pool_path:
                print(f"    Target pool: {args.code_embedding_target_pool_path}")
            if args.code_embedding_negative_pool_path:
                print(f"    Negative pool: {args.code_embedding_negative_pool_path}")
            print(f"    Use margin: {args.code_embedding_use_margin}, Temperature: {args.code_embedding_margin_temperature}")

        print(f"\n🚀 Executing workflow on {len(all_tasks)} tasks...")
        results = asyncio.run(engine.execute_tasks(all_tasks))

        # Save embedding pools if requested
        if args.use_code_embedding_similarity:
            save_embedding_pools(
                target_pool=prebuilt_target_pool,
                negative_pool=prebuilt_negative_pool,
                save_target_path=args.save_code_embedding_target_pool,
                save_negative_path=args.save_code_embedding_negative_pool
            )

        print("\n📊 Evaluating results...")
        summary = evaluate_results(results, dataset_alias=args.dataset)

        if args.print_samples > 0:
            print_sample_episodes(results, n_samples=args.print_samples)

        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save concise summary JSON
            summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")
            with open(summary_file, "w") as f:
                json.dump({args.dataset: summary}, f, indent=2)
            print(f"\n📊 Summary saved to: {summary_file}")
            
            # Save full results
            output_file = os.path.join(args.output_dir, f"generator_fixer_flow_results_{timestamp}.json")
            results_dict = [exclude_token_ids(episode.to_dict()) for episode in results]
            with open(output_file, "w") as f:
                json.dump(results_dict, f, indent=2)
            print(f"💾 Results saved to: {output_file}")

    print("\n✅ Done!")