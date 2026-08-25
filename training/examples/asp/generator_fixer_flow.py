from __future__ import annotations

import asyncio
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow

from examples.asp.prompts import _extract_failed_test_output
from examples.asp.code_embedding import (
    CodeEmbeddingConfig,
    CodeEmbedder,
    KNNBugSimilarity,
    ReferencePool,
)
from examples.asp.components import (
    BugGenerator,
    BugGeneratorConfig,
    BugFixer,
    BugFixerConfig,
)
from examples.asp.utils import (
    _get_problem as util_get_problem,
    _get_reference_solution as util_get_reference_solution,
    _get_pregenerated_bug as util_get_pregenerated_bug,
)
from examples.asp.utils import (
    normalize_task_info,
    check_bug_validity,
    _get_problem,
    _get_reference_solution,
    _get_pregenerated_bug,
    _shaped_generator_reward_from_solve_rate,
)


def _extract_example_bugs_from_tasks(
    tasks: List[Dict[str, Any]],
    n: int = 3,
) -> List[Dict[str, str]]:
    """Extract example bugs from a list of tasks that have pregenerated bugs.
    
    Returns a list of dicts with keys: problem, correct_code, buggy_code.
    """
    examples = []
    for task in tasks:
        buggy = util_get_pregenerated_bug(task)
        if not buggy:
            continue
        correct = util_get_reference_solution(task)
        problem = util_get_problem(task)
        if not correct:
            continue
        examples.append({
            "problem": problem,
            "correct_code": correct,
            "buggy_code": buggy,
        })
        if len(examples) >= n:
            break
    return examples


class GeneratorFixerWorkflow(Workflow):
    """Train-time: generator creates bug -> fixer fixes it (K attempts for solve-rate).
    Validation-time (BugBench style): if task contains a pre-generated bug
    (buggy_solution/buggy/buggy_code), skip generation and evaluate fixer on that held-out bug.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        reward_function: RewardFunction,
        generator_system_prompt: Optional[str] = None,
        fixer_system_prompt: Optional[str] = None,
        evaluate_codegen: bool = True,
        # Optional separate rollout engines for generator and fixer (inference only)
        generator_rollout_engine: Optional[RolloutEngine] = None,
        fixer_rollout_engine: Optional[RolloutEngine] = None,
        # Validation behavior
        use_pregenerated_bugs_in_validation: bool = True,
        # Training behavior: optionally use human/pregenerated bugs if present on the task.
        use_pregenerated_bugs_in_training: bool = False,
        pregenerated_bug_train_probability: float = 1.0,
        # Defines what Episode.is_correct means (and thus val/pass@k):
        # - "bugfix": success iff (bug_valid and fixer_pass_any)
        # - "codegen": success iff fixer_codegen_pass (validation only; falls back to bugfix on train)
        episode_success_mode: str = "bugfix",
        # Train generator on reference/pregenerated-bug episodes (ablation for
        # reviewer question: "what if we update the generator on mixed episodes?").
        # When True and a pregenerated bug is used during training, the generator
        # still produces its own bug, which is evaluated via the normal solve-rate
        # pipeline (K fixer attempts) to provide a generator RL reward.  The fixer
        # training signal still comes from the pregenerated/reference bug.
        train_generator_on_reference_episodes: bool = False,
        # Solve-rate evaluation knobs
        fixer_attempts_train: int = 8,
        fixer_attempts_val: int = 1,
        # Generator reward shaping knobs (SSR-like)
        generator_reward_mode: str = "band",  # "band" | "smooth" | "binary"
        solve_rate_band_low: float = 0.05,
        solve_rate_band_high: float = 0.25,
        gen_alpha_extreme: float = 0.2,        # penalty when solve_rate in {0,1}
        gen_invalid_bug_reward: float = -1.0,  # penalty for invalid bug
        # Fixer reward style
        fixer_reward_pm1: bool = False,       # False => {0,1}, True => {-1,+1}
        # Include failed test output in fixer prompts
        include_failed_test_output: bool = True,
        # Few-shot example bugs for generator prompt guidance
        generator_example_bugs: Optional[List[Dict[str, str]]] = None,  # List of {problem, correct_code, buggy_code}
        generator_example_bugs_from_tasks: Optional[List[Dict[str, Any]]] = None,  # Alternative: extract from tasks
        generator_n_example_bugs: int = 3,  # Number of examples to include
        # Code-embedding similarity (optional auxiliary reward)
        use_code_embedding_similarity: bool = False,
        code_embedding_reward_weight: float = 0.3,
        code_embedding_model_name: str = "voyage-code-3",
        code_embedding_embed_mode: str = "diff",  # "diff" | "buggy"
        code_embedding_include_problem: bool = False,
        code_embedding_top_k: int = 5,
        # Pools: you can pass tasks directly OR load pools from disk OR pass pre-built pools
        code_embedding_reference_bugs: Optional[List[Dict[str, Any]]] = None,
        code_embedding_negative_bugs: Optional[List[Dict[str, Any]]] = None,
        code_embedding_target_pool_path: Optional[str] = None,
        code_embedding_negative_pool_path: Optional[str] = None,
        # Pre-built pool objects (avoids rebuilding for each parallel workflow instance)
        code_embedding_target_pool: Optional[Any] = None,
        code_embedding_negative_pool: Optional[Any] = None,
        # If negative pool exists, use margin = target - negative and sigmoid it
        code_embedding_use_margin: bool = True,
        code_embedding_margin_temperature: float = 10.0,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)

        self.reward_function = reward_function
        
        # Use separate rollout engines if provided (for inference with different models)
        gen_engine = generator_rollout_engine if generator_rollout_engine is not None else rollout_engine
        fix_engine = fixer_rollout_engine if fixer_rollout_engine is not None else rollout_engine
        
        # Process example bugs for generator prompt
        example_bugs: Optional[List[Dict[str, str]]] = None
        if generator_example_bugs:
            example_bugs = list(generator_example_bugs)
        elif generator_example_bugs_from_tasks:
            example_bugs = _extract_example_bugs_from_tasks(
                generator_example_bugs_from_tasks,
                n=int(generator_n_example_bugs),
            )
        
        self.generator = BugGenerator(
            gen_engine,
            BugGeneratorConfig(
                system_prompt=generator_system_prompt,
                example_bugs=example_bugs,
                n_example_bugs=int(generator_n_example_bugs),
            ),
        )
        self.fixer = BugFixer(
            fix_engine,
            BugFixerConfig(
                system_prompt=fixer_system_prompt,
                include_failed_test_output=bool(include_failed_test_output),
            ),
        )

        self.evaluate_codegen = bool(evaluate_codegen)
        self.use_pregenerated_bugs_in_validation = bool(use_pregenerated_bugs_in_validation)
        self.use_pregenerated_bugs_in_training = bool(use_pregenerated_bugs_in_training)
        self.pregenerated_bug_train_probability = float(pregenerated_bug_train_probability)
        self.episode_success_mode = str(episode_success_mode).lower().strip()
        self.train_generator_on_reference_episodes = bool(train_generator_on_reference_episodes)

        self.fixer_attempts_train = max(1, int(fixer_attempts_train))
        self.fixer_attempts_val = max(1, int(fixer_attempts_val))

        self.generator_reward_mode = str(generator_reward_mode)
        self.solve_rate_band_low = float(solve_rate_band_low)
        self.solve_rate_band_high = float(solve_rate_band_high)
        self.gen_alpha_extreme = float(gen_alpha_extreme)
        self.gen_invalid_bug_reward = float(gen_invalid_bug_reward)

        self.fixer_reward_pm1 = bool(fixer_reward_pm1)

        # ---------------------------
        # Code-embedding similarity (aux reward)
        # ---------------------------
        self.use_code_embedding_similarity = bool(use_code_embedding_similarity)
        self.code_embedding_reward_weight = float(code_embedding_reward_weight)
        self.code_embedding_use_margin = bool(code_embedding_use_margin)
        self.code_embedding_margin_temperature = float(code_embedding_margin_temperature)

        self._code_embedder: Optional[CodeEmbedder] = None
        self._code_knn_target: Optional[KNNBugSimilarity] = None

        # --- code-embed baseline (EMA) for advantage-style shaping ---
        self._embed_baseline_ema = 0.0
        self._embed_baseline_beta = 0.99   # try 0.95–0.995
        self._embed_baseline_inited = False
        self._embed_delta_clip = 0.2       # optional safety clip

        if self.use_code_embedding_similarity:
            # (These imports are in-file, so this is mostly a sanity check.)
            if CodeEmbeddingConfig is None or CodeEmbedder is None or KNNBugSimilarity is None or ReferencePool is None:
                print("[CodeEmbedding] WARNING: code_embedding deps not available. Disabling.")
                self.use_code_embedding_similarity = False
            else:
                emb_cfg = CodeEmbeddingConfig(
                    enabled=True,
                    reward_weight=self.code_embedding_reward_weight,
                    model_name=str(code_embedding_model_name),
                    embed_mode=str(code_embedding_embed_mode),
                    include_problem=bool(code_embedding_include_problem),
                    top_k=int(code_embedding_top_k),
                )
                self._code_embedder = CodeEmbedder(emb_cfg)

                # Target scorer
                self._code_knn_target = KNNBugSimilarity(self._code_embedder, top_k=int(code_embedding_top_k))

                # 1) TARGET pool: pre-built object OR load OR build from tasks
                if code_embedding_target_pool is not None:
                    self._code_knn_target.target_pool = code_embedding_target_pool
                    print(f"[CodeEmbedding] Using pre-built TARGET pool (n={self._safe_len(code_embedding_target_pool)})")
                elif code_embedding_target_pool_path:
                    try:
                        pool = ReferencePool.load(str(code_embedding_target_pool_path))
                        self._code_knn_target.target_pool = pool
                        print(f"[CodeEmbedding] Loaded TARGET pool from {code_embedding_target_pool_path} (n={self._safe_len(pool)})")
                    except Exception as e:
                        print(f"[CodeEmbedding] WARNING: Failed loading TARGET pool: {e}")
                else:
                    if code_embedding_reference_bugs:
                        try:
                            self._code_knn_target.build_target_pool(list(code_embedding_reference_bugs))
                            tp = getattr(self._code_knn_target, "target_pool", None)
                            print(f"[CodeEmbedding] Built TARGET pool (n={self._safe_len(tp)})")
                        except Exception as e:
                            print(f"[CodeEmbedding] WARNING: Failed building TARGET pool: {e}")

                # 2) NEGATIVE pool: optional (NOTE: this must attach to _code_knn_target.negative_pool)
                neg_pool: Optional[ReferencePool] = None
                if code_embedding_negative_pool is not None:
                    neg_pool = code_embedding_negative_pool
                    print(f"[CodeEmbedding] Using pre-built NEGATIVE pool (n={self._safe_len(neg_pool)})")
                elif code_embedding_negative_pool_path:
                    try:
                        neg_pool = ReferencePool.load(str(code_embedding_negative_pool_path))
                        print(f"[CodeEmbedding] Loaded NEGATIVE pool from {code_embedding_negative_pool_path} (n={self._safe_len(neg_pool)})")
                    except Exception as e:
                        print(f"[CodeEmbedding] WARNING: Failed loading NEGATIVE pool: {e}")
                        neg_pool = None
                elif code_embedding_negative_bugs:
                    try:
                        # Build a ReferencePool using the helper method on KNNBugSimilarity,
                        # which correctly places it in `.negative_pool`.
                        tmp = KNNBugSimilarity(self._code_embedder, top_k=int(code_embedding_top_k))
                        tmp.build_negative_pool(list(code_embedding_negative_bugs))
                        neg_pool = tmp.negative_pool
                        print(f"[CodeEmbedding] Built NEGATIVE pool (n={self._safe_len(neg_pool)})")
                    except Exception as e:
                        print(f"[CodeEmbedding] WARNING: Failed building NEGATIVE pool: {e}")
                        neg_pool = None

                # Attach negative pool to scorer + configure relative scoring
                if neg_pool is not None and self._safe_len(neg_pool) > 0:
                    # KNNBugSimilarity expects a ReferencePool object here
                    self._code_knn_target.negative_pool = neg_pool

                    # Control relative scoring via embedder config
                    self._code_embedder.config.use_relative_score = bool(self.code_embedding_use_margin)
                    self._code_embedder.config.margin_temperature = float(self.code_embedding_margin_temperature)

                    print(
                        f"[CodeEmbedding] Attached NEGATIVE pool (n={self._safe_len(neg_pool)}), "
                        f"use_relative_score={self._code_embedder.config.use_relative_score}, "
                        f"margin_temperature={self._code_embedder.config.margin_temperature}"
                    )
                else:
                    # No negative pool; ensure absolute scoring if desired
                    if self._code_embedder is not None:
                        self._code_embedder.config.use_relative_score = False

                # Final sanity: target pool must exist and be non-empty
                tp = getattr(self._code_knn_target, "target_pool", None) if self._code_knn_target is not None else None
                if self._code_knn_target is None or self._safe_len(tp) == 0:
                    print("[CodeEmbedding] WARNING: Enabled but TARGET pool is empty. Disabling.")
                    self.use_code_embedding_similarity = False

    def _normalize_task_info(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_task_info(task)

    @staticmethod
    def _safe_len(x: Any) -> int:
        if x is None:
            return 0
        try:
            return len(x)
        except Exception:
            return 0

    def _score_code_embedding_aux_sync(
        self,
        *,
        problem: str,
        buggy_code: str,
        correct_code: Optional[str],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Returns:
          score01 in [0,1] and metadata (includes margin info if relative scoring enabled).
        """
        if not self.use_code_embedding_similarity or self._code_knn_target is None:
            return 0.0, {"disabled": True}

        score01, meta = self._code_knn_target.score_similarity(
            problem,
            buggy_code,
            correct_code=correct_code,
        )
        return float(score01), (meta or {})

    def _embed_advantage(self, sim01: float) -> Tuple[float, float]:
        """
        Returns (delta, baseline) where delta = sim01 - baseline, baseline is EMA.
        sim01 must be in [0,1].
        """
        sim01 = float(max(0.0, min(1.0, sim01)))
        if not self._embed_baseline_inited:
            self._embed_baseline_ema = sim01
            self._embed_baseline_inited = True
        else:
            beta = float(self._embed_baseline_beta)
            self._embed_baseline_ema = beta * self._embed_baseline_ema + (1.0 - beta) * sim01

        baseline = float(self._embed_baseline_ema)
        delta = float(sim01 - baseline)

        clip = float(getattr(self, "_embed_delta_clip", 0.0))
        if clip > 0:
            delta = max(-clip, min(clip, delta))

        return delta, baseline

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        is_validation = bool(getattr(self.rollout_engine, "validate", False))
        task_info = self._normalize_task_info(task)
        metrics: Dict[str, Any] = {}

        # ---------------------------
        # Decide bug source (pregenerated vs generator)
        # ---------------------------
        pregenerated_bug: Optional[str] = None
        allow_pregenerated = (is_validation and self.use_pregenerated_bugs_in_validation) or (
            (not is_validation) and self.use_pregenerated_bugs_in_training
        )
        if allow_pregenerated:
            cand = _get_pregenerated_bug(task)
            if cand is not None:
                if is_validation:
                    pregenerated_bug = cand
                else:
                    p = max(0.0, min(1.0, float(self.pregenerated_bug_train_probability)))
                    if p >= 1.0:
                        pregenerated_bug = cand
                    elif p <= 0.0:
                        pregenerated_bug = None
                    else:
                        seed = int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16) % (2**32)
                        if random.Random(seed).random() < p:
                            pregenerated_bug = cand

        # ---------------------------
        # BUG SOURCE
        # ---------------------------
        bug_traj: Optional[Trajectory] = None
        bug_step: Optional[Step] = None
        generated_buggy_code: Optional[str] = None

        if pregenerated_bug is None:
            bug_traj = await self.generator.generate_bug(task)
            bug_step = bug_traj.steps[0]
            buggy_code = bug_step.action
        elif self.train_generator_on_reference_episodes and not is_validation:
            bug_traj = await self.generator.generate_bug(task)
            bug_step = bug_traj.steps[0]
            generated_buggy_code = bug_step.action
            buggy_code = pregenerated_bug
        else:
            buggy_code = pregenerated_bug

        bug_id = hashlib.md5(buggy_code.encode("utf-8")).hexdigest()[:12]

        # ---------------------------
        # BUG VALIDITY (via reward_fn)
        # ---------------------------
        try:
            bug_reward_output = self.reward_function(task_info=task_info, action=buggy_code)
        except Exception as e:
            bug_reward_output = RewardOutput(
                reward=0.0,
                is_correct=False,
                metadata={"error": f"bug reward error: {e}"},
            )

        bug_meta = bug_reward_output.metadata or {}
        bug_valid, _has_compile_error = check_bug_validity(
            bug_meta=bug_meta,
            bug_reward_output=bug_reward_output,
            compile_errors_invalid=True,
        )
        total_tests = bug_meta.get("total_tests")
        passed_tests = bug_meta.get("passed_tests")

        # Extract failed test output for fixer prompt (if enabled)
        failed_test_output: Optional[str] = None
        if self.fixer.config.include_failed_test_output:
            failed_test_output = _extract_failed_test_output(bug_meta)

        # ---------------------------
        # SOLVER FIX (K attempts -> solve_rate)
        # ---------------------------
        fixer_attempts = self.fixer_attempts_val if is_validation else self.fixer_attempts_train

        fixer_trajs: List[Trajectory] = []
        fixer_passes: List[bool] = []

        if bug_valid:
            fix_tasks = [
                self.fixer.fix_bug(
                    task,
                    buggy_code=buggy_code,
                    failed_test_output=failed_test_output,
                )
                for _ in range(fixer_attempts)
            ]
            fixer_trajs = await asyncio.gather(*fix_tasks)

            for traj in fixer_trajs:
                fixed_code = traj.steps[0].action
                try:
                    out = self.reward_function(task_info=task_info, action=fixed_code)
                    fixer_passes.append(bool(out.is_correct))
                except Exception:
                    fixer_passes.append(False)

        num_pass = int(sum(1 for p in fixer_passes if p))
        solve_rate = float(num_pass) / float(fixer_attempts) if fixer_attempts > 0 else 0.0
        fixer_pass_any = bool(num_pass > 0)
        fixer_pass_all = bool(num_pass == fixer_attempts and fixer_attempts > 0)
        metrics["fixer_pass"] = float(solve_rate)

        # Per-attempt fixer reward
        for k, traj in enumerate(fixer_trajs):
            sp = fixer_passes[k]
            traj.steps[0].reward = float(1.0 if sp else (-1.0 if self.fixer_reward_pm1 else 0.0))

        # ---------------------------
        # GENERATOR EVALUATION ON REFERENCE EPISODES
        # When train_generator_on_reference_episodes is enabled and the fixer
        # trained on a pregenerated bug, we separately evaluate the generator's
        # own bug to compute a solve-rate-based reward for the generator.
        # ---------------------------
        gen_bug_valid_for_reward: Optional[bool] = None
        gen_solve_rate_for_reward: Optional[float] = None

        if generated_buggy_code is not None:
            try:
                gen_bug_reward_output = self.reward_function(
                    task_info=task_info, action=generated_buggy_code,
                )
            except Exception as e:
                gen_bug_reward_output = RewardOutput(
                    reward=0.0,
                    is_correct=False,
                    metadata={"error": f"gen bug reward error: {e}"},
                )

            gen_bug_meta = gen_bug_reward_output.metadata or {}
            gen_bug_valid_for_reward, _ = check_bug_validity(
                bug_meta=gen_bug_meta,
                bug_reward_output=gen_bug_reward_output,
                compile_errors_invalid=True,
            )

            if gen_bug_valid_for_reward:
                gen_failed_test_output: Optional[str] = None
                if self.fixer.config.include_failed_test_output:
                    gen_failed_test_output = _extract_failed_test_output(gen_bug_meta)

                gen_fix_tasks = [
                    self.fixer.fix_bug(
                        task,
                        buggy_code=generated_buggy_code,
                        failed_test_output=gen_failed_test_output,
                    )
                    for _ in range(fixer_attempts)
                ]
                gen_fixer_trajs = await asyncio.gather(*gen_fix_tasks)

                gen_fixer_passes: List[bool] = []
                for gtraj in gen_fixer_trajs:
                    fixed_code = gtraj.steps[0].action
                    try:
                        out = self.reward_function(task_info=task_info, action=fixed_code)
                        gen_fixer_passes.append(bool(out.is_correct))
                    except Exception:
                        gen_fixer_passes.append(False)

                gen_num_pass = sum(1 for p in gen_fixer_passes if p)
                gen_solve_rate_for_reward = (
                    float(gen_num_pass) / float(fixer_attempts)
                    if fixer_attempts > 0 else 0.0
                )
            else:
                gen_solve_rate_for_reward = 0.0

            metrics["gen_on_ref_bug_valid"] = float(gen_bug_valid_for_reward)
            metrics["gen_on_ref_solve_rate"] = float(gen_solve_rate_for_reward)

        # ---------------------------
        # OPTIONAL CODEGEN EVAL (VAL ONLY)
        # ---------------------------
        codegen_pass = False
        if self.evaluate_codegen and is_validation:
            codegen_traj = await self.fixer.generate_code(task)
            codegen_step = codegen_traj.steps[0]
            generated_code = codegen_step.action
            try:
                codegen_reward_output = self.reward_function(task_info=task_info, action=generated_code)
                codegen_pass = bool(codegen_reward_output.is_correct)
            except Exception:
                codegen_pass = False
            codegen_step.reward = 0.0

        # ---------------------------
        # CODE-EMBEDDING BUG SIMILARITY (aux reward) (only when generator generated)
        # ---------------------------
        code_embed_score = 0.0
        code_embed_meta: Dict[str, Any] = {}

        generator_produced_bug = pregenerated_bug is None or generated_buggy_code is not None
        if self.use_code_embedding_similarity and generator_produced_bug:
            embed_bug_code = generated_buggy_code if generated_buggy_code is not None else buggy_code
            embed_bug_valid = gen_bug_valid_for_reward if generated_buggy_code is not None else bug_valid
            if embed_bug_valid:
                loop = asyncio.get_running_loop()
                problem = _get_problem(task)
                correct_code = _get_reference_solution(task) or None

                code_embed_score, code_embed_meta = await loop.run_in_executor(
                    self.executor,
                    lambda: self._score_code_embedding_aux_sync(
                        problem=problem,
                        buggy_code=embed_bug_code,
                        correct_code=correct_code,
                    ),
                )
            else:
                code_embed_score, code_embed_meta = 0.0, {"skipped": True, "reason": "bug_invalid"}

        # ---------------------------
        # ASSIGN GENERATOR RL REWARD
        # ---------------------------
        if pregenerated_bug is not None and generated_buggy_code is None:
            generator_reward = 0.0
            generator_base_reward = 0.0
        else:
            # Determine which bug validity / solve rate drives the generator reward.
            # When the generator ran on a reference episode, use its own evaluation;
            # otherwise use the main fixer pipeline results.
            if generated_buggy_code is not None:
                r_bug_valid = gen_bug_valid_for_reward
                r_solve_rate = gen_solve_rate_for_reward
            else:
                r_bug_valid = bug_valid
                r_solve_rate = solve_rate

            generator_base_reward = _shaped_generator_reward_from_solve_rate(
                bug_valid=r_bug_valid,
                solve_rate=r_solve_rate,
                mode=self.generator_reward_mode,
                band_low=self.solve_rate_band_low,
                band_high=self.solve_rate_band_high,
                alpha_extreme=self.gen_alpha_extreme,
                invalid_bug_reward=self.gen_invalid_bug_reward,
            )

            generator_reward = float(generator_base_reward)

            if self.use_code_embedding_similarity and r_bug_valid:
                sim01 = float(max(0.0, min(1.0, code_embed_score)))
                delta, baseline = self._embed_advantage(sim01)
                w = max(0.0, float(self.code_embedding_reward_weight))
                generator_reward = float(generator_base_reward) + w * float(delta)

                metrics["code_embed_sim01"] = float(sim01)
                metrics["code_embed_baseline"] = float(baseline)
                metrics["code_embed_delta"] = float(delta)
                metrics["generator_base_reward"] = float(generator_base_reward)

        if bug_step is not None:
            bug_step.reward = float(generator_reward)

        # ---------------------------
        # BUILD EPISODE
        # ---------------------------
        trajectories: List[Trajectory] = []
        if bug_traj is not None:
            trajectories.append(bug_traj)
        trajectories.extend(fixer_trajs)

        # Bug validity and test metrics
        metrics["bug_valid"] = float(bug_valid)
        if total_tests is not None:
            metrics["bug_total_tests"] = int(total_tests)
        if passed_tests is not None:
            metrics["bug_passed_tests"] = int(passed_tests)

        # Generator reward metric
        metrics["generator_reward"] = float(bug_step.reward) if (bug_traj is not None and bug_step is not None) else 0.0

        # Fixer codegen pass
        if self.evaluate_codegen and is_validation:
            metrics["fixer_codegen_pass"] = float(codegen_pass)

        # Code-embed diagnostics (propagated from code_embedding.py meta)
        if self.use_code_embedding_similarity and code_embed_meta:
            metrics["code_embed_score"] = float(code_embed_score)
            if "avg_cosine_target" in code_embed_meta:
                metrics["code_embed_avg_cosine_target"] = float(code_embed_meta["avg_cosine_target"])
            if "avg_cosine_negative" in code_embed_meta:
                metrics["code_embed_avg_cosine_negative"] = float(code_embed_meta["avg_cosine_negative"])
            if "margin" in code_embed_meta:
                metrics["code_embed_margin"] = float(code_embed_meta["margin"])
            if "score_norm" in code_embed_meta:
                metrics["code_embed_score_norm"] = float(code_embed_meta["score_norm"])

        episode_info: Dict[str, Any] = {
            "fixer_pass_any": bool(fixer_pass_any),
            "fixer_pass_all": bool(fixer_pass_all),
            "is_validation": bool(is_validation),
            "bug_valid": bool(bug_valid),
            "codegen_pass": bool(codegen_pass),
            "workflow": "generator_fixer",
            "bug_id": bug_id,
        }

        episode = Episode(
            id=uid,
            task=task,
            trajectories=trajectories,
            is_correct=False,  # set below
            metrics=metrics,
            info=episode_info,
        )

        self.assign_episode_correctness(episode)
        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        """Assign episode correctness based on parameters stored in episode.info."""
        info = episode.info or {}
        fixer_pass_any = bool(info.get("fixer_pass_any", False))
        is_validation = bool(info.get("is_validation", False))
        bug_valid = bool(info.get("bug_valid", False))
        codegen_pass = bool(info.get("codegen_pass", False))

        mode = self.episode_success_mode
        if is_validation:
            if mode == "codegen":
                episode.is_correct = bool(codegen_pass)
            elif mode == "bugfix":
                episode.is_correct = bool(bug_valid and fixer_pass_any)
            else:
                episode.is_correct = bool(bug_valid and fixer_pass_any)
        else:
            episode.is_correct = bool(bug_valid and fixer_pass_any)
