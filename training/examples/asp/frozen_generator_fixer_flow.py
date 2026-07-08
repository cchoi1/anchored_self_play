from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine, OpenAIEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow

from examples.asp.prompts import _extract_failed_test_output
from examples.asp.components import (
    BugGenerator,
    BugGeneratorConfig,
    BugFixer,
    BugFixerConfig,
)
from examples.asp.utils import (
    normalize_task_info,
    check_bug_validity,
    _get_pregenerated_bug,
    _resolve_api_key,
)


class FrozenGeneratorFixerWorkflow(Workflow):
    """Train a fixer on a frozen BugGenerator's bugs.
    
    Similar to GeneratorFixerWorkflow, but the generator is frozen (not trained).
    The fixer is trainable and makes 1 fix attempt per bug/episode.
    
    Flow:
    1) Frozen BugGenerator generates a bug (or use pregenerated bug in validation).
    2) Reward function evaluates bug validity (unit tests run).
    3) Trainable fixer attempts to fix the bug (1 attempt).
    4) Fixer gets reward based on whether it passes tests.
    5) Only fixer trajectory is returned for training (generator is frozen).
    
    Notes:
      - Generator trajectory is included in episode but with 0 reward (for logging).
      - Supports pregenerated bugs in validation mode.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        reward_function: RewardFunction,
        # Frozen generator configuration.
        generator_rollout_engine: Optional[RolloutEngine] = None,
        generator_model: Optional[str] = None,
        generator_base_url: Optional[str] = None,
        generator_api_key: Optional[str] = None,
        generator_temperature: float = 0.6,
        generator_top_p: float = 0.95,
        generator_system_prompt: Optional[str] = None,
        # Fixer configuration.
        fixer_system_prompt: Optional[str] = None,
        # Training behavior.
        fixer_reward_pm1: bool = False,  # False => {0,1}, True => {-1,+1}
        # Include failed test output in fixer prompts
        include_failed_test_output: bool = True,
        # Validation behavior
        use_pregenerated_bugs_in_validation: bool = True,
        # Optional codegen evaluation (validation only)
        evaluate_codegen: bool = True,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.reward_function = reward_function
        self.fixer_reward_pm1 = bool(fixer_reward_pm1)
        self.use_pregenerated_bugs_in_validation = bool(use_pregenerated_bugs_in_validation)
        self.evaluate_codegen = bool(evaluate_codegen)

        # Build frozen generator engine if not provided.
        if generator_rollout_engine is None and generator_model:
            generator_base_url = generator_base_url or "https://api.openai.com/v1"
            generator_rollout_engine = OpenAIEngine(
                model=str(generator_model),
                tokenizer=None,
                base_url=str(generator_base_url),
                api_key=_resolve_api_key(str(generator_base_url), generator_api_key),
                sampling_params={"temperature": float(generator_temperature), "top_p": float(generator_top_p)},
                max_prompt_length=getattr(rollout_engine, "max_prompt_length", 8192),
                max_response_length=getattr(rollout_engine, "max_response_length", 8192),
                verbose=False,
            )

        if generator_rollout_engine is None:
            raise ValueError(
                "FrozenGeneratorFixerWorkflow requires a frozen generator. "
                "Provide generator_rollout_engine or generator_model."
            )

        self.frozen_generator = BugGenerator(
            generator_rollout_engine,
            BugGeneratorConfig(system_prompt=generator_system_prompt),
        )
        self.fixer = BugFixer(
            rollout_engine,
            BugFixerConfig(
                system_prompt=fixer_system_prompt,
                include_failed_test_output=bool(include_failed_test_output),
            ),
        )

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        is_validation = bool(getattr(self.rollout_engine, "validate", False)) or bool(kwargs.get("validate", False))
        task_info = normalize_task_info(task)

        # ---------------------------
        # BUG SOURCE
        # ---------------------------
        bug_traj: Optional[Trajectory] = None
        bug_step: Optional[Step] = None
        buggy_code: Optional[str] = None
        bug_source = "generated"

        # Check for pregenerated bugs first
        pregenerated_bug: Optional[str] = None
        if is_validation and self.use_pregenerated_bugs_in_validation:
            pregenerated_bug = _get_pregenerated_bug(task)

        if pregenerated_bug is not None:
            buggy_code = str(pregenerated_bug)
            bug_source = "pregenerated"
        else:
            # Generate bug using frozen generator
            bug_traj = await self.frozen_generator.generate_bug(task)
            bug_step = bug_traj.steps[0]
            buggy_code = bug_step.action

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
        bug_valid, has_compile_error = check_bug_validity(
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
        # FIXER FIX (single attempt)
        # ---------------------------
        fixer_traj: Optional[Trajectory] = None
        fixer_pass = False

        # Only run fixer on valid bugs (bugs that fail tests and have no compile errors)
        if bug_valid:
            fixer_traj = await self.fixer.fix_bug(
                task,
                buggy_code=buggy_code,
                failed_test_output=failed_test_output,
            )

            fixed_code = fixer_traj.steps[0].action
            try:
                out = self.reward_function(task_info=task_info, action=fixed_code)
                fixer_pass = bool(out.is_correct)
            except Exception:
                fixer_pass = False

            # Assign fixer reward
            if self.fixer_reward_pm1:
                r = 1.0 if fixer_pass else -1.0
            else:
                r = 1.0 if fixer_pass else 0.0
            fixer_traj.steps[0].reward = float(r)

        # Generator gets 0 reward (frozen, not trained)
        if bug_step is not None:
            bug_step.reward = 0.0

        # ---------------------------
        # OPTIONAL CODEGEN EVAL (VAL ONLY)
        # ---------------------------
        codegen_traj: Optional[Trajectory] = None
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
            # No training reward for codegen eval
            codegen_step.reward = 0.0

        # ---------------------------
        # BUILD EPISODE
        # ---------------------------
        metrics: Dict[str, Any] = {
            "fixer_pass": float(fixer_pass),
            "bug_valid": float(bug_valid),
        }

        # Bug test metrics
        if total_tests is not None:
            metrics["bug_total_tests"] = int(total_tests)
        if passed_tests is not None:
            metrics["bug_passed_tests"] = int(passed_tests)

        # Generator reward is always 0 (frozen)
        if bug_traj is not None and bug_step is not None:
            metrics["generator_reward"] = 0.0

        if self.evaluate_codegen:
            metrics["fixer_codegen_pass"] = float(codegen_pass)

        # Store correctness computation parameters in episode.info
        episode_info: Dict[str, Any] = {
            "fixer_pass": bool(fixer_pass),
            "is_validation": bool(is_validation),
            "bug_valid": bool(bug_valid),
            "codegen_pass": bool(codegen_pass),
            "buggy_code": buggy_code,
            "bug_source": bug_source,
            "bug_reward_metadata": bug_meta,
        }

        # Include generator model output if available
        if bug_traj is not None and bug_step is not None:
            bug_model_output = getattr(bug_step, "model_output", None)
            if bug_model_output and hasattr(bug_model_output, "model_dump"):
                episode_info["generator_model_output"] = bug_model_output.model_dump()

        episode = Episode(
            id=uid,
            task=task,
            trajectories=[fixer_traj] if fixer_traj is not None else [],
            is_correct=False,  # set below
            metrics=metrics,
            info=episode_info,
        )

        self.assign_episode_correctness(episode)
        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        """Assign episode correctness based on parameters stored in episode.info."""
        info = episode.info or {}
        fixer_pass = bool(info.get("fixer_pass", False))
        is_validation = bool(info.get("is_validation", False))
        bug_valid = bool(info.get("bug_valid", False))
        codegen_pass = bool(info.get("codegen_pass", False))

        if is_validation:
            # Validation: success if fixer passes (or codegen if enabled)
            if self.evaluate_codegen:
                episode.is_correct = bool(codegen_pass)
            else:
                episode.is_correct = fixer_pass
        else:
            # Training: success if bug is valid and fixer passes
            episode.is_correct = bool(bug_valid and fixer_pass)
