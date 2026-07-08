from __future__ import annotations

import os
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from rllm.agents.agent import Episode
from rllm.engine import ModelOutput, RolloutEngine, OpenAIEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow, TerminationReason

from examples.bugs_refactor.components import (
    BugFixer,
    BugFixerConfig,
    CodeSynthesizer,
    CodeSynthesisConfig,
)
from examples.bugs_refactor.utils import (
    normalize_task_info,
    maybe_extract_failed_test_output,
    _get_pregenerated_bug,
    _pass_ratio,
    _resolve_api_key,
)


class FixerWorkflow(Workflow):
    """Train a fixer on a frozen CodeSynthesizer's failures.
 
    Two modes:

    (A) Failure-only (only_train_on_failures=True) [default]
      1) Frozen CodeSynthesizer synthesizes code for the task.
      2) Reward function evaluates CodeSynthesizer output.  (unit tests run)
      3) If CodeSynthesizer fails, trainable fixer repairs the CodeSynthesizer output.
      4) Reward function evaluates fixed output.   (unit tests run again)
      5) Fixer reward = 1 iff (CodeSynthesizer failed) AND (fix passes all tests).

    (B) One-shot (only_train_on_failures=False)
      1) Frozen CodeSynthesizer synthesizes code for the task.
      2) Fixer always proposes a refined/fixed program.
      3) Reward function evaluates ONLY the final submitted code ONCE.
         (No CodeSynthesizer evaluation, no intermediate tests.)

    Notes:
      - Only the fixer trajectory is returned (frozen CodeSynthesizer is not trained).
      - If only_train_on_failures=True, CodeSynthesizer successes yield empty episodes (dropped) in training mode.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        reward_function: RewardFunction,
        # Frozen synthesizer configuration.
        synthesizer_rollout_engine: Optional[RolloutEngine] = None,
        synthesizer_model: Optional[str] = None,
        synthesizer_base_url: Optional[str] = None,
        synthesizer_api_key: Optional[str] = None,
        synthesizer_temperature: float = 0.6,
        synthesizer_top_p: float = 0.95,
        synthesizer_system_prompt: Optional[str] = None,
        # Fixer configuration.
        fixer_system_prompt: Optional[str] = None,
        # Training behavior.
        only_train_on_failures: bool = True,
        reward_pm1: bool = False,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.reward_function = reward_function
        self.only_train_on_failures = bool(only_train_on_failures)
        self.reward_pm1 = bool(reward_pm1)

        # Build frozen synthesizer engine if not provided.
        if synthesizer_rollout_engine is None and synthesizer_model:
            synthesizer_base_url = synthesizer_base_url or "https://api.openai.com/v1"
            synthesizer_rollout_engine = OpenAIEngine(
                model=str(synthesizer_model),
                tokenizer=None,
                base_url=str(synthesizer_base_url),
                api_key=_resolve_api_key(str(synthesizer_base_url), synthesizer_api_key),
                sampling_params={"temperature": float(synthesizer_temperature), "top_p": float(synthesizer_top_p)},
                max_prompt_length=getattr(rollout_engine, "max_prompt_length", 8192),
                max_response_length=getattr(rollout_engine, "max_response_length", 8192),
                verbose=False,
            )

        if synthesizer_rollout_engine is None:
            raise ValueError("FixerWorkflow requires a frozen synthesizer. Provide synthesizer_rollout_engine or synthesizer_model.")

        self.frozen_synthesizer = CodeSynthesizer(
            synthesizer_rollout_engine,
            CodeSynthesisConfig(system_prompt=synthesizer_system_prompt),
        )
        self.fixer = BugFixer(
            rollout_engine,
            BugFixerConfig(
                system_prompt=fixer_system_prompt,
                include_failed_test_output=bool(kwargs.get("include_failed_test_output", False)),
                max_failed_test_output_chars=int(kwargs.get("max_failed_test_output_chars", 4000)),
                trajectory_name="fixer",  # preserve old name
            ),
        )

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task, uid)
        task_info = normalize_task_info(task)

        # If only_train_on_failures=False => one-shot mode.
        one_shot_mode = (not self.only_train_on_failures)

        validate_mode = bool(getattr(self.rollout_engine, "validate", False)) or bool(kwargs.get("validate", False))
        bug_source = "generated"
        synthesizer_out: Optional[ModelOutput] = None

        # Check for pregenerated bugs first
        pregenerated_bug: Optional[str] = None
        if validate_mode:
            pregenerated_bug = _get_pregenerated_bug(task_info)
        
        # Only generate synthesizer code if we don't have a pregenerated bug.
        # When evaluating on pregenerated bugs, skip synthesizer generation entirely.
        if pregenerated_bug is not None:
            synthesizer_code: Optional[str] = None
            buggy_code_for_fixer = str(pregenerated_bug)
            bug_source = "pregenerated"
        else:
            # Generate fresh code for synthesizer evaluation (for fair comparison with fixer_codegen_pass).
            synthesizer_code, synthesizer_out = await self.frozen_synthesizer.synthesize(task_info)
            buggy_code_for_fixer = synthesizer_code

        # ---------------------------
        # (A) Failure-only evaluation (optional)
        # ---------------------------
        synthesizer_pass: Optional[bool]
        synthesizer_meta: Dict[str, Any]
        synthesizer_ratio: Optional[float]

        if one_shot_mode:
            # One-shot protocol: do NOT run tests on synthesizer output.
            synthesizer_pass = None
            synthesizer_meta = {}
            synthesizer_ratio = None
        elif synthesizer_code is None:
            # No synthesizer code generated (using pregenerated bug, skipped synthesizer generation).
            synthesizer_pass = None
            synthesizer_meta = {}
            synthesizer_ratio = None
        else:
            # Evaluate synthesizer/buggy code (first test run).
            try:
                synthesizer_reward_out: RewardOutput = self.reward_function(task_info=task_info, action=synthesizer_code)
            except Exception as e:
                synthesizer_reward_out = RewardOutput(reward=0.0, is_correct=False, metadata={"error": str(e)})

            synthesizer_pass = bool(synthesizer_reward_out.is_correct)
            synthesizer_meta = synthesizer_reward_out.metadata or {}
            synthesizer_ratio = _pass_ratio(synthesizer_meta)

            # Early exit in training if synthesizer passed and we only train on failures.
            if (not validate_mode) and synthesizer_pass and self.only_train_on_failures:
                metrics: Dict[str, Any] = {
                    "synthesizer_pass": 1.0,
                    "fixer_pass": 0.0,
                    "synthesizer_pass_ratio": 1.0,
                    "fixer_pass_ratio": 0.0,
                }
                return Episode(
                    id=uid,
                    task=task,
                    trajectories=[],
                    termination_reason=TerminationReason.ENV_DONE,
                    is_correct=True,
                    metrics=metrics,
                    info={
                        "buggy_code": buggy_code_for_fixer,
                        "bug_source": bug_source,
                        "synthesizer_reward_metadata": synthesizer_meta,
                    },
                )

        # ---------------------------
        # Run fixer (always in one-shot; conditional in failure-only via early exit above).
        # ---------------------------
        failed_test_output = None
        if (not one_shot_mode):
            # If using a pregenerated bug for fixer, evaluate it to get failed test output.
            # Otherwise, use synthesizer evaluation results.
            if pregenerated_bug is not None:
                # We're using a pregenerated bug in validation mode - evaluate it for failed test output.
                try:
                    bug_reward_out: RewardOutput = self.reward_function(task_info=task_info, action=buggy_code_for_fixer)
                    bug_meta = bug_reward_out.metadata or {}
                    if not bool(bug_reward_out.is_correct):
                        failed_test_output = maybe_extract_failed_test_output(
                            bug_meta,
                            enabled=bool(self.fixer.config.include_failed_test_output),
                            max_chars=int(self.fixer.config.max_failed_test_output_chars),
                        )
                except Exception:
                    pass  # If evaluation fails, failed_test_output stays None
            elif synthesizer_code is not None and not bool(synthesizer_pass):
                # Using synthesized code and it failed - extract from synthesizer metadata.
                failed_test_output = maybe_extract_failed_test_output(
                    synthesizer_meta,
                    enabled=bool(self.fixer.config.include_failed_test_output),
                    max_chars=int(self.fixer.config.max_failed_test_output_chars),
                )

        fixer_traj = await self.fixer.fix_bug(
            task_info,
            buggy_code=buggy_code_for_fixer,
            failed_test_output=failed_test_output,
        )
        fixer_step = fixer_traj.steps[0]
        fixed_code = fixer_step.action

        # Evaluate final submitted code.
        # - One-shot: this is the ONLY test run.
        # - Failure-only: this is the second test run.
        try:
            fix_reward_out: RewardOutput = self.reward_function(task_info=task_info, action=fixed_code)
        except Exception as e:
            fix_reward_out = RewardOutput(reward=0.0, is_correct=False, metadata={"error": str(e)})

        fix_pass = bool(fix_reward_out.is_correct)
        fix_meta = fix_reward_out.metadata or {}
        fix_ratio = _pass_ratio(fix_meta)

        # Compute fixer reward.
        if one_shot_mode:
            fixer_reward = (1.0 if fix_pass else -1.0) if self.reward_pm1 else (1.0 if fix_pass else 0.0)
        else:
            # If synthesizer_pass is None (e.g., using pregenerated bug), treat as if synthesizer failed
            if synthesizer_pass is not None and bool(synthesizer_pass):
                fixer_reward = -1.0 if self.reward_pm1 else 0.0
            else:
                fixer_reward = (1.0 if fix_pass else -1.0) if self.reward_pm1 else (1.0 if fix_pass else 0.0)

        fixer_step.reward = fixer_traj.reward = float(fixer_reward)

        # Metrics
        metrics: Dict[str, Any] = {"fixer_pass": float(fix_pass)}
        if one_shot_mode:
            metrics["one_shot"] = 1.0
            metrics["synthesizer_pass"] = None
            metrics["synthesizer_pass_ratio"] = None
        else:
            if synthesizer_pass is not None:
                metrics["synthesizer_pass"] = float(bool(synthesizer_pass))
            else:
                metrics["synthesizer_pass"] = None  # Skipped (using pregenerated bug)
            if synthesizer_ratio is not None:
                metrics["synthesizer_pass_ratio"] = synthesizer_ratio

        if fix_ratio is not None:
            metrics["fixer_pass_ratio"] = fix_ratio

        episode = Episode(
            id=uid,
            task=task,
            trajectories=[fixer_traj],
            is_correct=(fixer_reward > 0.0) if self.reward_pm1 else bool(fixer_reward),
            metrics=metrics,
            info={
                "buggy_code": buggy_code_for_fixer,
                "bug_source": bug_source,
                "synthesizer_reward_metadata": (synthesizer_meta if (not one_shot_mode) else None),
                "fixed_reward_metadata": fix_meta,
                "synthesizer_model_output": synthesizer_out.model_dump() if synthesizer_out and hasattr(synthesizer_out, "model_dump") else None,
            },
        )
        self.assign_episode_correctness(episode)
        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        m = episode.metrics or {}
        is_validation = bool(m.get("is_validation", False))
        if is_validation:
            episode.is_correct = bool(m.get("fixer_pass", 0.0))
        else:
            if self.only_train_on_failures:
                # "Correct" means: synthesizer failed AND fixer succeeded.
                episode.is_correct = bool(m.get("fixer_pass", 0.0)) and not bool(m.get("synthesizer_pass", 0.0))
            else:
                # "Correct" means: final submitted code passed.
                episode.is_correct = bool(m.get("fixer_pass", 0.0))
