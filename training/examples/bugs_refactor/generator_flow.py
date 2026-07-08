from __future__ import annotations

import os
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from rllm.agents.agent import Episode
from rllm.engine import RolloutEngine, OpenAIEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow

from examples.bugs_refactor.components import (
    BugGenerator,
    BugGeneratorConfig,
    BugFixer,
    BugFixerConfig,
)
from examples.bugs_refactor.utils import (
    normalize_task_info,
    check_bug_validity,
)


class BugGeneratorWorkflow(Workflow):
    """Workflow that trains the BugGenerator against a static fixer.

    High-level logic for each episode:

    1. BugGenerator sees the correct solution and produces a buggy version.
    2. We run unit tests on the buggy code and determine bug validity.
    3. Static fixer tries to fix the buggy code.
    4. We run unit tests on the fixed code.

    Reward:
      - Generator gets reward 1.0 iff:
          * bug is valid (no compile errors; fails >= 1 test)
          * fixer fails to fix it
      - else 0.0
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        reward_function: RewardFunction,
        generator_system_prompt: Optional[str] = None,
        fixer_rollout_engine: Optional[RolloutEngine] = None,
        fixer_model: Optional[str] = None,
        fixer_base_url: Optional[str] = None,
        fixer_temperature: float = 0.0,
        fixer_top_p: float = 1.0,
        fixer_max_prompt_length: Optional[int] = None,
        fixer_max_response_length: Optional[int] = None,
        fixer_system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.reward_function = reward_function
        self.generator = BugGenerator(
            rollout_engine,
            BugGeneratorConfig(system_prompt=generator_system_prompt),
        )

        # Static fixer uses a separate rollout engine (e.g., GPT-4o-mini).
        if fixer_rollout_engine is None and fixer_model:
            fixer_base_url = fixer_base_url or "https://api.openai.com/v1"

            api_key = os.getenv("OPENAI_API_KEY", "") or ""
            api_key = api_key.strip()

            is_openai_api = "api.openai.com" in fixer_base_url
            if is_openai_api and not api_key:
                raise ValueError(
                    "fixer_model was provided with fixer_base_url pointing to the OpenAI API, "
                    "but OPENAI_API_KEY is missing/empty. Please export OPENAI_API_KEY."
                )
            if not api_key:
                api_key = "EMPTY"

            fixer_rollout_engine = OpenAIEngine(
                model=str(fixer_model),
                tokenizer=None,
                base_url=str(fixer_base_url),
                api_key=str(api_key),
                max_prompt_length=(
                    fixer_max_prompt_length
                    if fixer_max_prompt_length is not None
                    else getattr(rollout_engine, "max_prompt_length", 4096)
                ),
                max_response_length=(
                    fixer_max_response_length
                    if fixer_max_response_length is not None
                    else getattr(rollout_engine, "max_response_length", 4096)
                ),
                sampling_params={"temperature": float(fixer_temperature), "top_p": float(fixer_top_p)},
            )

        if fixer_rollout_engine is not None:
            self.fixer = BugFixer(
                fixer_rollout_engine,
                BugFixerConfig(system_prompt=fixer_system_prompt, trajectory_name="bug_fixer"),
            )
        else:
            self.fixer = None

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        # 1) Generate bug
        bug_traj = await self.generator.generate_bug(task)
        bug_step = bug_traj.steps[0]
        buggy_code = bug_step.action

        # 2) Evaluate bug + validity
        task_info = normalize_task_info(task)
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

        # 3) Solver attempts to fix (static baseline)
        fixer_pass = False
        fixer_traj = None
        if self.fixer is not None:
            fixer_traj = await self.fixer.fix_bug(task, buggy_code)
            fixer_step = fixer_traj.steps[0]
            fixed_code = fixer_step.action

            try:
                fixer_reward_output = self.reward_function(task_info=task_info, action=fixed_code)
            except Exception as e:
                fixer_reward_output = RewardOutput(
                    reward=0.0,
                    is_correct=False,
                    metadata={"error": f"fixer reward error: {e}"},
                )
            fixer_pass = bool(fixer_reward_output.is_correct)

        # 4) Generator RL reward
        generator_reward = 1.0 if (bug_valid and not fixer_pass) else 0.0
        bug_step.reward = float(generator_reward)

        trajectories = [bug_traj]
        if fixer_traj is not None:
            trajectories.append(fixer_traj)

        metrics: Dict[str, Any] = {
            "bug_valid": float(bug_valid),
            "generator_reward": float(generator_reward),
            "bug_has_compile_error": float(has_compile_error),
            "fixer_pass": float(fixer_pass),
        }
        if total_tests is not None:
            metrics["bug_total_tests"] = int(total_tests)
        if passed_tests is not None:
            metrics["bug_passed_tests"] = int(passed_tests)

        episode = Episode(
            id=uid,
            task=task,
            trajectories=trajectories,
            is_correct=bool(bug_valid and not fixer_pass),
            metrics=metrics,
        )
        self.assign_episode_correctness(episode)
        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        # Keep episode.is_correct as set in run().
        return
