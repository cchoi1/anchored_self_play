# examples/asp/components.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rllm.agents.agent import Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine

from examples.asp.prompts import (
    _build_bug_generator_prompt,
    _build_bug_fixer_prompt,
    _build_code_generation_prompt,
)
from examples.asp.utils import (
    _get_problem,
    _get_reference_solution,
    _model_text,
    _make_trajectory,
)


# =============================================================================
# Component Configs
# =============================================================================

@dataclass
class BugGeneratorConfig:
    system_prompt: Optional[str] = None
    trajectory_name: str = "bug_generator"
    # Few-shot example bugs to include in prompt (list of dicts with problem, correct_code, buggy_code)
    example_bugs: Optional[List[Dict[str, str]]] = None
    n_example_bugs: int = 3  # Number of examples to include (if example_bugs provided)


@dataclass
class BugFixerConfig:
    system_prompt: Optional[str] = None
    include_failed_test_output: bool = False
    max_failed_test_output_chars: int = 4000
    trajectory_name: str = "bug_fixer"


@dataclass
class CodeGeneratorConfig:
    system_prompt: Optional[str] = None
    append_python_only_instruction: bool = True
    trajectory_name: str = "code_generator"


# Alias for backwards compatibility
CodeSynthesisConfig = CodeGeneratorConfig


# =============================================================================
# Component Classes
# =============================================================================

class BugGenerator:
    """One canonical bug generator used by both GeneratorFixerWorkflow and BugGeneratorWorkflow."""

    def __init__(self, rollout_engine: RolloutEngine, config: Optional[BugGeneratorConfig] = None):
        self.rollout_engine = rollout_engine
        self.config = config or BugGeneratorConfig()

    def _format_example_bugs(self) -> str:
        """Format example bugs for inclusion in the prompt."""
        examples = self.config.example_bugs
        if not examples:
            return ""
        
        n = min(self.config.n_example_bugs, len(examples))
        if n <= 0:
            return ""
        
        # Select examples (could be random, but deterministic for reproducibility)
        selected = examples[:n]
        
        example_strs = []
        for i, ex in enumerate(selected, 1):
            problem = ex.get("problem", "").strip()
            correct = ex.get("correct_code", "").strip()
            buggy = ex.get("buggy_code", "").strip()
            
            if not buggy:
                continue
                
            ex_str = f"### Example {i}\n"
            if problem:
                ex_str += f"**Problem:**\n{problem}\n\n"
            if correct:
                ex_str += f"**Correct solution:**\n```python\n{correct}\n```\n\n"
            ex_str += f"**Buggy version:**\n```python\n{buggy}\n```"
            example_strs.append(ex_str)
        
        if not example_strs:
            return ""
        
        return (
            "\n\n---\n\n"
            "Here are some examples of the style of subtle bugs you should introduce:\n\n"
            + "\n\n".join(example_strs)
            + "\n\n---\n\n"
            "Now generate a bug in a similar style for the following problem:\n"
        )

    async def generate_bug(self, task: Dict[str, Any]) -> Trajectory:
        problem = _get_problem(task)
        correct_code = _get_reference_solution(task)
        if not correct_code:
            raise KeyError("Task missing reference_solution/canonical_solution required for bug generation.")

        # Build base prompt
        prompt = _build_bug_generator_prompt(problem, correct_code)
        
        # Insert examples before the main task if available
        example_section = self._format_example_bugs()
        if example_section:
            # Insert examples after the instructions but before "Problem:"
            # Find where "Problem:" starts and insert examples before it
            problem_marker = "\nProblem:\n"
            if problem_marker in prompt:
                idx = prompt.find(problem_marker)
                prompt = prompt[:idx] + example_section + prompt[idx:]
            else:
                # Fallback: prepend examples to the whole prompt
                prompt = example_section + prompt
        print(f"Prompt: {prompt}")
        
        messages: List[Dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        out: ModelOutput = await self.rollout_engine.get_model_response(messages)
        buggy_code = _model_text(out)
        return _make_trajectory(
            name=self.config.trajectory_name,
            messages=messages,
            assistant_text=buggy_code,
            model_output=out,
        )


class BugFixer:
    """
    One canonical bug fixer used by:
      - GeneratorFixerWorkflow
      - FixerWorkflow 
      - BugGeneratorWorkflow (as static fixer baseline)
    """

    def __init__(self, rollout_engine: RolloutEngine, config: Optional[BugFixerConfig] = None):
        self.rollout_engine = rollout_engine
        self.config = config or BugFixerConfig()

    async def fix_bug(
        self,
        task: Dict[str, Any],
        buggy_code: str,
        failed_test_output: Optional[str] = None,
    ) -> Trajectory:
        """Generate a single fix attempt. Use fix_bugs() for multiple attempts."""
        problem = _get_problem(task)
        prompt = _build_bug_fixer_prompt(
            problem,
            buggy_code,
            include_failed_test_output=bool(self.config.include_failed_test_output),
            failed_test_output=failed_test_output,
        )

        messages: List[Dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        out: ModelOutput = await self.rollout_engine.get_model_response(messages)
        fixed_code = _model_text(out)
        return _make_trajectory(
            name=self.config.trajectory_name,
            messages=messages,
            assistant_text=fixed_code,
            model_output=out,
        )

    async def generate_code(self, task: Dict[str, Any]) -> Trajectory:
        prompt = _build_code_generation_prompt(task)

        messages: List[Dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        out: ModelOutput = await self.rollout_engine.get_model_response(messages)
        code = _model_text(out)
        return _make_trajectory(
            name=CodeGeneratorConfig.trajectory_name,
            messages=messages,
            assistant_text=code,
            model_output=out,
        )


class CodeSynthesizer:
    """Canonical code-from-scratch synthesis."""

    def __init__(self, rollout_engine: RolloutEngine, config: Optional[CodeGeneratorConfig] = None):
        self.rollout_engine = rollout_engine
        self.config = config or CodeGeneratorConfig()

    async def synthesize(self, task: Dict[str, Any]) -> Tuple[str, ModelOutput]:
        prompt = _build_code_generation_prompt(task)

        messages: List[Dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        out: ModelOutput = await self.rollout_engine.get_model_response(messages)
        code = _model_text(out)
        return code, out
