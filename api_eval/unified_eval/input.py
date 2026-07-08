import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.prompt_handler import PromptHandler

# in context examples
EXAMPLES = {
    "complete": (
        "Example:\n"
        "Buggy Implementation:\n"
        "def add(a, b):\n"
        "    return a - b\n\n"
        "Response:\n"
        "```python\n"
        "    return a + b\n"
        "```\n\n"
        "Explanation: Fixed the incorrect subtraction operator.\n\n"
    ),
    "instruct": (
        "Example:\n"
        "Problem:\nWrite a function to add two numbers.\n"
        "Buggy Implementation:\n"
        "def add(a, b):\n"
        "    return a - b\n\n"
        "Response:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Explanation: Fixed the incorrect subtraction operator.\n\n"
    ),
    "coder": (
        "Example:\n"
        "def add(a, b):\n\n"
        "Response:\n"
        "```python\n"
        "    return a + b\n"
        "```\n\n"
    ),
    "ambig_complete": (
        "Example:\n"
        "Implementation:\n"
        "def add(a, b):\n"
        "    return a - b\n\n"
        "Response:\n"
        "    return a + b\n"
        "```\n\n"
        "Explanation: Fixed subtraction to addition.\n\n"
        "Example:\n"
        "Implementation:\n"
        "def add(a, b):\n"
        "    return a + b\n\n"
        "Response:\n"
        "    return a + b\n"
        "```\n\n"
        "Explanation: The implementation is correct.\n\n"
    ),
    "ambig_instruct": (
        "Example:\n"
        "Problem:\nWrite a function to add two numbers.\n"
        "Implementation:\n"
        "def add(a, b):\n"
        "    return a - b\n\n"
        "Response:\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Explanation: Fixed subtraction to addition.\n\n"
        "Example:\n"
        "Problem:\nWrite a function to add two numbers.\n"
        "Implementation:\n"
        "def add(a, b):\n"
        "    return a + b\n\n"
        "Response:\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Explanation: The implementation is correct.\n\n"
    ),
}

DEBUGGER_COMPLETE = PromptHandler(
    template=(
        "Your task is to fix the buggy implementation of a function.\n\n"
        "Rules:\n"
        "1. Respond with the entire corrected function body.\n"
        "2. Do not include function headers, docstrings, comments, or tests.\n"
        "3. Preserve correct parts of the original.\n\n"
        "Response Format:\n"
        "1. Corrected function body in ```python ```\n"
        "2. Brief explanation prefixed with 'Explanation:'\n\n"
        "{example}"
        "Buggy Implementation:\n"
        "{prompt}"
        "{mutation}\n\n"
        "Response:\n\n"
    ),
    input_keys=["example", "prompt", "mutation"],
    output_format=str,
    strict_input=False,
    name="DEBUGGER_COMPLETE",
)

DEBUGGER_INSTRUCT = PromptHandler(
    template=(
        "Your task is to fix buggy code.\n\n"
        "Respond with:\n"
        "1. The entire correct code in ```python ```\n"
        "2. Brief explanation prefixed with 'Explanation:'\n\n"
        "{example}"
        "Problem:\n"
        "{prompt}\n"
        "Buggy Implementation:\n"
        "{mutation}\n\n"
        "Response:\n\n"
    ),
    input_keys=["example", "prompt", "mutation"],
    output_format=str,
    strict_input=False,
    name="DEBUGGER_INSTRUCT",
)

CODER_COMPLETE = PromptHandler(
    template=(
        "Complete the function below. Do not include the function header. "
        "Surround code with ```python ```.\n\n"
        "{example}"
        "{prompt}\n"
        "Response:\n"
    ),
    input_keys=["example", "prompt"],
    output_format=str,
    strict_input=False,
    name="CODER_COMPLETE",
)

CODER_INSTRUCT = PromptHandler(
    template=("{prompt}\nSurround code with ```python ```.\n"),
    input_keys=["prompt"],
    output_format=str,
    strict_input=False,
    name="CODER_INSTRUCT",
)

DEBUGGER_TRAINING = PromptHandler(
    template=(
        "You are an expert Python debugging assistant.\n\n"
        "You will be given:\n\n"
        "1. A problem description.\n"
        "2. A buggy Python implementation that may fail some hidden unit tests.\n\n"
        "Your task:\n\n"
        "- Carefully read the code and identify the bug(s).\n"
        "- Produce a fixed version of the code that makes all unit tests pass.\n"
        "- Preserve the original function signature, imports, and I/O format.\n"
        "- Keep the solution reasonably close to the given implementation.\n"
        "- Output **only** the full corrected Python code inside a single ```python``` block.\n\n"
        "Problem:\n{prompt}\n\n"
        "Buggy implementation:\n{mutation}\n\n"
        "Now fix the bugs in this code. Return the entire function with the fixed code inside a ```python``` block:"
    ),
    input_keys=["prompt", "mutation"],
    output_format=str,
    strict_input=False,
    name="DEBUGGER_TRAINING",
)

DEBUGGER_AMBIG_COMPLETE = PromptHandler(
    template=(
        "You are given an implementation which may or may not be buggy. "
        "If buggy, fix it. If correct, return as is.\n\n"
        "Rules:\n"
        "1. Respond with the entire correct function body.\n"
        "2. Do not include function headers, docstrings, or tests.\n\n"
        "Response Format:\n"
        "1. Function body in ```\n"
        "2. Brief explanation prefixed with 'Explanation:'\n\n"
        "{example}"
        "Implementation:\n"
        "{prompt}"
        "{mutation}\n\n"
        "Response:\n\n"
    ),
    input_keys=["example", "prompt", "mutation"],
    output_format=str,
    strict_input=False,
    name="DEBUGGER_AMBIG_COMPLETE",
)

DEBUGGER_AMBIG_INSTRUCT = PromptHandler(
    template=(
        "You are given an implementation which may or may not be buggy. "
        "If buggy, fix it. If correct, return as is.\n\n"
        "Respond with:\n"
        "1. The entire correct code in ```\n"
        "2. Brief explanation prefixed with 'Explanation:'\n\n"
        "{example}"
        "Problem:\n{prompt}\n"
        "Implementation:\n{mutation}\n\n"
        "Response:\n\n"
    ),
    input_keys=["example", "prompt", "mutation"],
    output_format=str,
    strict_input=False,
    name="DEBUGGER_AMBIG_INSTRUCT",
)

DIFF_HANDLER = PromptHandler(
    template=(
        "You are an expert Python debugging assistant.\n\n"
        "You will be given:\n"
        "1. A problem description.\n"
        "2. A buggy Python implementation.\n\n"
        "Your task:\n"
        "- Identify the bug(s) and produce a unified git diff to fix the bugs.\n"
        "- Ensure the correct function signature, imports, and I/O format.\n"
        "- Output one unified diff in a ```diff fenced block.\n\n"
        "- Use code without line number prefixes.\n\n" 
        "Problem:\n{prompt}\n\n"
        "Buggy Implementation:\n{mutation}\n\n"
        "Return the unified diff in a ```diff``` block:\n\n"
    ),
    input_keys=["prompt", "mutation"],
    output_format=str,
    strict_input=False,
    name="DIFF_HANDLER",
)

SOLVE_THEN_PATCH = PromptHandler(
     template=(
        "You are an expert Python debugging assistant.\n\n"
        "You will be given:\n"
        "1. A problem description.\n"
        "2. A buggy Python implementation.\n\n"
        "Your task:\n"
        "FIRST: Write a fixed code solution from scratch.\n"
        "- Provide the full corrected code in a ```python fenced block.\n"
        "- Use code without line number prefixes.\n\n" 
        "SECOND: Produce a unified git diff to patch the given code to match your solution.\n"
        "- Output one unified diff in a ```diff fenched block.\n\n"
        "- Use code without line number prefixes.\n\n" 
        "- Ensure the correct function signature, imports, and I/O format.\n\n"
        "Problem:\n{prompt}\n\n"
        "Buggy implementation:\n{mutation}\n\n"
        "Fix the bugs. Return the correct function inside a ```python``` block annd the unified diff in a ```diff``` block:"
    ),
    input_keys=["prompt", "mutation"],
    output_format=str,
    strict_input=False,
    name="2_STAGE"
)


TEST_CASES = PromptHandler(
    template=(
        "You are an expert Python debugging assistant.\n\n"
        "You will be given:\n"
        "1. A problem description.\n"
        "2. A buggy Python implementation.\n\n"
        "3. Its results on the problem's test cases.\n\n"
        "Your task:\n"
        "- Identify the bug(s) and produce a fixed version.\n"
        "- Preserve the original function signature, imports, and I/O format.\n"
        "- Output the full corrected code in a ```python``` block.\n\n"
        "Problem:\n{prompt}\n\n"
        "Buggy implementation:\n{mutation}\n\n"
        "Test case results:\n{test_cases}\n\n"
        "Fix the bugs. Return the entire function inside a ```python``` block:"
    ),
    input_keys=["prompt", "mutation", "test_cases"],
    output_format=str,  
    strict_input=False,
    name="TEST_CASES",
)

MODE_HANDLERS = {
    "solver-complete": DEBUGGER_COMPLETE,
    "solver-instruct": DEBUGGER_INSTRUCT,
    "solver-ambig-complete": DEBUGGER_AMBIG_COMPLETE,
    "solver-ambig-instruct": DEBUGGER_AMBIG_INSTRUCT,
    "solver-diff": DIFF_HANDLER,
    "solver-attacker-style": DEBUGGER_TRAINING,
    "solver-test-cases": TEST_CASES,
    "solver-solve-then-patch": SOLVE_THEN_PATCH,
    "coder-complete": CODER_COMPLETE,
    "coder-instruct": CODER_INSTRUCT,
}

EXAMPLE_MAP = {
    DEBUGGER_COMPLETE: "complete",
    DEBUGGER_INSTRUCT: "instruct",
    DEBUGGER_AMBIG_COMPLETE: "ambig_complete",
    DEBUGGER_AMBIG_INSTRUCT: "ambig_instruct",
    CODER_COMPLETE: "coder",
}

def _nl(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _strip_fences(code: str) -> str:
    if not code or "```" not in code:
        return code
    try:
        start = code.index("```")
        end = code.index("```", start + 3)
        fenced = code[start + 3:end].splitlines()
        if fenced and fenced[0].strip().isalpha():
            fenced = fenced[1:]
        return "\n".join(fenced)
    except ValueError:
        return code.replace("```", "")


def prepare_prompt(
    row: Dict[str, Any],
    mode: str,
    mutation_column: str = "response",
    canonical_control: bool = False,
    example: bool = True,
) -> Dict[str, Any]:
    handler = MODE_HANDLERS.get(mode)
    task_id = row.get("task_id") or row.get("uid") or row.get("1", "unknown")
    
    example_text = ""
    if example and handler in EXAMPLE_MAP:
        example_text = EXAMPLES[EXAMPLE_MAP[handler]]
    
    complete_handlers = (DEBUGGER_COMPLETE, DEBUGGER_AMBIG_COMPLETE, CODER_COMPLETE)
    instruct_handlers = (DEBUGGER_INSTRUCT, DEBUGGER_AMBIG_INSTRUCT, CODER_INSTRUCT, DEBUGGER_TRAINING, DIFF_HANDLER, TEST_CASES, SOLVE_THEN_PATCH)
    
    if handler in complete_handlers:
        prompt = _nl(row.get("complete_prompt"))
    elif handler in instruct_handlers:
        prompt = _nl(row.get("instruct_prompt"))
    else:
        prompt = ""
    
    mutation = ""
    if handler not in (CODER_COMPLETE, CODER_INSTRUCT):
        code_header = _nl(row.get("code_prompt") or row.get("starter_code", "") or "")
        if canonical_control:
            canonical = row.get("canonical_solution") or row.get("reference_solution") or ""
            mutation_base = _strip_fences(canonical) if canonical else ""
        else:
            raw_mut = row.get(mutation_column)
            if raw_mut and "```" in raw_mut:
                try:
                    start = raw_mut.index("```")
                    end = raw_mut.index("```", start + 3)
                except ValueError:
                    raw_mut = raw_mut.replace("```", "")
                else:
                    fenced = raw_mut[start + 3:end].splitlines()
                    if fenced and fenced[0].strip().isalpha():
                        fenced = fenced[1:]
                    raw_mut = raw_mut[:start] + "\n".join(fenced) + raw_mut[end + 3:]
            mutation_base = raw_mut or f"{code_header}{row.get('buggy') or ''}" 
        # clean mutation
        lines = _nl(mutation_base).splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        mutation = "\n".join(lines) + "\n" if lines else ""
    
    if handler in (DIFF_HANDLER, SOLVE_THEN_PATCH):
        # add line numbers to mutation for diff-based modes
        lines = mutation.splitlines() if mutation else []
        numbered = "\n".join(f"{i+1:04d}: {ln}" for i, ln in enumerate(lines))
        numbered = numbered + ("\n" if mutation.endswith("\n") else "") if mutation else ""
        rendered = handler.template.format(prompt=prompt, mutation=numbered)
    elif handler is TEST_CASES:
        # get test case results from mutation_info column
        test_cases = row.get("mutation_info", "") or ""
        rendered = handler.template.format(prompt=prompt, mutation=mutation, test_cases=test_cases)
    elif handler:
        rendered = handler.template.format(example=example_text, prompt=prompt, mutation=mutation)
    else:
        prompt = _nl(row.get(mode, "") or "")
        rendered = prompt
    
    canonical = row.get("canonical_solution") or row.get("reference_solution", "")
    problem = {
        "task_id": task_id,
        "prompt_text": rendered,
        "test": row.get("test") or row.get("ground_truth", ""),
        "entry_point": row.get("entry_point", ""),
        "canonical_solution": _strip_fences(canonical) if canonical else "",
        "complete_prompt": row.get("complete_prompt", ""),
        "instruct_prompt": row.get("instruct_prompt", ""),
        "code_prompt": row.get("code_prompt") or row.get("starter_code", ""),
    }
    
    return {
        "task_id": task_id,
        "handler": handler,
        "rendered_prompt": rendered,
        "prompt_text": prompt,
        "mutation": mutation,
        "problem": problem,
        "mode": mode,
    }


def dataset_for_mode(mode: str) -> str:
    instruct_modes = {"solver-attacker-style", "solver-test-cases", "solver-solve-then-patch"}
    if mode == "complete_prompt" or "instruct" in mode or "diff" in mode or mode in instruct_modes:
        return "bigcodebench-instruct"
    return "bigcodebench-complete"


def load_data(input_file: str):
    if ".csv" in input_file:
        import pandas as pd
        return pd.read_csv(input_file).to_dict("records")
    from datasets import load_dataset
    split = "test"
    return [dict(row) for row in load_dataset(input_file, split=split)]
