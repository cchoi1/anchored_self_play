from typing import Any, Dict
from human_eval.execution import check_correctness as humaneval_check_correctness
from bigcodebench.eval import untrusted_check
from src.utils.logger_utils import setup_logger
from src.utils.execute.mbpp import check_correctness as mbpp_check_correctness
from src.utils.execute.kodcode.testing_util import check_correctness as kodcode_check_correctness

logger = setup_logger()

class PassRateMetric:
    """Metric to compute whether a completion passes the test cases."""

    def __init__(self, dataset: str, timeout: int = 5):
        """
        Initialize the PassRateMetric with dataset and timeout.

        Args:
            dataset (str): Dataset identifier ('humaneval', 'bigcodebench', 'apps', 'ds1000').
            timeout (int): Timeout in seconds for verification (default: 5).
        """
        super().__init__()
        self.dataset = dataset
        self.timeout = timeout

    def __call__(self, problem: Dict[str, Any], completion: str, **kwargs) -> Dict[str, Any]:
        """
        Compute pass/fail status for a completion.

        Args:
            problem: Problem dictionary.
            completion: Generated code.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, Any]: Dictionary with 'pass_rate' mapped to True/False.
        """
        try:
            if self.dataset == "bigcodebench-complete":
                if "complete_prompt" in problem:
                    prefix = problem["complete_prompt"]
                    # Check if completion already contains the function header
                    # (instruction-tuned models often ignore "don't include header")
                    entry_point = problem.get("entry_point", "")
                    completion_has_header = (
                        entry_point and 
                        f"def {entry_point}" in completion[:500]
                    )
                    code = completion if completion_has_header else prefix + completion
                else:
                    code = problem["prompt"] + completion  
                result = untrusted_check(
                    code=code,
                    test_code=problem["test"],
                    entry_point=problem["entry_point"],
                    max_as_limit=300*1024,
                    max_data_limit=300*1024,
                    max_stack_limit=300*1024,
                    min_time_limit=5,
                    gt_time_limit=5
                )
                return result[0] == 'pass', result
            elif self.dataset == "bigcodebench-instruct":
                result = untrusted_check(
                    code=completion,
                    test_code=problem["test"],
                    entry_point=problem["entry_point"],
                    max_as_limit=300*1024,
                    max_data_limit=300*1024,
                    max_stack_limit=300*1024,
                    min_time_limit=5,
                    gt_time_limit=5
                )
                return result[0] == 'pass', result
            elif self.dataset == "humaneval":
                result = humaneval_check_correctness(problem=problem, completion=completion, timeout=self.timeout)
                return result['passed'], result
            elif self.dataset == "mbpp":
                result = mbpp_check_correctness(check_program=completion, timeout=self.timeout, task_id=problem['task_id'], completion_id=problem['task_id'])
                return result['passed'], result
            elif self.dataset == "kodcode-complete":
                code = completion
                result = kodcode_check_correctness(problem=problem, completion=code, timeout=self.timeout)
                return result['passed'], result
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset}")
        except Exception as e:
            logger.error(f"Error computing pass_rate: {e}")
            return False, {"error": str(e)}

