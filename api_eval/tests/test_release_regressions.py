import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


API_EVAL_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, API_EVAL_ROOT)

# Keep these unit tests independent of optional execution-harness packages.
human_eval = types.ModuleType("human_eval")
human_eval.__path__ = []
human_eval_execution = types.ModuleType("human_eval.execution")
human_eval_execution.check_correctness = lambda **kwargs: {"passed": False}
bigcodebench = types.ModuleType("bigcodebench")
bigcodebench.__path__ = []
bigcodebench_eval = types.ModuleType("bigcodebench.eval")
bigcodebench_eval.untrusted_check = lambda **kwargs: ("fail", None)
sys.modules.setdefault("human_eval", human_eval)
sys.modules.setdefault("human_eval.execution", human_eval_execution)
sys.modules.setdefault("bigcodebench", bigcodebench)
sys.modules.setdefault("bigcodebench.eval", bigcodebench_eval)

from unified_eval.input import _get_test_case_feedback
from unified_eval.run_eval import (
    _load_continue_state,
    _pending_rows,
    _require_unique_task_ids,
)


class TestTestCaseFeedback(unittest.TestCase):
    def test_prefers_materialized_feedback(self):
        row = {"mutation_info": {"failed": "assert solve(1) == 2"}}
        self.assertEqual(
            json.loads(_get_test_case_feedback(row, "def solve(): pass")),
            row["mutation_info"],
        )

    def test_generates_feedback_when_column_is_absent(self):
        class FakeMetric:
            def __init__(self, dataset):
                self.dataset = dataset

            def __call__(self, **kwargs):
                return False, ["fail", "AssertionError: expected 2"]

        fake_module = types.ModuleType("src.metrics.pass_rate")
        fake_module.PassRateMetric = FakeMetric
        row = {"ground_truth": "tests", "entry_point": "solve"}
        with patch.dict(sys.modules, {"src.metrics.pass_rate": fake_module}):
            feedback = json.loads(_get_test_case_feedback(row, "def solve(): return 1"))
        self.assertFalse(feedback["passed"])
        self.assertIn("AssertionError", feedback["details"][1])


class TestResumeCoverage(unittest.TestCase):
    def test_absent_and_incomplete_rows_are_both_pending(self):
        rows = [{"uid": "a"}, {"uid": "b"}, {"uid": "c"}]
        task_ids = _require_unique_task_ids(rows, "test")
        pending = _pending_rows(rows, task_ids, {"a": {}, "b": {}}, {"b"})
        self.assertEqual([row["uid"] for row in pending], ["b", "c"])

    def test_continue_file_rejects_duplicate_ids(self):
        payload = {"results": [{"task_id": "a"}, {"task_id": "a"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "duplicate task ID"):
                _load_continue_state(handle.name)


if __name__ == "__main__":
    unittest.main()
