import contextlib
import faulthandler
import io
import multiprocessing as mp
import os
import platform
import signal
import tempfile
import traceback
from typing import Dict, Optional


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    faulthandler.disable()

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    import builtins
    builtins.exit = None
    builtins.quit = None
    builtins.help = None

    os.environ["OMP_NUM_THREADS"] = "1"

    import subprocess
    subprocess.Popen = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["tkinter"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None


def _run_tests(problem: Dict, completion: str, timeout: float, res_list):
    with create_tempdir():
        with open("solution.py", "w") as f:
            f.write(completion)

        import sys
        sys.path.insert(0, os.getcwd())
        reliability_guard()

        test_code = problem["test"]
        if "from solution import" not in test_code:
            test_code = "from solution import *\n" + test_code

        try:
            exec_globals: Dict[str, object] = {}
            with swallow_io(), time_limit(timeout):
                exec(test_code, exec_globals)
                test_funcs = [
                    v for v in exec_globals.values()
                    if callable(v) and getattr(v, "__name__", "").startswith("test_")
                ]
                total = len(test_funcs) or 1
                passed = 0
                tracebacks = []
                for fn in test_funcs:
                    try:
                        fn()
                        passed += 1
                    except Exception:
                        tracebacks.append(traceback.format_exc())

            reward = passed / total
            res_list.append({
                "status": "ok",
                "passed": passed,
                "total": total,
                "reward": reward,
                "tracebacks": tracebacks,
            })

        except TimeoutException:
            res_list.append({
                "status": "timeout",
                "passed": 0,
                "total": 1,
                "reward": 0.0,
                "tracebacks": [f"Timed out after {timeout:.1f} s"],
            })

        except BaseException:
            tb = traceback.format_exc()
            res_list.append({
                "status": "failed",
                "passed": 0,
                "total": 1,
                "reward": 0.0,
                "tracebacks": [tb],
            })


def _run_tests_queue(problem: Dict, completion: str, timeout: float, result_queue):
    res_list = []
    try:
        _run_tests(problem, completion, timeout, res_list)
        if res_list:
            result_queue.put(res_list[0])
        else:
            result_queue.put({
                "status": "failed",
                "passed": 0,
                "total": 1,
                "reward": 0.0,
                "tracebacks": ["No result returned"],
            })
    except Exception as e:
        result_queue.put({
            "status": "failed",
            "passed": 0,
            "total": 1,
            "reward": 0.0,
            "tracebacks": [str(e)],
        })


def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """Returns a dict with fractional reward and stats for KodCode solution."""
    result_queue = mp.Queue()
    p = mp.Process(target=_run_tests_queue, args=(problem, completion, timeout, result_queue))
    p.start()
    p.join(timeout + 2)  # grace margin

    if p.is_alive():
        p.kill()
        p.join(1)

    res = None
    try:
        res = result_queue.get(timeout=1.0)
    except Exception:
        pass

    try:
        result_queue.close()
        result_queue.join_thread()
    except Exception:
        pass

    if res is None:
        res = {
            "status": "timeout",
            "passed": 0,
            "total": 1,
            "reward": 0.0,
            "tracebacks": [f"Timed out after {timeout:.1f} s"],
        }

    return {
        "task_id": problem.get("task_id", ""),
        "passed": res["passed"] == res["total"],
        "result": res["status"],
        "completion_id": completion_id,
        "tests_passed": res["passed"],
        "tests_total": res["total"],
        "reward": res["reward"],
        "tracebacks": res.get("tracebacks", []),
    }

