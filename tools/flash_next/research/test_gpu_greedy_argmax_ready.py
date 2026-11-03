import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).parent


class ReadyBenchmarkTests(unittest.TestCase):
    def test_prefill_waits_for_gpu_without_score_materialization(self):
        tree = ast.parse((ROOT / "benchmark_gpu_greedy_argmax_ready.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_request")
        clock = SimpleNamespace(now=0.0)
        usage = SimpleNamespace(**{name: 0 for name in (
            "ru_majflt", "ru_minflt", "ru_utime", "ru_stime", "ru_nvcsw", "ru_nivcsw")})
        options = {}

        class Params:
            def __init__(self, model):
                pass

            def set_search_options(self, **kwargs):
                options.update(kwargs)

        class Generator:
            def __init__(self, model, params):
                self.tokens = []

            def append_tokens(self, prompt):
                self.prompt = list(prompt)
                clock.now = 0.1

            def is_done(self):
                return False

            def generate_next_token(self):
                clock.now = max(clock.now, 2.0) + 0.1
                self.tokens.append(10 + len(self.tokens))

            def get_next_tokens(self):
                return [self.tokens[-1]]

            def get_sequence(self, index):
                return self.prompt + self.tokens

            def get_output(self, name):
                raise AssertionError("Must not clone full model output to synchronize")

            def get_logits(self):
                raise AssertionError("Must not materialize lazy search scores")

        namespace = dict(time=SimpleNamespace(perf_counter=lambda: clock.now),
                         resource=SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: usage),
                         gc=SimpleNamespace(collect=lambda: None), token_hash=lambda tokens: str(tokens))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<ready-benchmark>", "exec"), namespace)
        result = namespace["run_request"](
            SimpleNamespace(GeneratorParams=Params, Generator=Generator), None,
            SimpleNamespace(decode=lambda tokens: str(tokens)), [1, 2, 3, 4], 3, 16)
        self.assertAlmostEqual(result["prefill_seconds"], 2.1)
        self.assertAlmostEqual(result["ttft_seconds"], 2.1)
        self.assertAlmostEqual(result["prefill_tps"], 4 / 2.1)
        self.assertAlmostEqual(result["decode_tps"], 10)
        self.assertTrue(math.isfinite(result["decode_tps"]))
        self.assertEqual(result["prefill_timing_scope"], "append_tokens_through_first_token_selection")
        self.assertEqual(options["min_length"], options["max_length"])
        self.assertEqual(options["max_length"], 7)
        self.assertEqual(result["output_token_ids"], [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
