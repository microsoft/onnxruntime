import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace

from build_dense_paired_matrix_reset import CLASS, FLAG, PROGRAM, transform
from prepare_dense_paired_matrix_reset_proofs import make_runner
from test_hc_projection_inner_unroll import render


class ExplicitResetTests(unittest.TestCase):
    def test_tile_sum_initialization_inside_actual_loop(self):
        code = render(PROGRAM, 0, False, False, "", FLAG, class_name=CLASS, constructor_args=())
        start = code.index("for (var tile =")
        reset = code.index("sums[j] = f16(0)")
        self.assertGreater(reset, start)
        self.assertLess(reset, code.index("for (var word ="))
        self.assertIn("totals[j] = f16(0)", code[:start])
        self.assertIn("sums[j] += dots[j] + dot_value", code)
        self.assertIn("value += partials[local_idx * 32u + stream]", code)

    def test_new_flag_and_provider_keep_original_fallback(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text()
        result = transform(source)
        self.assertIn(FLAG + " must be 0 or 1", result)
        self.assertNotIn('"ORT_WEBGPU_DENSE_PAIRED_MATRIX"', result)
        self.assertTrue(result.endswith(source[source.index("\nStatus ApplyMatMulNBits("):]))

    def test_guard_rejects_timeout_and_accepts_proved_numerical_exit(self):
        namespace = {"__name__": "guard-test", "__file__": __file__}
        exec(compile(make_runner(), "runner", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "attempt"
            def child(status, code):
                def run(*args, **kwargs):
                    prefix.with_suffix(".guard.json").write_text(json.dumps(
                        dict(status=status, exit_code=code, gpu_uuid=namespace["UUID"])))
                    prefix.with_suffix(".log").write_text("DENSE comparisons=64 mismatches=64 K=640 N=1")
                    return SimpleNamespace(returncode=code)
                return run
            with patch("subprocess.run", child("failed", 2)):
                self.assertEqual(namespace["run_guard"]("true", prefix)[0], 2)
            prefix.with_suffix(".guard.json").unlink()
            prefix.with_suffix(".log").unlink()
            with patch("subprocess.run", child("run_timed_out", 124)):
                with self.assertRaisesRegex(RuntimeError, "did not prove completed child"):
                    namespace["run_guard"]("true", prefix)


if __name__ == "__main__":
    unittest.main()
