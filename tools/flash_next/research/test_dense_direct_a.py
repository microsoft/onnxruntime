from pathlib import Path
import inspect
import re
import unittest

from build_dense_direct_a import FLAG, direct_a_template, transform
from build_dense_double_buffer import transform_compare
from qsa_rank_decode_windows import DENSE_DIRECT_A_FLAG, DIRECT_A_FLAG, run_qsa_windows


class DenseDirectATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/"
                            "block32-template-v1/dense_block32_template.inc").read_text()
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/matmul_nbits.cc").read_text()

    def test_no_activation_shared_array_and_one_final_barrier(self):
        result = direct_a_template(self.template)
        actual = "\n".join(line for line in result.splitlines() if not line.lstrip().startswith("//"))
        self.assertNotIn("tile_A", actual)
        self.assertNotIn("loadSHMA", actual)
        self.assertIn('__var_a->GetByOffset("kidx / 4u + index")', actual)
        self.assertEqual(actual.count('workgroupBarrier();\\n'), 1)
        self.assertIn('ss << "  }\\n";\nss << "  workgroupBarrier();\\n";', actual)

    def test_dot_expressions_shared_rounding_and_serial_fold_unchanged(self):
        result = direct_a_template(self.template)
        restored = re.sub(r"read_dense_a\((a_offset(?: \+ \d+)?), kidx\)", r"tile_A[\1]", result)
        for token in ("dot(", "inter_results[", "output_value +="):
            before = [line for line in self.template.splitlines() if line.startswith("ss <<") and token in line]
            after = [line for line in restored.splitlines() if line.startswith("ss <<") and token in line]
            self.assertEqual(before, after)

    def test_canonical_gate_invalid_selector_and_distinct_replay_variant(self):
        result = transform(self.source)
        self.assertIn(f'ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0)', result)
        self.assertIn("direct_a_mode == 0 || direct_a_mode == 1", result)
        self.assertIn(f"{FLAG} must be 0 or 1", result)
        self.assertIn("direct_a_mode == 1 && a->IsDataType<MLFloat16>() && !context.EnableMatmulFp32Accumulation()", result)
        self.assertIn("direct_a_route ? 5 : block32_mode == 1 ? 3", result)
        self.assertIn("class DenseBlock32Program", result)
        self.assertIn("class DenseDirectAProgram", result)
        self.assertIn('Program{"MatMulNBitsDirectAF16"}', result)
        begin = self.source.index("class DenseBlock32Program")
        end = self.source.index("\n}\n\nStatus MatMulNBitsProgram::", begin)
        self.assertIn(self.source[begin:end], result)

    def test_window_flag_is_separate_from_prior_direct_a_experiment(self):
        self.assertEqual(DENSE_DIRECT_A_FLAG, FLAG)
        self.assertNotEqual(DIRECT_A_FLAG, FLAG)
        self.assertIn('DENSE_DIRECT_A_FLAG: "dense_direct_a"', inspect.getsource(run_qsa_windows))

    def test_shared_clone_refactor_preserves_built_direct_a_source(self):
        built = Path("/home/sayanshaw/repos/qwen38-int4-dense-direct-a/build-v1/matmul_nbits.cc").read_text()
        self.assertEqual(transform(self.source), built)

    def test_native_fp32_and_window_cli_wiring(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/native_compare.cpp").read_text()
        comparator = transform_compare(source.replace("ORT_WEBGPU_DENSE_BLOCK32", FLAG), FLAG)
        self.assertIn('{"enableMatmulFp32Accumulation", fp32}', comparator)
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn('add_argument("--compare-dense-direct-a-windows", action="store_true")', benchmark)
        self.assertIn("args.compare_dense_direct_a_windows or", benchmark)
        self.assertIn('"dense_direct_a_decode_windows" if args.compare_dense_direct_a_windows else', benchmark)
        self.assertIn("DENSE_DIRECT_A_FLAG if args.compare_dense_direct_a_windows else", benchmark)
        self.assertIn(f'"{FLAG}",', benchmark)

    def test_direct_vector_reads_are_in_bounds_for_every_canonical_tail(self):
        for k in (640, 2560, 6144, 10240):
            for kidx in range(0, k, 1024):
                for lane in range(32):
                    if kidx // 32 + lane < k // 32:
                        for component in range(8):
                            self.assertLess(kidx // 4 + lane * 8 + component, k // 4)


if __name__ == "__main__":
    unittest.main()
