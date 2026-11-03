from pathlib import Path
import inspect
import unittest

from build_dense_private_partials import FLAG, private_partials_template, transform
from qsa_rank_decode_windows import DENSE_PRIVATE_PARTIALS_FLAG, run_qsa_windows


class DensePrivatePartialsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.component = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1")
        cls.source = (cls.component / "matmul_nbits.cc").read_text()
        cls.template = (cls.component.parent / "block32-template-v1/dense_block32_template.inc").read_text()

    def test_activation_staging_dot_order_barriers_and_serial_fold_unchanged(self):
        result = private_partials_template(self.template)
        original = "\n".join(line for line in self.template.splitlines() if not line.lstrip().startswith("//")) + "\n"
        for token in ("tile_A", "loadSHMA", "dot(", "workgroupBarrier()", "output_value +="):
            self.assertEqual([line for line in original.splitlines() if token in line],
                             [line for line in result.splitlines() if token in line])
        self.assertIn('var private_partial = acc_element_t(0);', result)
        self.assertIn('private_partial += sum;', result)
        self.assertIn('if (kidx + " << __param_tile_size_k << "u >= uniforms.K)', result)
        self.assertIn('inter_results[idy][idx] = private_partial;', result)
        self.assertIn('ss << "    }\\n";\nss << "    if (kidx + "', result)
        self.assertIn('ss << "    }\\n";\nss << "    workgroupBarrier();\\n";', result)

    def test_inactive_final_tile_lanes_publish_earlier_partial_contributions(self):
        for k in (640, 2560, 6144, 10240):
            private = [0] * 32
            shared = [0] * 32
            for kidx in range(0, k, 1024):
                for lane in range(32):
                    if kidx // 32 + lane < k // 32:
                        private[lane] += 1
                if kidx + 1024 >= k:
                    shared[:] = private
            self.assertEqual(shared, [len(range(lane * 32, k, 1024)) for lane in range(32)])

    def test_all_128_threads_publish_only_their_owned_column_and_k_lane(self):
        slots = [(thread // 32, thread % 32) for thread in range(128)]
        self.assertEqual(len(set(slots)), 128)
        self.assertEqual(set(slots), {(col, lane) for col in range(4) for lane in range(32)})
        result = transform(self.source)
        function = result[result.rindex("Status ApplyDensePrivatePartialsMatMulNBits("):]
        self.assertIn("constexpr uint32_t workgroup_size = 128;", function)
        self.assertIn("constexpr uint32_t tile_size = 4;", function)

    def test_canonical_fp16_gate_parent_preservation_and_replay_identity(self):
        result = transform(self.source)
        begin = self.source.index("class DenseBlock32Program")
        end = self.source.index("\n}\n\nStatus MatMulNBitsProgram::", begin)
        self.assertIn(self.source[begin:end], result)
        self.assertIn(f'"{FLAG}", 0', result)
        self.assertIn(f'"{FLAG} must be 0 or 1"', result)
        self.assertIn("private_partials_mode == 0 || private_partials_mode == 1", result)
        self.assertIn("private_partials_mode == 1 && a->IsDataType<MLFloat16>() && !context.EnableMatmulFp32Accumulation()", result)
        self.assertIn("private_partials_route ? 6 : block32_mode == 1 ? 3", result)
        self.assertIn('Program{"MatMulNBitsPrivatePartialsF16"}', result)
        self.assertNotIn("DirectA", result)

    def test_window_cli_flag_and_result_prefix_are_wired(self):
        self.assertEqual(DENSE_PRIVATE_PARTIALS_FLAG, FLAG)
        self.assertIn('DENSE_PRIVATE_PARTIALS_FLAG: "dense_private_partials"', inspect.getsource(run_qsa_windows))
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn('add_argument("--compare-dense-private-partials-windows", action="store_true")', benchmark)
        self.assertIn("args.compare_dense_private_partials_windows or", benchmark)
        self.assertIn('"dense_private_partials_decode_windows" if args.compare_dense_private_partials_windows else', benchmark)
        self.assertIn("DENSE_PRIVATE_PARTIALS_FLAG if args.compare_dense_private_partials_windows else", benchmark)
        self.assertIn(f'"{FLAG}",', benchmark)


if __name__ == "__main__":
    unittest.main()
