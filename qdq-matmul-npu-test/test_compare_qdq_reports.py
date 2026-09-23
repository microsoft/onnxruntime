"""Regression tests for paired CPU/EP QDQ-ablation report comparisons."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
from onnx import numpy_helper

from compare_qdq_reports import compare_reports, read_report
from drift_investigation.qdq_ablation import activation_pairs, bypass_pairs, write_variant
from drift_investigation.test_qdq_ablation import make_model
from run_acc import file_sha256, model_external_data_sha256


class CompareQdqReportsTest(unittest.TestCase):
    def test_allows_removed_qdq_only_external_files_but_not_changed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, variant = root / "source.onnx", root / "variant.onnx"
            model = make_model()
            for node in model.graph.node[:2]:
                node.input[1], node.input[2] = "only_scale", "only_zp"
            model.graph.initializer.extend([
                numpy_helper.from_array(np.array(0.5, dtype=np.float32), "only_scale"),
                numpy_helper.from_array(np.array(128, dtype=np.uint8), "only_zp"),
            ])
            onnx.save_model(
                model, original, save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=0,
            )
            graph = ir.load(original)
            bypass_pairs(graph, [pair for pair in activation_pairs(graph.graph) if pair.name == "group_a_Q"])
            write_variant(graph, original, variant, 1)
            before = model_external_data_sha256(original)
            after = model_external_data_sha256(variant)
            self.assertEqual(set(before) - set(after), {"only_scale", "only_zp"})
            inputs_path, baseline_arrays, variant_arrays = (root / name for name in (
                "inputs.npz", "baseline.npz", "variant_outputs.npz",
            ))
            np.savez(inputs_path, x=np.zeros((1, 4), dtype=np.float32))
            np.savez(baseline_arrays, cpu_0=np.zeros((1, 4)), ep_0=np.zeros((1, 4)))
            np.savez(variant_arrays, cpu_0=np.ones((1, 4)), ep_0=np.zeros((1, 4)))
            report = {
                "model": str(original), "model_sha256": file_sha256(original),
                "provider": "QNNExecutionProvider", "provider_options": {},
                "inputs": str(inputs_path), "inputs_sha256": file_sha256(inputs_path),
                "valid_mask": None, "output_names": ["y"], "external_data_sha256": before,
                "ablation_pairs_removed": 0, "output_arrays": str(baseline_arrays),
            }
            ablated = dict(
                report, model=str(variant), model_sha256=file_sha256(variant),
                external_data_sha256=after, ablation_pairs_removed=1, output_arrays=str(variant_arrays),
            )
            self.assertEqual(compare_reports(report, ablated)["outputs"]["y"]["cpu_shift_mae"], 1.0)
            with self.assertRaisesRegex(ValueError, "shared or new files"):
                compare_reports(report, dict(ablated, external_data_sha256=dict(after, weight="different")))
            with self.assertRaisesRegex(ValueError, "retained or non-QDQ"):
                compare_reports(report, dict(
                    ablated, external_data_sha256={key: value for key, value in after.items() if key != "weight"},
                ))
            with self.assertRaisesRegex(ValueError, "device"):
                compare_reports(dict(report, device={"device_id": 1}), dict(ablated, device={"device_id": 2}))

    def test_detects_cpu_only_qdq_change_and_rejects_different_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs_path = root / "inputs.npz"
            np.savez(inputs_path, x=np.asarray([0], dtype=np.float32),
                     valid_mask=np.asarray([[True, False]]))
            baseline_arrays, variant_arrays = root / "baseline.npz", root / "variant.npz"
            np.savez(baseline_arrays, cpu_0=np.asarray([[[0.], [5.]]]), ep_0=np.asarray([[[0.], [6.]]]))
            np.savez(variant_arrays, cpu_0=np.asarray([[[1.], [5.]]]), ep_0=np.asarray([[[0.], [6.]]]))
            report = {
                "provider": "QNNExecutionProvider",
                "provider_options": {},
                "cpu_fallback": False,
                "ep_context_nodes": 1,
                "inputs": str(inputs_path),
                "inputs_sha256": file_sha256(inputs_path),
                "valid_mask": "valid_mask",
                "output_names": ["vision_features"],
                "external_data_sha256": {"weights.onnx.data": "identical"},
                "ablation_pairs_removed": 0,
                "output_arrays": str(baseline_arrays),
            }
            variant = dict(report, ablation_pairs_removed=1, output_arrays=str(variant_arrays))
            result = compare_reports(report, variant)
            output = result["outputs"]["vision_features"]
            self.assertEqual(output["cpu_shift_mae"], 1.0)
            self.assertEqual(output["ep_shift_mae"], 0.0)
            self.assertEqual(output["variant_ep_cpu_mae"], 1.0)
            with self.assertRaisesRegex(ValueError, "CPU graph optimization"):
                compare_reports(dict(report, cpu_optimization="basic"), variant)
            variant["inputs_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "inputs_sha256"):
                compare_reports(report, variant)

            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(read_report(report_path)["provider"], "QNNExecutionProvider")
            report["cpu_fallback"] = True
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disabled CPU fallback"):
                read_report(report_path)


if __name__ == "__main__":
    unittest.main()
