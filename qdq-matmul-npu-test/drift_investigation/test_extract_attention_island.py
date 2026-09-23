"""Test attention-island extraction and exact CPU input replay."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort

from drift_investigation.extract_attention_island import (
    capture_inputs, main, make_island_model, make_tapped_model, select_island,
)
from drift_investigation.test_qdq_ablation import make_attention_model


class ExtractAttentionIslandTest(unittest.TestCase):
    def test_island_in_another_directory_has_its_own_external_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "source.onnx"
            island_path = root / "repro" / "island.onnx"
            inputs_path = root / "inputs.npz"
            baseline_path = root / "baseline.npz"
            inputs = {
                "query": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
                "keys": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
                "values": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
            }
            onnx.save_model(
                make_attention_model(), source_path, save_as_external_data=True,
                all_tensors_to_one_file=True, location="source.onnx.data", size_threshold=0,
            )
            np.savez(inputs_path, **inputs)
            baseline = ort.InferenceSession(str(source_path)).run(None, inputs)[0]
            np.savez(baseline_path, cpu_0=baseline)
            with patch("sys.argv", [
                "extract_attention_island", str(source_path),
                "--softmax", "attention_softmax",
                "--tap-output", str(root / "tapped.onnx"),
                "--island-output", str(island_path),
                "--inputs", str(inputs_path),
                "--baseline-arrays", str(baseline_path),
                "--island-inputs", str(root / "island_inputs.npz"),
            ]):
                main()
            self.assertTrue((root / "repro" / "island.onnx.data").is_file())
            locations = {
                entry.value for tensor in onnx.load(island_path, load_external_data=False).graph.initializer
                for entry in tensor.external_data if entry.key == "location"
            }
            self.assertLessEqual(locations, {"island.onnx.data"})
            self.assertTrue(onnx.load(island_path).graph.node)

    def test_island_replays_real_upstream_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "source.onnx"
            tapped_path = root / "tapped.onnx"
            island_path = root / "island.onnx"
            inputs_path = root / "inputs.npz"
            baseline_path = root / "baseline.npz"
            island_inputs_path = root / "island_inputs.npz"
            source = ir.serde.deserialize_model(make_attention_model())
            source.graph.sort()
            onnx.save(ir.serde.serialize_model(source), source_path)
            inputs = {
                "query": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
                "keys": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
                "values": np.asarray([[[1, 0], [0, 1]]], dtype=np.float32),
            }
            np.savez(inputs_path, **inputs)
            baseline = ort.InferenceSession(str(source_path)).run(None, inputs)[0]
            np.savez(baseline_path, cpu_0=baseline)

            boundary = select_island(source, "attention_softmax")
            onnx.save(ir.serde.serialize_model(make_tapped_model(source, boundary)), tapped_path)
            onnx.save(ir.serde.serialize_model(make_island_model(source, boundary)), island_path)
            capture_inputs(
                tapped_path, island_path, inputs_path, baseline_path, island_inputs_path, boundary, None
            )
            with np.load(island_inputs_path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"query", "key", "value", "attention_mask"})
                result = ort.InferenceSession(str(island_path)).run(
                    None, {name: archive[name] for name in archive.files}
                )[0]
            np.testing.assert_array_equal(result, baseline)


if __name__ == "__main__":
    unittest.main()
