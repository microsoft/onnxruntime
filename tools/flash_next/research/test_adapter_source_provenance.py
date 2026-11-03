import hashlib
import json
from pathlib import Path
import unittest

from build_adapter_tensor_reuse_sdk import command_for, recover_original_n96


REPO = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf")


class AdapterSourceProvenanceTests(unittest.TestCase):
    def test_recovery_matches_exact_original_n96_source_hash(self):
        source = REPO / "onnxruntime/core/providers/webgpu/math/matmul.cc"
        manifest = json.loads((REPO / "build-int4-matmul-gemv-n96/build-manifest.json").read_text())
        original = recover_original_n96(source.read_text())
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), manifest["source_sha256"])
        self.assertIn("ORT_WEBGPU_MATMUL_GEMV_N96", original)
        self.assertNotIn("ORT_WEBGPU_MATMUL_F32_DECODE_GEMV", original)

    def test_sdk_preinclude_and_dependency_paths_are_scoped(self):
        command = ["c++", "-include", "/old/pch", "-MF", "/old/dependency",
                   "-MT", "/old/target", "-c", "/original/source.cc", "-o", "/old/object"]
        actual = command_for(command, Path("/new/object.o"), Path("/new/sdk"))
        self.assertEqual(actual[actual.index("-include") + 1], "/new/sdk/ep/adapters.h")
        self.assertEqual(actual[actual.index("-c") + 1], "/original/source.cc")
        self.assertEqual(actual[actual.index("-o") + 1], "/new/object.o")
        self.assertEqual(actual[actual.index("-MF") + 1], "/new/object.o.d")
        self.assertEqual(actual[actual.index("-MT") + 1], "/new/object.o")
        self.assertEqual(command[command.index("-MF") + 1], "/old/dependency")

    def test_original_sdk_diagnostic_path_is_selected_per_source(self):
        main = ["c++", "-c", "/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/test.cc", "-o", "/old"]
        fp8 = ["c++", "-c", "/workspace/onnxruntime-webgpu-fp8-checkpoint/onnxruntime/test.cc", "-o", "/old"]
        for command, expected in (
            (main, "/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/"),
            (fp8, "/workspace/onnxruntime-webgpu-fp8-checkpoint/include/onnxruntime/")):
            actual = command_for(command, Path("/new/object"), Path("/new/sdk"))
            self.assertIn("-fmacro-prefix-map=/new/sdk/=" + expected, actual)


if __name__ == "__main__":
    unittest.main()
