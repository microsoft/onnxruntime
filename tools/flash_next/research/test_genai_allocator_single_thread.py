from pathlib import Path
import unittest

from build_genai_allocator_single_thread import ADDITION, ANCHOR, FLAG, transform_source


MODEL = Path("/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/src/models/model.cpp")


class AllocatorSingleThreadTests(unittest.TestCase):
    def test_exact_reversible_source_delta(self):
        original = MODEL.read_text()
        modified = transform_source(original)
        restored = modified.replace('#include "env_utils.h"\n', "").replace(ADDITION, "")
        self.assertEqual(restored, original)
        self.assertEqual(modified.count(FLAG), 1)

    def test_only_dummy_webgpu_session_changes(self):
        modified = transform_source(MODEL.read_text())
        init = modified.split("void EnsureDeviceOrtInit(", 1)[1].split(
            "void ", 1)[0]
        self.assertIn("if (type == DeviceType::WEBGPU)", init)
        self.assertIn("bool allocator_single_thread = false;", init)
        self.assertIn(f'GetEnv("{FLAG}", allocator_single_thread);', init)
        self.assertIn("if (allocator_single_thread) session_options->SetIntraOpNumThreads(1);", init)
        self.assertLess(init.index(ADDITION), init.index("SetProviderSessionOptions("))
        self.assertLess(init.index(ADDITION), init.index("OrtSession::Create("))
        self.assertNotIn(FLAG, modified.split("DeviceInterface* Model::CreateSessionOptionsFromConfig", 1)[1])

    def test_duplicate_or_missing_anchor_rejected(self):
        original = MODEL.read_text()
        with self.assertRaises(AssertionError):
            transform_source(original + "\n" + ANCHOR)
        with self.assertRaises(AssertionError):
            transform_source(original.replace(ANCHOR, ""))
        with self.assertRaises(AssertionError):
            transform_source(transform_source(original))


if __name__ == "__main__":
    unittest.main()
