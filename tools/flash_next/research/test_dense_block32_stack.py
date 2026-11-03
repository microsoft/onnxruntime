import copy
import json
from pathlib import Path
import unittest

from build_dense_block32 import FLAG
from build_dense_block32_stack import COMPONENT, ORIGINAL_OBJECT, combined_link
from qualify_dense_block32 import verify_flags


BASE = Path("/home/sayanshaw/repos")


class DenseBlock32StackTests(unittest.TestCase):
    def setUp(self):
        self.parent = json.loads((BASE / "qwen38-int4-dsa-query-cache/build-v2/build-manifest.json").read_text())
        self.component = json.loads((BASE / "qwen38-int4-dense-block32/build-v1/build-manifest.json").read_text())

    def test_only_dense_object_and_output_change(self):
        original = copy.deepcopy(self.parent)
        output = Path("/workspace/qwen38-int4-dense-block32/build-v2/test.so")
        command = combined_link(self.parent, self.component, output)
        expected = [str(COMPONENT / "matmul_nbits.cc.o") if arg == ORIGINAL_OBJECT else arg
                    for arg in self.parent["link_command"]]
        expected[expected.index("-o") + 1] = str(output)
        self.assertEqual(command, expected)
        self.assertEqual(self.parent, original)
        query_objects = [arg for arg in command if arg.endswith("dynamic_sparse_attention.cc.o")]
        self.assertEqual(len(query_objects), 1)
        self.assertIn("qwen38-int4-dsa-query-cache/build-v2", query_objects[0])

    def test_ambiguous_or_missing_objects_are_rejected(self):
        for count in (0, 2):
            parent = copy.deepcopy(self.parent)
            parent["link_command"] = [arg for arg in parent["link_command"] if arg != ORIGINAL_OBJECT]
            parent["link_command"].extend([ORIGINAL_OBJECT] * count)
            with self.assertRaises(AssertionError):
                combined_link(parent, self.component, Path("/unused"))
        component = copy.deepcopy(self.component)
        component["link_command"].remove(str(COMPONENT / "matmul_nbits.cc.o"))
        with self.assertRaises(AssertionError):
            combined_link(self.parent, component, Path("/unused"))

    def test_stacked_metadata_requires_both_optimizations(self):
        flags = {flag: "1" for flag in (
            FLAG, "ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DENSE_TILE4",
            "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16", "ORT_WEBGPU_DSA_QUERY_CACHE")}
        verify_flags({"load_experiment_flags": flags}, True)
        for flag in flags:
            changed = dict(flags, **{flag: "0"})
            with self.assertRaises(AssertionError):
                verify_flags({"load_experiment_flags": changed}, True)
        flags["ORT_WEBGPU_DSA_QUERY_CACHE"] = "0"
        verify_flags({"load_experiment_flags": flags}, False)


if __name__ == "__main__":
    unittest.main()
