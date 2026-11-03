import unittest

from run_early_scratch_native_v1 import BUILD, FLAG, ROOT, command_for, validate_routing


class EarlyScratchLaunchTests(unittest.TestCase):
    def test_isolated_candidate_original_controls_and_deployed_gate(self):
        prefix = ROOT / "early-scratch-test-gpu0"
        for candidate in (False, True):
            command = command_for(prefix, int(candidate), 0, candidate, True)
            self.assertEqual(command[:5], ["taskset", "-c", "0-7", "bash", "-lc"])
            shell = command[-1]
            self.assertIn(f"{FLAG}={int(candidate)}", shell)
            self.assertIn("ORT_WEBGPU_QMOE_SUBGROUP_GATE=1", shell)
            self.assertIn("ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1", shell)
            self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=0", shell)
            self.assertIn("ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0", shell)
            self.assertNotIn("ORT_WEBGPU_QMOE_LAZY_UNUSED_SCRATCH", shell)
            self.assertNotIn("native-flag0-gpu0-shaders.txt", shell)
            self.assertIn("early-scratch-test-gpu0.json", shell)
            self.assertIn("qmoe_batch32.onnx", shell)
            provider = "/workspace/" + str(BUILD.relative_to("/home/sayanshaw/repos")) + "/libonnxruntime_providers_webgpu.so"
            self.assertEqual(provider in shell, candidate)
        profiled = command_for(prefix, 1, 0, True, True, True)
        self.assertTrue(profiled[-1].endswith("early-scratch-test-gpu0-ort-profile"))
        self.assertIn(f"{FLAG}=2", command_for(prefix, 2, 0, True)[-1])
        self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", command_for(prefix, 1, 1, True)[-1])

    def test_actual_gpu_events_are_required_for_all_three_deployed_programs(self):
        names = ("QmoeHierarchicalGate1Token", "QmoeFC1SwiGLUBitcastF16", "QmoeFC2FinalMixBitcastF16")
        threads = (512, 128, 320)
        events = [dict(cat="Kernel", pid=-1, tid=-1, ph="X", dur=1, name="node&" + name,
                       args={"cache_key": f"test:{size},1,1:"})
                  for name, size in zip(names, threads) for _ in range(144 * 32)]
        events.extend(dict(cat="Kernel", pid=-1, tid=-1, ph="X", dur=1, name="node&Concat")
                      for _ in range(144))
        self.assertEqual(validate_routing(events), dict(dict.fromkeys(names, 144 * 32), Concat=144))
        with self.assertRaises(RuntimeError):
            validate_routing(events[1:])
        events[0]["name"] = "node&QmoeGate1Token"
        with self.assertRaises(RuntimeError):
            validate_routing(events)
        events[0]["name"] = "node&" + names[0]
        events[0]["dur"] = 0
        with self.assertRaises(RuntimeError):
            validate_routing(events)

    def test_batch_preserves_weights_and_fingerprints_every_finite_output(self):
        import onnx
        from prepare_early_scratch_batch import SOURCE, batch_benchmark, batch_model
        original = onnx.load(SOURCE, load_external_data=False)
        model = batch_model(original)
        self.assertEqual(len(model.graph.node), 33)
        self.assertEqual(sum(node.op_type == "QMoE" for node in model.graph.node), 32)
        self.assertEqual(model.graph.node[-1].op_type, "Concat")
        self.assertEqual(list(model.graph.initializer), list(original.graph.initializer))
        for node in model.graph.node[:-1]:
            self.assertEqual(list(node.input), list(original.graph.node[0].input))
        self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[0].dim_value, 32)
        from pathlib import Path
        source = Path(__file__).with_name("fc2_private_a_benchmark.cpp")
        changed = batch_benchmark(source.read_text())
        self.assertIn("i < 163840;", changed)
        self.assertIn("std::isfinite(values[i].ToFloat())", changed)
        self.assertIn("std::memcmp(values, values + node * 2560, 5120)", changed)


if __name__ == "__main__":
    unittest.main()
