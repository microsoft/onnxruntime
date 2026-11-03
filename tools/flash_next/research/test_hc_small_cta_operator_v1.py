import unittest

from prepare_webgpu_int4_metrics import ROOT
from run_hc_small_cta_operator_v1 import FLAG, command_for, measured_gpu


class SmallCtaOperatorTests(unittest.TestCase):
    def test_explicit_matched_selectors_and_gpu0_placement(self):
        prefix = ROOT / "unused-small-cta-test"
        for flag in (0, 1):
            command = command_for(prefix, flag, True)
            self.assertEqual(command[:5], ["taskset", "-c", "0-7", "bash", "-lc"])
            self.assertIn(f"{FLAG}={flag}", command[-1])
            self.assertIn("ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0", command[-1])
            self.assertIn("/evidence/webgpu-perf-oct20/unused-small-cta-test-ort-profile", command[-1])
            self.assertIn("postnorm_shared0_rows1_scale2_alpha1.0.onnx", command[-1])
        self.assertTrue(command_for(prefix, 0, False)[-1].endswith(" -"))
        with self.assertRaises(ValueError):
            command_for(prefix, 2, False)

    def test_native_gpu_identity_warmup_and_exact_count(self):
        event = dict(cat="Kernel", pid=-1, tid=-1, ph="X", dur=17,
                     name="node&HCPostNormProjectionSmallCtaF16", args={"cache_key": "program:128,1,1:inputs"})
        self.assertEqual(measured_gpu([event] * 16 + [dict(event, dur=12)] * 128, 1), [12] * 128)
        for events in ([event] * 143, [dict(event, pid=1)] * 144,
                       [dict(event, args={"cache_key": "program:256,1,1:inputs"})] * 144):
            with self.assertRaises(RuntimeError):
                measured_gpu(events, 1)


if __name__ == "__main__":
    unittest.main()
