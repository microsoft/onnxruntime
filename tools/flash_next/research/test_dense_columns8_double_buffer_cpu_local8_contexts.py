import copy
import json
import unittest

from qualify_dense_columns8_double_buffer_cpu_local8_contexts import (
    BUILD, ROOT, RUNTIME, STAMP, VARIANT, candidates_above_target, verify_screen,
)


class DoubleBufferContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = json.loads((BUILD / "build-manifest.json").read_text())["provider_sha256"]
        cls.runtime = json.loads((RUNTIME / "build-manifest.json").read_text())["runtime_sha256"]
        cls.proofs = {(context, repeat): json.loads(
            (ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}screen-{STAMP}.qualification.json").read_text())
            for context in (2048, 8192) for repeat in (False, True)}

    def test_completed_initial_and_repeat_contexts(self):
        for (context, repeat), proof in self.proofs.items():
            self.assertTrue(verify_screen(proof, context, repeat, self.provider, self.runtime))

    def test_unknown_status_mixed_providers_and_missing_mode_rejected(self):
        original = self.proofs[8192, True]
        for field, value in (("status", "success"), ("provider_sha256", "wrong"),
                             ("parent_runtime_sha256", "wrong"), ("rows", original["rows"][:1])):
            proof = copy.deepcopy(original)
            proof[field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify_screen(proof, 8192, True, self.provider, self.runtime)

    def test_nonpositive_request_and_nonfinite_ttft_rejected(self):
        for field, value in (("request_ratios", [1.01, 1.01, 1.0]),
                             ("matched_request_ttft_ratio", float("nan"))):
            proof = copy.deepcopy(self.proofs[8192, True])
            proof["rows"][0]["comparison"][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify_screen(proof, 8192, True, self.provider, self.runtime)

    def test_every_fresh_context_and_mode_must_exceed_100(self):
        rows = [dict(context=context, initial=copy.deepcopy(self.proofs[8192, False]),
                     fresh=copy.deepcopy(self.proofs[8192, True]))
                for context in (2048, 8192, 32768)]
        self.assertFalse(candidates_above_target(rows))
        for row in rows:
            for mode in row["fresh"]["rows"]:
                mode["candidate"]["median_decode_tps"] = 100.000001
        self.assertTrue(candidates_above_target(rows))
        rows[-1]["fresh"]["rows"][-1]["candidate"]["median_decode_tps"] = 100
        self.assertFalse(candidates_above_target(rows))


if __name__ == "__main__":
    unittest.main()
