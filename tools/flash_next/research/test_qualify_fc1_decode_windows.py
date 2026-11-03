import unittest

from qualify_fc1_decode_windows import VARIANTS, verify_profile_proof


class Fc1WindowQualificationTests(unittest.TestCase):
    def test_each_candidate_requires_actual_full_profile_provenance_and_coverage(self):
        for variant, (_, program, _) in VARIANTS.items():
            proof = dict(status="original_sequences_qualified", phase="profile", provider_sha256="provider",
                         context=2048, full_validation=True, zero_major_faults=True,
                         original_output_prefixes_match=True, fixed_grouped_prefill_selector=1,
                         result=dict(candidate_program_dispatches={program: 48}))
            with self.subTest(variant=variant):
                verify_profile_proof(proof, "provider", 2048, True, program)
            for field, value in (("status", "failed"), ("phase", "windows"), ("provider_sha256", "wrong"),
                                 ("context", 8192), ("full_validation", False), ("zero_major_faults", False),
                                 ("original_output_prefixes_match", False), ("fixed_grouped_prefill_selector", 0),
                                 ("result", dict(candidate_program_dispatches={program: 47})),
                                 ("result", dict(candidate_program_dispatches={"wrong_program": 48}))):
                with self.subTest(variant=variant, field=field), self.assertRaises(AssertionError):
                    verify_profile_proof({**proof, field: value}, "provider", 2048, True, program)


if __name__ == "__main__":
    unittest.main()
