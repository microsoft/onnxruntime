import json
import unittest

from qualify_hc_up_parallel2_standard import verify_standard
from qualify_qsa_score16_standard import collect_standard_cases
from qualify_quickgelu_mul_long_context import ROOT


class QualificationRegressionTests(unittest.TestCase):
    def test_existing_chunk256_standard_rows_reproduce_exactly(self):
        def read(name):
            return json.loads((ROOT / name).read_text())

        native = read("dense-tile4-v1-native-gpu7-20261030.qualification.json")
        paired = read("dense-tile4-v1-windows-contexts-20261030.qualification.json")
        parent = read("hc-up-parallel2-v1-contexts-20261029.qualification.json")
        controls = read("int4-pending64-standard-contexts.json")
        expected = read("dense-tile4-v1-contexts-20261030.qualification.json")

        def verify_case(*args):
            return verify_standard(*args, experiment_flag="ORT_WEBGPU_DENSE_TILE4")

        actual = collect_standard_cases("dense-tile4-v1", native, parent, controls, paired, {},
            verifier=verify_case, stamp="20261030")
        self.assertEqual(actual, expected["rows"])


if __name__ == "__main__":
    unittest.main()
