import unittest
from unittest.mock import MagicMock, patch

import profile_matmul_nbits


class TestProfileMatMulNBits(unittest.TestCase):
    def test_session_disables_fpa_intb_gemm(self):
        options = MagicMock()
        session = MagicMock()
        with (
            patch.object(profile_matmul_nbits.onnxruntime, "SessionOptions", return_value=options),
            patch.object(profile_matmul_nbits.onnxruntime, "InferenceSession", return_value=session) as create_session,
        ):
            result = profile_matmul_nbits.make_session(b"model")

        self.assertIs(result, session)
        options.add_session_config_entry.assert_called_once_with("ep.cuda.fpa_intb_gemm", "0")
        create_session.assert_called_once_with(b"model", options, providers=["CUDAExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
