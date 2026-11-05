# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model


class TestUnsupportedQuantization(unittest.TestCase):
    def test_retired_entry_points(self):
        for entry_point in (qnn_preprocess_model, get_qnn_qdq_config):
            with (
                self.subTest(entry_point=entry_point.__name__),
                self.assertRaisesRegex(NotImplementedError, "QNN .* is no longer supported"),
            ):
                entry_point("missing_model.onnx", None)


if __name__ == "__main__":
    unittest.main()
