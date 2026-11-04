# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run inference through the natively compiled Python extension."""

import sys

import numpy as np
import onnxruntime_pybind11_state as ort


def main():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(options, sys.argv[1], True, False)
    session.initialize_session(["CPUExecutionProvider"], [{}], set())
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
    output = session.run(["C"], {"A": a, "B": b}, None)[0]
    np.testing.assert_array_equal(output, a + b)
    assert output.dtype == np.float32
    print("Python/NumPy Add inference passed.")


if __name__ == "__main__":
    main()
