# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Test reference implementation and model for ONNX Runtime conrtib op trilu

import unittest

import numpy as np
import onnx
from onnx_contrib_ops_helper import expect


class ONNXReferenceImplementationTest(unittest.TestCase):
    def test_col2im(self) -> None:
        input = np.array(
            [
                [
                    [1.0, 6.0, 11.0, 16.0, 21.0],  # (1, 5, 5)
                    [2.0, 7.0, 12.0, 17.0, 22.0],
                    [3.0, 8.0, 13.0, 18.0, 23.0],
                    [4.0, 9.0, 14.0, 19.0, 24.0],
                    [5.0, 0.0, 15.0, 20.0, 25.0],
                ]
            ]
        ).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)
        node = onnx.helper.make_node(
            "Col2Im", ["input", "image_shape", "block_shape"], ["col2im_reference_implementation"]
        )

        col2im_reference_implementation = np.array(
            [
                [
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0],  # (1, 1, 5, 5)
                        [6.0, 7.0, 8.0, 9.0, 0.0],
                        [11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0, 25.0],
                    ]
                ]
            ]
        ).astype(np.float32)

        expect(
            node,
            inputs=[input, image_shape, block_shape],
            outputs=[col2im_reference_implementation],
            name="test_col2im",
        )


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
