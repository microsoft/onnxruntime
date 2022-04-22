# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Test reference implementation and model for ONNX Runtime conrtib op trilu

import unittest

import numpy as np
import onnx
from onnx_contrib_ops_helper import expect


def triu_reference_implementation(x, k=0):
    return np.triu(x, k)


def tril_reference_implementation(x, k=0):
    return np.tril(x, k)


class ONNXReferenceImplementationTest(unittest.TestCase):
    def test_triu(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = triu_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_triu")

    def test_triu_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([-1]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_neg")

    def test_triu_out_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([-7]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_out_neg")

    def test_triu_pos(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([2]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_pos")

    def test_triu_out_pos(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([6]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_out_pos")

    def test_triu_square(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 5, 5).astype(np.float32)
        y = triu_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_triu_square")

    def test_triu_square_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 5, 5).astype(np.float32)
        k = np.array([-1]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_square_neg")

    def test_triu_one_row_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 1, 5).astype(np.float32)
        k = np.array([-7]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_one_row_neg")

    def test_triu_square_pos(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 5, 5).astype(np.float32)
        k = np.array([2]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_square_pos")

    def test_triu_zero(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 0, 5).astype(np.float32)
        k = np.array([6]).astype(np.int64)
        y = triu_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_triu_zero")

    def test_tril(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        y = tril_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_tril")

    def test_tril_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([-1]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_neg")

    def test_tril_out_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([-7]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_out_neg")

    def test_tril_pos(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([2]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_pos")

    def test_tril_out_pos(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        k = np.array([6]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_out_pos")

    def test_tril_square(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 5, 5).astype(np.float32)
        y = tril_reference_implementation(x)
        expect(node, inputs=[x], outputs=[y], name="test_tril_square")

    def test_tril_square_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 5, 5).astype(np.float32)
        k = np.array([-1]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_square_neg")

    def test_tril_one_row_neg(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 1, 5).astype(np.float32)
        k = np.array([-7]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_one_row_neg")

    def test_tril_square_pos(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 5, 5).astype(np.float32)
        k = np.array([2]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_square_pos")

    def test_tril_zero(self):
        node = onnx.helper.make_node(
            "Trilu",
            inputs=["x", "k"],
            outputs=["y"],
            upper=0,
            domain="com.microsoft",
        )

        x = np.random.randn(3, 0, 5).astype(np.float32)
        k = np.array([6]).astype(np.int64)
        y = tril_reference_implementation(x, k)
        expect(node, inputs=[x, k], outputs=[y], name="test_tril_zero")


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
