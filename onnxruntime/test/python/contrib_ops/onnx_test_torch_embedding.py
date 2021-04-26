# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Test reference implementation and model for ONNX Runtime conrtib op torch_embedding

import onnx
import unittest
import numpy as np
from onnx_contrib_ops_helper import expect


def torch_embedding_reference_implementation(weight, indices, padding_idx=None, scale=False):
    return np.take(weight, indices, axis=0)


class ONNXReferenceImplementationTest(unittest.TestCase):
    def test_torch_embedding(self):
        node = onnx.helper.make_node(
            'TorchEmbedding',
            inputs=['w', 'x'],
            outputs=['y'],
            domain="com.microsoft",
        )

        x = np.random.randn(2, 4).astype(np.int64)
        w = np.random.randn(10, 3).astype(np.float32)
        y = torch_embedding_reference_implementation(w, x)
        expect(node, inputs=[w, x], outputs=[y], name='test_torch_embedding')

    def test_torch_embedding_long(self):
        node = onnx.helper.make_node(
            'TorchEmbedding',
            inputs=['w', 'x'],
            outputs=['y'],
            domain="com.microsoft",
        )

        x = np.random.randn(2, 4).astype(np.int64)
        w = np.random.randn(10, 3).astype(np.int64)
        y = torch_embedding_reference_implementation(w, x)
        expect(node, inputs=[w, x], outputs=[y], name='test_torch_embedding_long')

    def test_torch_embedding_zero_dim(self):
        node = onnx.helper.make_node(
            'TorchEmbedding',
            inputs=['w', 'x'],
            outputs=['y'],
            domain="com.microsoft",
        )

        x = np.random.randn(0, 4).astype(np.int64)
        w = np.random.randn(10, 3).astype(np.float32)
        y = torch_embedding_reference_implementation(w, x)
        expect(node, inputs=[w, x], outputs=[y], name='test_torch_embedding_zero_dim')

    def test_torch_embedding_padding_idx(self):
        node = onnx.helper.make_node(
            'TorchEmbedding',
            inputs=['w', 'x', 'padding_idx'],
            outputs=['y'],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4).astype(np.int64)
        w = np.random.randn(10, 3).astype(np.float32)
        padding_idx = np.random.randint(3, size=1).astype(np.int64)
        y = torch_embedding_reference_implementation(w, x, padding_idx)
        expect(node, inputs=[w, x, padding_idx], outputs=[y], name='test_torch_embedding_padding_idx')

    def test_torch_embedding_scale_grad_by_freq(self):
        node = onnx.helper.make_node(
            'TorchEmbedding',
            inputs=['w', 'x', 'padding_idx', 'scale'],
            outputs=['y'],
            domain="com.microsoft",
        )

        x = np.random.randn(3, 4).astype(np.int64)
        w = np.random.randn(10, 3).astype(np.float32)
        padding_idx = np.random.randint(3, size=1).astype(np.int64)
        scale = np.array([1]).astype(np.bool)
        y = torch_embedding_reference_implementation(w, x, padding_idx, scale)
        expect(node, inputs=[w, x, padding_idx, scale], outputs=[y], name='test_torch_embedding_scale_grad_by_freq')


if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)
