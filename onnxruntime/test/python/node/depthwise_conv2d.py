# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
import tensorflow as tf
import tf2onnx
from onnx.backend.test.case.base import Base
from . import expect
from onnx.backend.sample.ops.abs import abs

x_in = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
], dtype=np.float32).reshape((1, 3, 2, 1))

kernel_in = np.array([
    [1., 2.],
    [3., 4]
], dtype=np.float32).reshape((2, 1, 1, 2))


# Create a model using low-level tf.* APIs
class SimpleDepthwiseConv2DValidPadding(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=x_in.shape, dtype=tf.float32)])
  def __call__(self, x):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID');

class SimpleDepthwiseConv2DSamePadding(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=x_in.shape, dtype=tf.float32)])
  def __call__(self, x):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME');

class Abs(Base):

    @staticmethod
    def export() -> None:
        onnx_opset_number = 13
        # The following example is from https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        simple_conv2D_model = SimpleDepthwiseConv2DValidPadding()
        output_tensor = simple_conv2D_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(simple_conv2D_model.__call__,input_signature, opset=onnx_opset_number)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleDepthwiseConv2DValidPadding')

        simple_conv2D_model = SimpleDepthwiseConv2DSamePadding()
        output_tensor = simple_conv2D_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(simple_conv2D_model.__call__,input_signature, opset=onnx_opset_number)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleDepthwiseConv2DSamePadding')
