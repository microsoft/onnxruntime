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

x_in = np.array([[1., 2., 3., 4.],
                 [5., 6., 7., 8.],
                 [9., 10., 11., 12.]]).astype(np.float32)
x_in = np.expand_dims(np.expand_dims(x_in, axis=0), axis=3)

# Create a model using low-level tf.* APIs
class SimpleMaxPool2dValidPadding(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=x_in.shape, dtype=tf.float32)])
  def __call__(self, x):
    return tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="VALID");

class SimpleMaxPool2dSamePadding(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=x_in.shape, dtype=tf.float32)])
  def __call__(self, x):
    return tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2), padding="SAME");
    
class MaxPool2DTest(Base):

    @staticmethod
    def export() -> None:
        onnx_opset_number = 13
        # The following example is from https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d
        
        tf_model = SimpleMaxPool2dValidPadding()
        output_tensor = tf_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(tf_model.__call__,input_signature, opset=onnx_opset_number)
        onnx_model=onnx.shape_inference.infer_shapes(onnx_model)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleMaxPool2dValidPadding')

        tf_model = SimpleMaxPool2dSamePadding()
        output_tensor = tf_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(tf_model.__call__,input_signature, opset=onnx_opset_number)
        onnx_model=onnx.shape_inference.infer_shapes(onnx_model)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleMaxPool2dSamePadding')