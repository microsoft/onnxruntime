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

kernel_in = np.array([
    [ [[2, 0.1]], [[3, 0.2]] ],
    [ [[0, 0.3]],[[1, 0.4]] ], ])

x_in = np.array([[
          [[2], [1], [2], [0], [1]],
          [[1], [3], [2], [2], [3]],
          [[1], [1], [3], [3], [0]],
          [[2], [2], [0], [1], [1]],
          [[0], [0], [3], [1], [2]], ]]).astype(np.float32)

# Create a model using low-level tf.* APIs
class SimpleConv2DValidPadding(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=x_in.shape, dtype=tf.float32)])
  def __call__(self, x):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID');

class SimpleConv2DValidPaddingWithSigmoid(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=x_in.shape, dtype=tf.float32)])
  def __call__(self, x):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return tf.math.sigmoid(tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID'));
    
class SimpleConv2DSamePadding(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=x_in.shape, dtype=tf.float32)])
  def __call__(self, x):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME');
    

class SimpleConv2DSamePaddingUnknownBatch(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None,5,5,1], dtype=tf.float32)])
  def __call__(self, x):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME');

class SimpleConv2DSamePaddingUnknownHW(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[1,None,None,1], dtype=tf.float32)])
  def __call__(self, x):
    kernel = tf.constant(kernel_in, dtype=tf.float32)
    return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME');

class Abs(Base):

    @staticmethod
    def export() -> None:
        onnx_opset_number = 13
        # The following example is from https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        simple_conv2D_model = SimpleConv2DValidPadding()
        output_tensor = simple_conv2D_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(simple_conv2D_model.__call__,input_signature, opset=onnx_opset_number)
        onnx_model=onnx.shape_inference.infer_shapes(onnx_model)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleConv2DValidPadding')

        simple_conv2D_model = SimpleConv2DValidPaddingWithSigmoid()
        output_tensor = simple_conv2D_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(simple_conv2D_model.__call__,input_signature, opset=onnx_opset_number)
        onnx_model=onnx.shape_inference.infer_shapes(onnx_model)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleConv2DValidPaddingWithSigmoid')

        simple_conv2D_model = SimpleConv2DSamePadding()
        output_tensor = simple_conv2D_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(simple_conv2D_model.__call__,input_signature, opset=onnx_opset_number)
        onnx_model=onnx.shape_inference.infer_shapes(onnx_model)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleConv2DSamePadding')
               
        simple_conv2D_model = SimpleConv2DSamePaddingUnknownBatch()
        output_tensor = simple_conv2D_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(simple_conv2D_model.__call__,input_signature, opset=onnx_opset_number)
        onnx_model=onnx.shape_inference.infer_shapes(onnx_model)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleConv2DSamePaddingUnknownBatch')
        
        simple_conv2D_model = SimpleConv2DSamePaddingUnknownHW()
        output_tensor = simple_conv2D_model(x_in)
        input_signature = [tf.TensorSpec(x_in.shape, x_in.dtype)]
        onnx_model, _ = tf2onnx.convert.from_function(simple_conv2D_model.__call__,input_signature, opset=onnx_opset_number)
        onnx_model=onnx.shape_inference.infer_shapes(onnx_model)
        expect(onnx_model, inputs=[x_in], outputs=[output_tensor.numpy()],
               name='test_SimpleConv2DSamePaddingUnknownHW')
