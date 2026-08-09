# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
# Taken from https://github.com/onnx/onnxmltools/blob/master/tests/end2end/test_custom_op.py.
import unittest

import numpy as np
import onnxmltools
from keras import Sequential
from keras import backend as K
from keras.layers import Conv2D, Layer, MaxPooling2D

import onnxruntime as onnxrt


class ScaledTanh(Layer):
    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return self.alpha * K.tanh(self.beta * x)

    def compute_output_shape(self, input_shape):
        return input_shape


def custom_activation(scope, operator, container):
    # type:(ScopeBase, OperatorBase, ModelContainer) -> None
    container.add_node(
        "ScaledTanh",
        operator.input_full_names,
        operator.output_full_names,
        op_version=1,
        alpha=operator.original_operator.alpha,
        beta=operator.original_operator.beta,
    )


class TestInferenceSessionKeras(unittest.TestCase):
    def test_run_model_conv(self):
        # keras model
        N, C, H, W = 2, 3, 5, 5  # noqa: N806
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

        model = Sequential()
        model.add(
            Conv2D(
                2,
                kernel_size=(1, 2),
                strides=(1, 1),
                padding="valid",
                input_shape=(H, W, C),
                data_format="channels_last",
            )
        )
        model.add(ScaledTanh(0.9, 2.0))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_last"))

        model.compile(optimizer="sgd", loss="mse")
        actual = model.predict(x)
        self.assertIsNotNone(actual)

        # conversion
        converted_model = onnxmltools.convert_keras(model, custom_conversion_functions={ScaledTanh: custom_activation})
        self.assertIsNotNone(converted_model)

        # runtime
        content = converted_model.SerializeToString()
        rt = onnxrt.InferenceSession(content, providers=onnxrt.get_available_providers())
        input = {rt.get_inputs()[0].name: x}
        actual_rt = rt.run(None, input)
        self.assertEqual(len(actual_rt), 1)
        np.testing.assert_allclose(actual, actual_rt[0], rtol=1e-05, atol=1e-08)


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
