# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import onnxruntime as ort


def sigmoid(z):
    """Computes the sigmoid of the given numpy array."""
    return 1 / (1 + np.exp(-z))


class GRU:
    """GRU utility class for testing.

    This class exposes four copmutation methods:
    - forward_np: computes the GRU forward pass using numpy
    - forward_ort: computes the GRU forward pass using ORT
    - backward_np: computes the GRU backward pass using numpy
    - backward_ort: computes the GRU backward pass using ORT
    and two onnx model generation methods:
    - forward_graph: generates the GRU forward onnx graph (with the GRUTraining node)
    - backward_graph: generates the GRU backward onnx graph (with the GRUGrad node)
    """

    def __init__(self, sequence_length, batch_size, input_size, hidden_size, linear_before_reset):
        """Initializes the GRU class.

        Args:
            sequence_length (int): the sequence length
            batch_size (int): the batch size
            input_size (int): the input size
            hidden_size (int): the hidden size
        """
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self.input_size = input_size
        self._hidden_size = hidden_size
        self._num_directions = 1
        self._linear_before_reset = linear_before_reset
        self._forward_model = None
        self._backward_model = None

    def forward_np(
        self,
        inputs,
        weights,
        recurrence_weights,
        bias=None,
        initial_hidden_state=None,
    ):
        """Computes the GRU forward pass using numpy.

        The computation follows the following rules:
        - zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        - rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        - ht = tanh(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
        - Ht = (1 - zt) (.) ht + zt (.) Ht-1


        Args:
            input (np.array): the input tensor of shape (sequence_length, batch_size, input_size)
            weights (np.array): the weight tensor of shape
                                (num_directions, 3 * hidden_size, input_size)
            recurrence_weights (np.array): the recurrence weight tensor of shape
                                           (num_directions, 3 * hidden_size, hidden_size)
            bias (np.array, optional): the bias tensor of shape
                                       (num_directions, 6 * hidden_size). Defaults to None.
            H0 (np.array, optional): the initial hidden state tensor of shape
                                     (num_directions, batch_size, hidden_size).
                                     Defaults to None.

        Returns:
            HAll (np.array): all hidden states tensor of shape
                             (sequence_length, num_directions, batch_size, hidden_size)
            HFinal (np.array): the final hidden state tensor of shape
                               (num_directions, batch_size, hidden_size)
            ZRH (np.array): all intermediate values of the gates tensor of shape
                             (sequence_length, num_directions, batch_size, 3 * hidden_size)
        """
        all_hidden_states = np.zeros(
            (self._sequence_length, self._num_directions, self._batch_size, self._hidden_size), np.float32
        )
        final_hidden_state = np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
        zrh = np.zeros(
            (self._sequence_length, self._num_directions, self._batch_size, 3 * self._hidden_size), np.float32
        )

        weights_z = np.squeeze(weights[:, : self._hidden_size, :], axis=0)
        weights_r = np.squeeze(weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
        weights_h = np.squeeze(weights[:, 2 * self._hidden_size :, :], axis=0)

        rweights_z = np.squeeze(recurrence_weights[:, : self._hidden_size, :], axis=0)
        rweights_r = np.squeeze(recurrence_weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
        rweights_h = np.squeeze(recurrence_weights[:, 2 * self._hidden_size :, :], axis=0)

        wb_z = (
            np.squeeze(bias[:, : self._hidden_size], axis=0)
            if bias is not None
            else np.zeros((self._hidden_size), np.float32)
        )
        wb_r = (
            np.squeeze(bias[:, self._hidden_size : 2 * self._hidden_size], axis=0)
            if bias is not None
            else np.zeros((self._hidden_size), np.float32)
        )
        wb_h = (
            np.squeeze(bias[:, 2 * self._hidden_size : 3 * self._hidden_size], axis=0)
            if bias is not None
            else np.zeros((self._hidden_size), np.float32)
        )
        rb_z = (
            np.squeeze(bias[:, 3 * self._hidden_size : 4 * self._hidden_size], axis=0)
            if bias is not None
            else np.zeros((self._hidden_size), np.float32)
        )
        rb_r = (
            np.squeeze(bias[:, 4 * self._hidden_size : 5 * self._hidden_size], axis=0)
            if bias is not None
            else np.zeros((self._hidden_size), np.float32)
        )
        rb_h = (
            np.squeeze(bias[:, 5 * self._hidden_size :], axis=0)
            if bias is not None
            else np.zeros((self._hidden_size), np.float32)
        )

        for idx in range(self._batch_size):
            prev_h = (
                initial_hidden_state[0, idx, :]
                if initial_hidden_state is not None
                else np.zeros((self._hidden_size), np.float32)
            )
            for t in range(self._sequence_length):
                current_input = inputs[t, idx, :]

                update_gate = sigmoid(np.dot(current_input, weights_z.T) + np.dot(prev_h, rweights_z.T) + wb_z + rb_z)
                reset_gate = sigmoid(np.dot(current_input, weights_r.T) + np.dot(prev_h, rweights_r.T) + wb_r + rb_r)
                if self._linear_before_reset:
                    hidden_gate = np.tanh(
                        np.dot(current_input, weights_h.T) + (reset_gate * (np.dot(prev_h, rweights_h.T) + rb_h)) + wb_h
                    )
                else:
                    hidden_gate = np.tanh(
                        np.dot(current_input, weights_h.T) + np.dot((reset_gate * prev_h), rweights_h.T) + wb_h + rb_h
                    )

                zrh[t, 0, idx, : self._hidden_size] = update_gate
                zrh[t, 0, idx, self._hidden_size : 2 * self._hidden_size] = reset_gate
                zrh[t, 0, idx, 2 * self._hidden_size :] = hidden_gate

                final_hidden_state[0, idx, :] = ((1 - update_gate) * hidden_gate) + (update_gate * prev_h)

                all_hidden_states[t, 0, idx, :] = final_hidden_state[0, idx, :]

                prev_h = final_hidden_state[0, idx, :]

        return all_hidden_states, final_hidden_state, zrh

    def forward_ort(
        self,
        inputs,
        weights,
        recurrence_weights,
        bias=None,
        initial_hidden_state=None,
    ):
        """Run GRU forward pass using ONNX Runtime."""
        ort_inputs = {"inputs": inputs, "weights": weights, "recurrence_weights": recurrence_weights}
        if bias is not None:
            ort_inputs["bias"] = bias
        if initial_hidden_state is not None:
            ort_inputs["initial_hidden_state"] = initial_hidden_state

        ort_outs = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "gru.onnx")
            onnx.save(self._forward_model, model_path)
            ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            ort_outs = ort_session.run(None, ort_inputs)

        return ort_outs

    def forward_graph(
        self,
        bias=True,
        sequence_lengths=False,
        initial_hidden_state=True,
    ):
        """Create a graph for GRU forward pass."""
        inputs = helper.make_tensor_value_info(
            "inputs", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        weights = helper.make_tensor_value_info(
            "weights", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size, self.input_size]
        )
        recurrence_weights = helper.make_tensor_value_info(
            "recurrence_weights", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size, self._hidden_size]
        )
        bias = (
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [self._num_directions, 6 * self._hidden_size])
            if bias
            else None
        )
        sequence_lengths = (
            helper.make_tensor_value_info("sequence_lengths", TensorProto.INT64, [self._batch_size])
            if sequence_lengths
            else None
        )
        initial_hidden_state = (
            helper.make_tensor_value_info(
                "initial_hidden_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if initial_hidden_state
            else None
        )

        all_hidden_states = helper.make_tensor_value_info(
            "all_hidden_states",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        final_hidden_state = helper.make_tensor_value_info(
            "final_hidden_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
        )
        zrh = helper.make_tensor_value_info(
            "zrh",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, 3 * self._hidden_size],
        )

        gru = helper.make_node(
            "GRUTraining",
            inputs=[
                "inputs",
                "weights",
                "recurrence_weights",
                "bias" if bias else "",
                "sequence_lengths" if sequence_lengths else "",
                "initial_hidden_state" if initial_hidden_state else "",
            ],
            outputs=["all_hidden_states", "final_hidden_state", "zrh"],
            domain="com.microsoft",
            hidden_size=self._hidden_size,
            linear_before_reset=1 if self._linear_before_reset else 0,
        )

        graph = helper.make_graph(
            [gru],
            "gru",
            [
                gi
                for gi in [
                    inputs,
                    weights,
                    recurrence_weights,
                    bias,
                    sequence_lengths,
                    initial_hidden_state,
                ]
                if gi
            ],
            [all_hidden_states, final_hidden_state, zrh],
        )

        self._forward_model = helper.make_model(
            graph,
            producer_name="orttraining",
            opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)],
        )

        return self._forward_model

    def backward_np(
        self,
        inputs,
        weights,
        recurrence_weights,
        bias,
        initial_hidden_state,
        all_hidden_states,
        zrh,
        grad_all_hidden_states,
        grad_final_hidden_state=None,
    ):
        """Compute the backward pass of GRU using numpy.

        The computation follows the following rules:
        - zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        - rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        - ht = tanh(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
        - Ht = (1 - zt) (.) ht + zt (.) Ht-1


        Args:
            inputs (np.ndarray): input tensor of shape (sequence_length, batch_size, input_size)
            weights (np.ndarray): weight tensor of shape (num_directions, 3 * hidden_size, input_size)
            recurrence_weights (np.ndarray): recurrence weight tensor of shape (num_directions, 3 * hidden_size, hidden_size)
            bias (bool): whether to compute the bias gradient or not
            initial_hidden_state (np.ndarray): initial hidden state tensor of shape (num_directions, batch_size, hidden_size)
            all_hidden_states (np.ndarray): output tensor of shape (sequence_length, num_directions, batch_size, hidden_size)
            zrh (np.ndarray): update, reset and hidden gate tensor of shape (sequence_length, num_directions, batch_size, 3 * hidden_size)
            grad_all_hidden_states (np.ndarray): gradient of HAll
            grad_final_hidden_state (np.ndarray): gradient of Ht

        Returns:
            tuple[np.ndarray]: gradients of inputs, weights, recurrence_weights, bias, initial_hidden_state
        """
        grad_inputs = np.zeros((self._sequence_length, self._batch_size, self.input_size), np.float32)
        grad_weights = np.zeros((self._num_directions, 3 * self._hidden_size, self.input_size), np.float32)
        grad_recurrence_weights = np.zeros((self._num_directions, 3 * self._hidden_size, self._hidden_size), np.float32)
        grad_bias = np.zeros((self._num_directions, 6 * self._hidden_size), np.float32) if bias is not None else None
        grad_initial_hidden_state = (
            np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
            if initial_hidden_state is not None
            else None
        )

        weights_z = np.squeeze(weights[:, : self._hidden_size, :], axis=0)
        weights_r = np.squeeze(weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
        weights_h = np.squeeze(weights[:, 2 * self._hidden_size :, :], axis=0)

        rweights_z = np.squeeze(recurrence_weights[:, : self._hidden_size, :], axis=0)
        rweights_r = np.squeeze(recurrence_weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
        rweights_h = np.squeeze(recurrence_weights[:, 2 * self._hidden_size :, :], axis=0)

        rb_h = (
            np.squeeze(bias[:, 5 * self._hidden_size :], axis=0)
            if bias is not None
            else np.zeros((self._hidden_size), np.float32)
        )

        for idx in range(self._batch_size):
            grad_h = (
                grad_final_hidden_state[0, idx, :]
                if grad_final_hidden_state is not None
                else np.zeros((self._hidden_size), np.float32)
            )
            for t in reversed(range(self._sequence_length)):
                current_input = inputs[t, idx, :]

                update_gate = zrh[t, 0, idx, : self._hidden_size]
                reset_gate = zrh[t, 0, idx, self._hidden_size : 2 * self._hidden_size]
                hidden_gate = zrh[t, 0, idx, 2 * self._hidden_size :]

                grad_h += grad_all_hidden_states[t, 0, idx, :]

                prev_h = (
                    all_hidden_states[t - 1, 0, idx, :]
                    if t > 0
                    else initial_hidden_state[0, idx, :] if initial_hidden_state is not None else 0
                )

                grad_update_gate = (prev_h - hidden_gate) * grad_h
                grad_hidden_gate = grad_h * (1 - update_gate)

                grad_update_activation = grad_update_gate * update_gate * (1 - update_gate)
                grad_hidden_activation = grad_hidden_gate * (1 - (hidden_gate * hidden_gate))

                if self._linear_before_reset:
                    grad_reset_gate = grad_hidden_activation * (np.dot(prev_h, rweights_h.T) + rb_h)
                else:
                    grad_reset_gate = np.dot(grad_hidden_activation, rweights_h) * prev_h
                grad_reset_activation = grad_reset_gate * reset_gate * (1 - reset_gate)

                grad_inputs[t, idx, :] = (
                    np.dot(grad_update_activation, weights_z)
                    + np.dot(grad_reset_activation, weights_r)
                    + np.dot(grad_hidden_activation, weights_h)
                )
                if self._linear_before_reset:
                    grad_h = (
                        grad_h * update_gate
                        + np.dot(grad_update_activation, rweights_z)
                        + np.dot(grad_reset_activation, rweights_r)
                        + np.dot(grad_hidden_activation * reset_gate, rweights_h)
                    )
                else:
                    grad_h = (
                        grad_h * update_gate
                        + np.dot(grad_update_activation, rweights_z)
                        + np.dot(grad_reset_activation, rweights_r)
                        + (np.dot(grad_hidden_activation, rweights_h) * reset_gate)
                    )

                if t == 0 and grad_initial_hidden_state is not None:
                    grad_initial_hidden_state[0, idx, :] = grad_h

                grad_weights[0, : self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_update_activation, axis=0).T, np.expand_dims(current_input, axis=0)
                )
                grad_weights[0, self._hidden_size : 2 * self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_reset_activation, axis=0).T, np.expand_dims(current_input, axis=0)
                )
                grad_weights[0, 2 * self._hidden_size :, :] += np.dot(
                    np.expand_dims(grad_hidden_activation, axis=0).T, np.expand_dims(current_input, axis=0)
                )

                grad_recurrence_weights[0, : self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_update_activation, axis=0).T, np.expand_dims(prev_h, axis=0)
                )
                grad_recurrence_weights[0, self._hidden_size : 2 * self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_reset_activation, axis=0).T, np.expand_dims(prev_h, axis=0)
                )
                if self._linear_before_reset:
                    grad_recurrence_weights[0, 2 * self._hidden_size :, :] += np.dot(
                        np.expand_dims(grad_hidden_activation * reset_gate, axis=0).T, np.expand_dims(prev_h, axis=0)
                    )
                else:
                    grad_recurrence_weights[0, 2 * self._hidden_size :, :] += np.dot(
                        np.expand_dims(grad_hidden_activation, axis=0).T, np.expand_dims(prev_h * reset_gate, axis=0)
                    )

                if grad_bias is not None:
                    grad_bias[0, : self._hidden_size] += grad_update_activation
                    grad_bias[0, self._hidden_size : 2 * self._hidden_size] += grad_reset_activation
                    grad_bias[0, 2 * self._hidden_size : 3 * self._hidden_size] += grad_hidden_activation
                    grad_bias[0, 3 * self._hidden_size : 4 * self._hidden_size] += grad_update_activation
                    grad_bias[0, 4 * self._hidden_size : 5 * self._hidden_size] += grad_reset_activation
                    if self._linear_before_reset:
                        grad_bias[0, 5 * self._hidden_size :] = grad_bias[0, 5 * self._hidden_size :] + (
                            grad_hidden_activation * reset_gate
                        )
                    else:
                        grad_bias[0, 5 * self._hidden_size :] += grad_hidden_activation

        return tuple(
            [
                out
                for out in [
                    grad_inputs,
                    grad_weights,
                    grad_recurrence_weights,
                    grad_bias,
                    grad_initial_hidden_state,
                ]
                if out is not None
            ]
        )

    def backward_graph(
        self,
        bias=True,
        sequence_lengths=False,
        initial_hidden_state=True,
        final_hidden_state=False,
    ):
        """Generate the ONNX graph for the backward pass of the GRU operator."""
        inputs = helper.make_tensor_value_info(
            "inputs", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        weights = helper.make_tensor_value_info(
            "weights", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size, self.input_size]
        )
        recurrence_weights = helper.make_tensor_value_info(
            "recurrence_weights", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size, self._hidden_size]
        )
        bias = (
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [self._num_directions, 6 * self._hidden_size])
            if bias
            else None
        )
        sequence_lengths = (
            helper.make_tensor_value_info("sequence_lengths", TensorProto.INT64, [self._batch_size])
            if sequence_lengths
            else None
        )
        initial_hidden_state = (
            helper.make_tensor_value_info(
                "initial_hidden_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if initial_hidden_state
            else None
        )

        all_hidden_states = helper.make_tensor_value_info(
            "all_hidden_states",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        final_hidden_state = (
            helper.make_tensor_value_info(
                "final_hidden_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if final_hidden_state
            else None
        )
        zrh = helper.make_tensor_value_info(
            "zrh",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, 3 * self._hidden_size],
        )
        grad_all_hidden_states = helper.make_tensor_value_info(
            "grad_all_hidden_states",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        grad_final_hidden_state = (
            helper.make_tensor_value_info(
                "grad_final_hidden_state",
                TensorProto.FLOAT,
                [self._num_directions, self._batch_size, self._hidden_size],
            )
            if final_hidden_state
            else None
        )

        grad_inputs = helper.make_tensor_value_info(
            "grad_inputs", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        grad_weights = helper.make_tensor_value_info(
            "grad_weights", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size, self.input_size]
        )
        grad_recurrence_weights = helper.make_tensor_value_info(
            "grad_recurrence_weights",
            TensorProto.FLOAT,
            [self._num_directions, 3 * self._hidden_size, self._hidden_size],
        )
        grad_bias = (
            helper.make_tensor_value_info("grad_bias", TensorProto.FLOAT, [self._num_directions, 6 * self._hidden_size])
            if bias
            else None
        )
        grad_initial_hidden_state = (
            helper.make_tensor_value_info(
                "grad_initial_hidden_state",
                TensorProto.FLOAT,
                [self._num_directions, self._batch_size, self._hidden_size],
            )
            if initial_hidden_state
            else None
        )

        gru = helper.make_node(
            "GRUGrad",
            inputs=[
                "inputs",
                "weights",
                "recurrence_weights",
                "bias" if bias is not None else "",
                "sequence_lengths" if sequence_lengths is not None else "",
                "initial_hidden_state" if initial_hidden_state is not None else "",
                "all_hidden_states" if all_hidden_states is not None else "",
                "zrh" if zrh is not None else "",
                "grad_all_hidden_states" if grad_all_hidden_states is not None else "",
                "grad_final_hidden_state" if grad_final_hidden_state is not None else "",
            ],
            outputs=[
                "grad_inputs",
                "grad_weights",
                "grad_recurrence_weights",
                "grad_bias" if grad_bias is not None else "",
                "grad_initial_hidden_state" if grad_initial_hidden_state is not None else "",
            ],
            domain="com.microsoft",
            hidden_size=self._hidden_size,
            linear_before_reset=1 if self._linear_before_reset else 0,
        )

        graph = helper.make_graph(
            [gru],
            "gru",
            [
                gi
                for gi in [
                    inputs,
                    weights,
                    recurrence_weights,
                    bias,
                    sequence_lengths,
                    initial_hidden_state,
                    all_hidden_states,
                    zrh,
                    grad_all_hidden_states,
                    grad_final_hidden_state,
                ]
                if gi
            ],
            [
                go
                for go in [
                    grad_inputs,
                    grad_weights,
                    grad_recurrence_weights,
                    grad_bias,
                    grad_initial_hidden_state,
                ]
                if go
            ],
        )

        self._backward_model = helper.make_model(
            graph,
            producer_name="orttraining",
            opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)],
        )

        return self._backward_model

    def backward_ort(
        self,
        inputs,
        weights,
        recurrence_weights,
        bias=None,
        initial_hidden_state=None,
        all_hidden_states=None,
        zrh=None,
        grad_all_hidden_states=None,
        grad_final_hidden_state=None,
    ):
        """Run GRU backward using ONNX Runtime.

        Users must call backward_graph before calling this function.
        """
        ort_inputs = {
            "inputs": inputs,
            "weights": weights,
            "recurrence_weights": recurrence_weights,
            "all_hidden_states": all_hidden_states,
            "zrh": zrh,
            "grad_all_hidden_states": grad_all_hidden_states,
        }
        if bias is not None:
            ort_inputs["bias"] = bias
        if initial_hidden_state is not None:
            ort_inputs["initial_hidden_state"] = initial_hidden_state
        if grad_final_hidden_state is not None:
            ort_inputs["grad_final_hidden_state"] = grad_final_hidden_state

        ort_outs = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "gru_gradient.onnx")
            onnx.save(self._backward_model, model_path)
            ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs


@pytest.mark.parametrize("sequence_length", [2, 4, 16, 32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("input_size", [32])
@pytest.mark.parametrize("hidden_size", [32])
@pytest.mark.parametrize("linear_before_reset", [True, False])
def test_gru_forward(sequence_length, batch_size, input_size, hidden_size, linear_before_reset):
    num_directions = 1

    gru = GRU(sequence_length, batch_size, input_size, hidden_size, linear_before_reset)
    _ = gru.forward_graph(bias=True, sequence_lengths=False, initial_hidden_state=True)

    inputs = np.random.rand(sequence_length, batch_size, input_size).astype(np.float32)
    weights = np.random.rand(num_directions, 3 * hidden_size, input_size).astype(np.float32)
    recurrence_weights = np.random.rand(num_directions, 3 * hidden_size, hidden_size).astype(np.float32)
    bias = np.random.rand(num_directions, 6 * hidden_size).astype(np.float32)
    initial_hidden_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)

    outs_ort = gru.forward_ort(inputs, weights, recurrence_weights, bias, initial_hidden_state)
    outs_np = gru.forward_np(inputs, weights, recurrence_weights, bias, initial_hidden_state)

    for ort_out, np_out in zip(outs_ort, outs_np):
        assert np.allclose(ort_out, np_out, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize("sequence_length", [2, 4, 16, 32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("input_size", [32])
@pytest.mark.parametrize("hidden_size", [32])
@pytest.mark.parametrize("linear_before_reset", [True, False])
def test_gru_backward(sequence_length, batch_size, input_size, hidden_size, linear_before_reset):
    np.random.seed(seed=None)
    num_directions = 1

    gru = GRU(sequence_length, batch_size, input_size, hidden_size, linear_before_reset)
    _ = gru.backward_graph(bias=True, sequence_lengths=False, initial_hidden_state=True, final_hidden_state=True)

    inputs = np.random.rand(sequence_length, batch_size, input_size).astype(np.float32)
    weights = np.random.rand(num_directions, 3 * hidden_size, input_size).astype(np.float32)
    recurrence_weights = np.random.rand(num_directions, 3 * hidden_size, hidden_size).astype(np.float32)
    bias = np.random.rand(num_directions, 6 * hidden_size).astype(np.float32)
    initial_hidden_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)

    all_hidden_states = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    zrh = np.random.rand(sequence_length, num_directions, batch_size, 3 * hidden_size).astype(np.float32)
    grad_all_hidden_states = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    grad_final_hidden_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)

    outs_ort = gru.backward_ort(
        inputs,
        weights,
        recurrence_weights,
        bias,
        initial_hidden_state,
        all_hidden_states,
        zrh,
        grad_all_hidden_states,
        grad_final_hidden_state,
    )
    outs_np = gru.backward_np(
        inputs,
        weights,
        recurrence_weights,
        bias,
        initial_hidden_state,
        all_hidden_states,
        zrh,
        grad_all_hidden_states,
        grad_final_hidden_state,
    )

    for ort_out, np_out in zip(outs_ort, outs_np):
        assert np.allclose(ort_out, np_out, rtol=1e-01, atol=1e-03)
