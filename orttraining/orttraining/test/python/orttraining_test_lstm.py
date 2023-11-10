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


class LSTM:
    """LSTM utility class for testing.

    This class exposes four copmutation methods:
    - forward_np: computes the LSTM forward pass using numpy
    - forward_ort: computes the LSTM forward pass using ORT
    - backward_np: computes the LSTM backward pass using numpy
    - backward_ort: computes the LSTM backward pass using ORT
    and two onnx model generation methods:
    - forward_graph: generates the LSTM forward onnx graph (with the LSTMTraining node)
    - backward_graph: generates the LSTM backward onnx graph (with the LSTMGrad node)
    """

    def __init__(self, sequence_length, batch_size, input_size, hidden_size):
        """Initializes the LSTM class.

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
        self._forward_model = None
        self._backward_model = None

    def forward_np(
        self,
        inputs,
        weights,
        recurrence_weights,
        bias=None,
        initial_hidden_state=None,
        initial_cell_state=None,
        peephole_weights=None,
    ):
        """Computes the LSTM forward pass using numpy.

        The computation follows the following rules:
        - it = sigmoid(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        - ft = sigmoid(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        - ct = tanh(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        - Ct = ft (.) Ct-1 + it (.) ct
        - ot = sigmoid(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        - Ht = ot (.) h(Ct)


        Args:
            input (np.array): the input tensor of shape (sequence_length, batch_size, input_size)
            weights (np.array): the weight tensor of shape
                                (num_directions, 4 * hidden_size, input_size)
            recurrence_weights (np.array): the recurrence weight tensor of shape
                                           (num_directions, 4 * hidden_size, hidden_size)
            bias (np.array, optional): the bias tensor of shape
                                       (num_directions, 8 * hidden_size). Defaults to None.
            H0 (np.array, optional): the initial hidden state tensor of shape
                                     (num_directions, batch_size, hidden_size).
                                     Defaults to None.
            C0 (np.array, optional): the initial cell state tensor of shape
                                     (num_directions, batch_size, hidden_size).
                                     Defaults to None.
            P (np.array, optional): the peephole weight tensor of shape
                                    (num_directions, 3 * hidden_size).
                                    Defaults to None.

        Returns:
            HAll (np.array): all hidden states tensor of shape
                             (sequence_length, num_directions, batch_size, hidden_size)
            HFinal (np.array): the final hidden state tensor of shape
                               (num_directions, batch_size, hidden_size)
            CFinal (np.array): the final cell state tensor of shape
                               (num_directions, batch_size, hidden_size)
            CAll (np.array): all cell states tensor of shape
                             (sequence_length, num_directions, batch_size, hidden_size)
            IOFC (np.array): all intermediate values of the gates tensor of shape
                             (sequence_length, num_directions, batch_size, 4 * hidden_size)
        """
        all_hidden_states = np.zeros(
            (self._sequence_length, self._num_directions, self._batch_size, self._hidden_size), np.float32
        )
        final_hidden_state = np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
        final_cell_state = np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
        all_cell_states = np.zeros(
            (self._sequence_length, self._num_directions, self._batch_size, self._hidden_size), np.float32
        )
        iofc = np.zeros(
            (self._sequence_length, self._num_directions, self._batch_size, 4 * self._hidden_size), np.float32
        )
        for idx in range(self._batch_size):
            prev_h = (
                initial_hidden_state[0, idx, :]
                if initial_hidden_state is not None
                else np.zeros((self._hidden_size), np.float32)
            )
            prev_c = (
                initial_cell_state[0, idx, :]
                if initial_cell_state is not None
                else np.zeros((self._hidden_size), np.float32)
            )
            for t in range(self._sequence_length):
                current_input = inputs[t, idx, :]
                weights_i = np.squeeze(weights[:, : self._hidden_size, :], axis=0)
                weights_o = np.squeeze(weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                weights_f = np.squeeze(weights[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                weights_c = np.squeeze(weights[:, 3 * self._hidden_size :, :], axis=0)

                rweights_i = np.squeeze(recurrence_weights[:, : self._hidden_size, :], axis=0)
                rweights_o = np.squeeze(recurrence_weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                rweights_f = np.squeeze(recurrence_weights[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                rweights_c = np.squeeze(recurrence_weights[:, 3 * self._hidden_size :, :], axis=0)

                wb_i = (
                    np.squeeze(bias[:, : self._hidden_size], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                wb_o = (
                    np.squeeze(bias[:, self._hidden_size : 2 * self._hidden_size], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                wb_f = (
                    np.squeeze(bias[:, 2 * self._hidden_size : 3 * self._hidden_size], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                wb_c = (
                    np.squeeze(bias[:, 3 * self._hidden_size : 4 * self._hidden_size], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                rb_i = (
                    np.squeeze(bias[:, 4 * self._hidden_size : 5 * self._hidden_size], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                rb_o = (
                    np.squeeze(bias[:, 5 * self._hidden_size : 6 * self._hidden_size], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                rb_f = (
                    np.squeeze(bias[:, 6 * self._hidden_size : 7 * self._hidden_size], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                rb_c = (
                    np.squeeze(bias[:, 7 * self._hidden_size :], axis=0)
                    if bias is not None
                    else np.zeros((self._hidden_size), np.float32)
                )

                peephole_weights_i = (
                    np.squeeze(peephole_weights[:, : self._hidden_size], axis=0)
                    if peephole_weights is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                peephole_weights_o = (
                    np.squeeze(peephole_weights[:, self._hidden_size : 2 * self._hidden_size], axis=0)
                    if peephole_weights is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                peephole_weights_f = (
                    np.squeeze(peephole_weights[:, 2 * self._hidden_size : 3 * self._hidden_size], axis=0)
                    if peephole_weights is not None
                    else np.zeros((self._hidden_size), np.float32)
                )

                input_gate = sigmoid(
                    np.dot(current_input, weights_i.T)
                    + np.dot(prev_h, rweights_i.T)
                    + wb_i
                    + rb_i
                    + peephole_weights_i * prev_c
                )
                forget_gate = sigmoid(
                    np.dot(current_input, weights_f.T)
                    + np.dot(prev_h, rweights_f.T)
                    + wb_f
                    + rb_f
                    + peephole_weights_f * prev_c
                )
                control_gate = np.tanh(np.dot(current_input, weights_c.T) + np.dot(prev_h, rweights_c.T) + wb_c + rb_c)

                final_cell_state[0, idx, :] = forget_gate * prev_c + input_gate * control_gate

                output_gate = sigmoid(
                    np.dot(current_input, weights_o.T)
                    + np.dot(prev_h, rweights_o.T)
                    + wb_o
                    + rb_o
                    + peephole_weights_o * final_cell_state[0, idx, :]
                )

                iofc[t, 0, idx, : self._hidden_size] = input_gate
                iofc[t, 0, idx, self._hidden_size : 2 * self._hidden_size] = output_gate
                iofc[t, 0, idx, 2 * self._hidden_size : 3 * self._hidden_size] = forget_gate
                iofc[t, 0, idx, 3 * self._hidden_size :] = control_gate

                final_hidden_state[0, idx, :] = output_gate * np.tanh(final_cell_state[0, idx, :])

                all_hidden_states[t, 0, idx, :] = final_hidden_state[0, idx, :]
                all_cell_states[t, 0, idx, :] = final_cell_state[0, idx, :]

                prev_h = final_hidden_state[0, idx, :]
                prev_c = final_cell_state[0, idx, :]

        return all_hidden_states, final_hidden_state, final_cell_state, all_cell_states, iofc

    def forward_ort(
        self,
        inputs,
        weights,
        recurrence_weights,
        bias=None,
        initial_hidden_state=None,
        initial_cell_state=None,
        peephole_weights=None,
    ):
        """Run LSTM forward pass using ONNX Runtime."""
        ort_inputs = {"inputs": inputs, "weights": weights, "recurrence_weights": recurrence_weights}
        if bias is not None:
            ort_inputs["bias"] = bias
        if initial_hidden_state is not None:
            ort_inputs["initial_hidden_state"] = initial_hidden_state
        if initial_cell_state is not None:
            ort_inputs["initial_cell_state"] = initial_cell_state
        if peephole_weights is not None:
            ort_inputs["peephole_weights"] = peephole_weights

        ort_outs = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "lstm.onnx")
            onnx.save(self._forward_model, model_path)
            ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            ort_outs = ort_session.run(None, ort_inputs)

        return ort_outs

    def forward_graph(
        self,
        bias=True,
        sequence_lengths=False,
        initial_hidden_state=True,
        initial_cell_state=True,
        peephole_weights=False,
    ):
        """Create a graph for LSTM forward pass."""
        inputs = helper.make_tensor_value_info(
            "inputs", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        weights = helper.make_tensor_value_info(
            "weights", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self.input_size]
        )
        recurrence_weights = helper.make_tensor_value_info(
            "recurrence_weights", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self._hidden_size]
        )
        bias = (
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [self._num_directions, 8 * self._hidden_size])
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
        initial_cell_state = (
            helper.make_tensor_value_info(
                "initial_cell_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if initial_cell_state
            else None
        )
        peephole_weights = (
            helper.make_tensor_value_info(
                "peephole_weights", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size]
            )
            if peephole_weights
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
        final_cell_state = helper.make_tensor_value_info(
            "final_cell_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
        )
        all_cell_states = helper.make_tensor_value_info(
            "all_cell_states",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        iofc = helper.make_tensor_value_info(
            "iofc",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, 4 * self._hidden_size],
        )

        lstm = helper.make_node(
            "LSTMTraining",
            inputs=[
                "inputs",
                "weights",
                "recurrence_weights",
                "bias" if bias else "",
                "sequence_lengths" if sequence_lengths else "",
                "initial_hidden_state" if initial_hidden_state else "",
                "initial_cell_state" if initial_cell_state else "",
                "peephole_weights" if peephole_weights else "",
            ],
            outputs=["all_hidden_states", "final_hidden_state", "final_cell_state", "all_cell_states", "iofc"],
            domain="com.microsoft",
            hidden_size=self._hidden_size,
        )

        graph = helper.make_graph(
            [lstm],
            "lstm",
            [
                gi
                for gi in [
                    inputs,
                    weights,
                    recurrence_weights,
                    bias,
                    sequence_lengths,
                    initial_hidden_state,
                    initial_cell_state,
                    peephole_weights,
                ]
                if gi
            ],
            [all_hidden_states, final_hidden_state, final_cell_state, all_cell_states, iofc],
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
        initial_cell_state,
        peephole_weights,
        all_hidden_states,
        all_cell_states,
        iofc,
        grad_all_hidden_states,
        grad_final_hidden_state=None,
        grad_final_cell_state=None,
    ):
        """Compute the backward pass of LSTM using numpy.

        This is a reference implementation for testing purpose. The equations used here are from
        deriving the gradients based on:
        - it = sigmoid(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        - ft = sigmoid(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        - ct = tanh(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        - Ct = ft (.) Ct-1 + it (.) ct
        - ot = sigmoid(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        - Ht = ot (.) h(Ct)


        Args:
            inputs (np.ndarray): input tensor of shape (sequence_length, batch_size, input_size)
            weights (np.ndarray): weight tensor of shape (num_directions, 4 * hidden_size, input_size)
            recurrence_weights (np.ndarray): recurrence weight tensor of shape (num_directions, 4 * hidden_size, hidden_size)
            bias (bool): whether to compute the bias gradient or not
            initial_hidden_state (np.ndarray): initial hidden state tensor of shape (num_directions, batch_size, hidden_size)
            initial_cell_state (np.ndarray): initial cell state tensor of shape (num_directions, batch_size, hidden_size)
            peephole_weights (bool): whether to compute the peephole weights gradient or not
            all_hidden_states (np.ndarray): output tensor of shape (sequence_length, num_directions, batch_size, hidden_size)
            all_cell_states (np.ndarray): cell state tensor of shape (sequence_length, num_directions, batch_size, hidden_size)
            iofc (np.ndarray): input, output, forget, and cell gate tensor of shape (sequence_length, num_directions, batch_size, 4 * hidden_size)
            grad_all_hidden_states (np.ndarray): gradient of HAll
            grad_final_hidden_state (np.ndarray): gradient of Ht
            grad_final_cell_state (np.ndarray): gradient of Ct

        Returns:
            tuple[np.ndarray]: gradients of inputs, weights, recurrence_weights, bias, initial_hidden_state, initial_cell_state, peephole_weightsP
        """
        grad_inputs = np.zeros((self._sequence_length, self._batch_size, self.input_size), np.float32)
        grad_weights = np.zeros((self._num_directions, 4 * self._hidden_size, self.input_size), np.float32)
        grad_recurrence_weights = np.zeros((self._num_directions, 4 * self._hidden_size, self._hidden_size), np.float32)
        grad_bias = np.zeros((self._num_directions, 8 * self._hidden_size), np.float32) if bias is True else None
        grad_initial_hidden_state = (
            np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
            if initial_hidden_state is not None
            else None
        )
        grad_initial_cell_state = (
            np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
            if initial_cell_state is not None
            else None
        )
        grad_peephole_weights = (
            np.zeros((self._num_directions, 3 * self._hidden_size), np.float32) if peephole_weights is True else None
        )

        for idx in range(self._batch_size):
            grad_h = (
                grad_final_hidden_state[0, idx, :]
                if grad_final_hidden_state is not None
                else np.zeros((self._hidden_size), np.float32)
            )
            grad_c = (
                grad_final_cell_state[0, idx, :]
                if grad_final_cell_state is not None
                else np.zeros((self._hidden_size), np.float32)
            )
            for t in reversed(range(self._sequence_length)):
                current_input = inputs[t, idx, :]
                weights_i = np.squeeze(weights[:, : self._hidden_size, :], axis=0)
                weights_o = np.squeeze(weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                weights_f = np.squeeze(weights[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                weights_c = np.squeeze(weights[:, 3 * self._hidden_size :, :], axis=0)

                rweights_i = np.squeeze(recurrence_weights[:, : self._hidden_size, :], axis=0)
                rweights_o = np.squeeze(recurrence_weights[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                rweights_f = np.squeeze(recurrence_weights[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                rweights_c = np.squeeze(recurrence_weights[:, 3 * self._hidden_size :, :], axis=0)

                input_gate = iofc[t, 0, idx, : self._hidden_size]
                output_gate = iofc[t, 0, idx, self._hidden_size : 2 * self._hidden_size]
                forget_gate = iofc[t, 0, idx, 2 * self._hidden_size : 3 * self._hidden_size]
                control_gate = iofc[t, 0, idx, 3 * self._hidden_size :]

                grad_h += grad_all_hidden_states[t, 0, idx, :]
                grad_c += grad_h * output_gate * (1 - np.tanh(all_cell_states[t, 0, idx, :]) ** 2)
                grad_prev_c = grad_c * forget_gate
                if t == 0 and grad_initial_cell_state is not None:
                    grad_initial_cell_state[0, idx, :] = grad_prev_c

                grad_input_gate = grad_c * control_gate
                grad_output_gate = grad_h * np.tanh(all_cell_states[t, 0, idx, :])
                grad_forget_gate = grad_c * (
                    all_cell_states[t - 1, 0, idx, :]
                    if t > 0
                    else initial_cell_state[0, idx, :]
                    if initial_cell_state is not None
                    else 0
                )
                grad_control_gate = grad_c * input_gate

                grad_input_activation = grad_input_gate * input_gate * (1 - input_gate)
                grad_output_activation = grad_output_gate * output_gate * (1 - output_gate)
                grad_forget_activation = grad_forget_gate * forget_gate * (1 - forget_gate)
                grad_control_activation = grad_control_gate * (1 - control_gate**2)

                grad_inputs[t, idx, :] = (
                    np.dot(grad_input_activation, weights_i)
                    + np.dot(grad_output_activation, weights_o)
                    + np.dot(grad_forget_activation, weights_f)
                    + np.dot(grad_control_activation, weights_c)
                )
                grad_h = (
                    np.dot(grad_input_activation, rweights_i)
                    + np.dot(grad_output_activation, rweights_o)
                    + np.dot(grad_forget_activation, rweights_f)
                    + np.dot(grad_control_activation, rweights_c)
                )
                if t == 0 and grad_initial_hidden_state is not None:
                    grad_initial_hidden_state[0, idx, :] = grad_h

                grad_weights[0, : self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_input_activation, axis=0).T, np.expand_dims(current_input, axis=0)
                )
                grad_weights[0, self._hidden_size : 2 * self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_output_activation, axis=0).T, np.expand_dims(current_input, axis=0)
                )
                grad_weights[0, 2 * self._hidden_size : 3 * self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_forget_activation, axis=0).T, np.expand_dims(current_input, axis=0)
                )
                grad_weights[0, 3 * self._hidden_size :, :] += np.dot(
                    np.expand_dims(grad_control_activation, axis=0).T, np.expand_dims(current_input, axis=0)
                )

                prev_h = (
                    all_hidden_states[t - 1, 0, idx, :]
                    if t > 0
                    else initial_hidden_state[0, idx, :]
                    if initial_hidden_state is not None
                    else 0
                )
                grad_recurrence_weights[0, : self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_input_activation, axis=0).T, np.expand_dims(prev_h, axis=0)
                )
                grad_recurrence_weights[0, self._hidden_size : 2 * self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_output_activation, axis=0).T, np.expand_dims(prev_h, axis=0)
                )
                grad_recurrence_weights[0, 2 * self._hidden_size : 3 * self._hidden_size, :] += np.dot(
                    np.expand_dims(grad_forget_activation, axis=0).T, np.expand_dims(prev_h, axis=0)
                )
                grad_recurrence_weights[0, 3 * self._hidden_size :, :] += np.dot(
                    np.expand_dims(grad_control_activation, axis=0).T, np.expand_dims(prev_h, axis=0)
                )

                if grad_bias is not None:
                    grad_bias[0, : self._hidden_size] += grad_input_activation
                    grad_bias[0, self._hidden_size : 2 * self._hidden_size] += grad_output_activation
                    grad_bias[0, 2 * self._hidden_size : 3 * self._hidden_size] += grad_forget_activation
                    grad_bias[0, 3 * self._hidden_size : 4 * self._hidden_size] += grad_control_activation
                    grad_bias[0, 4 * self._hidden_size : 5 * self._hidden_size] += grad_input_activation
                    grad_bias[0, 5 * self._hidden_size : 6 * self._hidden_size] += grad_output_activation
                    grad_bias[0, 6 * self._hidden_size : 7 * self._hidden_size] += grad_forget_activation
                    grad_bias[0, 7 * self._hidden_size :] += grad_control_activation

                if grad_peephole_weights is not None:
                    grad_peephole_weights[0, : self._hidden_size] += grad_input_activation * (
                        all_cell_states[t - 1, 0, idx, :]
                        if t > 0
                        else initial_cell_state[0, idx, :]
                        if initial_cell_state is not None
                        else 0
                    )
                    grad_peephole_weights[0, self._hidden_size : 2 * self._hidden_size] += (
                        grad_output_activation * all_cell_states[t, 0, idx, :]
                    )
                    grad_peephole_weights[
                        0, 2 * self._hidden_size : 3 * self._hidden_size
                    ] += grad_forget_activation * (
                        all_cell_states[t - 1, 0, idx, :]
                        if t > 0
                        else initial_cell_state[0, idx, :]
                        if initial_cell_state is not None
                        else 0
                    )

                grad_c = grad_prev_c

        return tuple(
            [
                out
                for out in [
                    grad_inputs,
                    grad_weights,
                    grad_recurrence_weights,
                    grad_bias,
                    grad_initial_hidden_state,
                    grad_initial_cell_state,
                    grad_peephole_weights,
                ]
                if out is not None
            ]
        )

    def backward_graph(
        self,
        bias=True,
        sequence_lengths=False,
        initial_hidden_state=True,
        initial_cell_state=True,
        peephole_weights=False,
        final_hidden_state=False,
        final_cell_state=False,
    ):
        """Generate the ONNX graph for the backward pass of the LSTM operator."""
        inputs = helper.make_tensor_value_info(
            "inputs", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        weights = helper.make_tensor_value_info(
            "weights", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self.input_size]
        )
        recurrence_weights = helper.make_tensor_value_info(
            "recurrence_weights", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self._hidden_size]
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
        initial_cell_state = (
            helper.make_tensor_value_info(
                "initial_cell_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if initial_cell_state
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
        final_cell_state = (
            helper.make_tensor_value_info(
                "final_cell_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if final_cell_state
            else None
        )
        all_cell_states = helper.make_tensor_value_info(
            "all_cell_states",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        iofc = helper.make_tensor_value_info(
            "iofc",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, 4 * self._hidden_size],
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
        grad_final_cell_state = (
            helper.make_tensor_value_info(
                "grad_final_cell_state", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if final_cell_state
            else None
        )

        grad_inputs = helper.make_tensor_value_info(
            "grad_inputs", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        grad_weights = helper.make_tensor_value_info(
            "grad_weights", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self.input_size]
        )
        grad_recurrence_weights = helper.make_tensor_value_info(
            "grad_recurrence_weights",
            TensorProto.FLOAT,
            [self._num_directions, 4 * self._hidden_size, self._hidden_size],
        )
        grad_bias = (
            helper.make_tensor_value_info("grad_bias", TensorProto.FLOAT, [self._num_directions, 8 * self._hidden_size])
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
        grad_initial_cell_state = (
            helper.make_tensor_value_info(
                "grad_initial_cell_state",
                TensorProto.FLOAT,
                [self._num_directions, self._batch_size, self._hidden_size],
            )
            if initial_cell_state
            else None
        )
        grad_peephole_weights = (
            helper.make_tensor_value_info(
                "grad_peephole_weights", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size]
            )
            if peephole_weights
            else None
        )

        lstm = helper.make_node(
            "LSTMGrad",
            inputs=[
                "inputs",
                "weights",
                "recurrence_weights",
                "sequence_lengths" if sequence_lengths is not None else "",
                "initial_hidden_state" if initial_hidden_state is not None else "",
                "initial_cell_state" if initial_cell_state is not None else "",
                "all_hidden_states" if all_hidden_states is not None else "",
                "all_cell_states" if all_cell_states is not None else "",
                "iofc" if iofc is not None else "",
                "grad_all_hidden_states" if grad_all_hidden_states is not None else "",
                "grad_final_hidden_state" if grad_final_hidden_state is not None else "",
                "grad_final_cell_state" if grad_final_cell_state else "",
            ],
            outputs=[
                "grad_inputs",
                "grad_weights",
                "grad_recurrence_weights",
                "grad_bias" if grad_bias is not None else "",
                "grad_initial_hidden_state" if grad_initial_hidden_state is not None else "",
                "grad_initial_cell_state" if grad_initial_cell_state is not None else "",
                "grad_peephole_weights" if grad_peephole_weights is not None else "",
            ],
            domain="com.microsoft",
            hidden_size=self._hidden_size,
        )

        graph = helper.make_graph(
            [lstm],
            "lstm",
            [
                gi
                for gi in [
                    inputs,
                    weights,
                    recurrence_weights,
                    sequence_lengths,
                    initial_hidden_state,
                    initial_cell_state,
                    all_hidden_states,
                    all_cell_states,
                    iofc,
                    grad_all_hidden_states,
                    grad_final_hidden_state,
                    grad_final_cell_state,
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
                    grad_initial_cell_state,
                    grad_peephole_weights,
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
        initial_hidden_state=None,
        initial_cell_state=None,
        all_hidden_states=None,
        all_cell_states=None,
        iofc=None,
        grad_all_hidden_states=None,
        grad_final_hidden_state=None,
        grad_final_cell_state=None,
    ):
        """Run LSTM backward using ONNX Runtime.

        Users must call backward_graph before calling this function.
        """
        ort_inputs = {
            "inputs": inputs,
            "weights": weights,
            "recurrence_weights": recurrence_weights,
            "all_hidden_states": all_hidden_states,
            "all_cell_states": all_cell_states,
            "iofc": iofc,
            "grad_all_hidden_states": grad_all_hidden_states,
        }
        if initial_hidden_state is not None:
            ort_inputs["initial_hidden_state"] = initial_hidden_state
        if initial_cell_state is not None:
            ort_inputs["initial_cell_state"] = initial_cell_state
        if grad_final_hidden_state is not None:
            ort_inputs["grad_final_hidden_state"] = grad_final_hidden_state
        if grad_final_cell_state is not None:
            ort_inputs["grad_final_cell_state"] = grad_final_cell_state

        ort_outs = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "lstm_gradient.onnx")
            onnx.save(self._backward_model, model_path)
            ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs


@pytest.mark.parametrize("sequence_length", [2, 4, 16, 32])
@pytest.mark.parametrize("batch_size", [2, 4, 32])
@pytest.mark.parametrize("input_size", [2, 4, 32])
@pytest.mark.parametrize("hidden_size", [2, 4, 32])
def test_lstm_forward(sequence_length, batch_size, input_size, hidden_size):
    num_directions = 1

    lstm = LSTM(sequence_length, batch_size, input_size, hidden_size)
    _ = lstm.forward_graph(
        bias=True, sequence_lengths=False, initial_hidden_state=True, initial_cell_state=True, peephole_weights=False
    )

    inputs = np.random.rand(sequence_length, batch_size, input_size).astype(np.float32)
    weights = np.random.rand(num_directions, 4 * hidden_size, input_size).astype(np.float32)
    recurrence_weights = np.random.rand(num_directions, 4 * hidden_size, hidden_size).astype(np.float32)
    bias = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)
    initial_hidden_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    initial_cell_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    peephole_weights = None

    outs_ort = lstm.forward_ort(
        inputs, weights, recurrence_weights, bias, initial_hidden_state, initial_cell_state, peephole_weights
    )
    outs_np = lstm.forward_np(
        inputs, weights, recurrence_weights, bias, initial_hidden_state, initial_cell_state, peephole_weights
    )

    for ort_out, np_out in zip(outs_ort, outs_np):
        assert np.allclose(ort_out, np_out, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize("sequence_length", [2, 4, 16, 32])
@pytest.mark.parametrize("batch_size", [2, 4, 32])
@pytest.mark.parametrize("input_size", [2, 4, 32])
@pytest.mark.parametrize("hidden_size", [2, 4, 32])
def test_lstm_backward(sequence_length, batch_size, input_size, hidden_size):
    num_directions = 1

    lstm = LSTM(sequence_length, batch_size, input_size, hidden_size)
    _ = lstm.backward_graph(
        bias=True,
        sequence_lengths=False,
        initial_hidden_state=True,
        initial_cell_state=True,
        peephole_weights=False,
        final_hidden_state=True,
        final_cell_state=True,
    )

    inputs = np.random.rand(sequence_length, batch_size, input_size).astype(np.float32)
    weights = np.random.rand(num_directions, 4 * hidden_size, input_size).astype(np.float32)
    recurrence_weights = np.random.rand(num_directions, 4 * hidden_size, hidden_size).astype(np.float32)
    bias = True
    initial_hidden_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    initial_cell_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    peephole_weights = False

    all_hidden_states = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    all_cell_states = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    iofc = np.random.rand(sequence_length, num_directions, batch_size, 4 * hidden_size).astype(np.float32)
    grad_all_hidden_states = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    grad_final_hidden_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    grad_final_cell_state = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)

    outs_ort = lstm.backward_ort(
        inputs,
        weights,
        recurrence_weights,
        initial_hidden_state,
        initial_cell_state,
        all_hidden_states,
        all_cell_states,
        iofc,
        grad_all_hidden_states,
        grad_final_hidden_state,
        grad_final_cell_state,
    )
    outs_np = lstm.backward_np(
        inputs,
        weights,
        recurrence_weights,
        bias,
        initial_hidden_state,
        initial_cell_state,
        peephole_weights,
        all_hidden_states,
        all_cell_states,
        iofc,
        grad_all_hidden_states,
        grad_final_hidden_state,
        grad_final_cell_state,
    )

    for ort_out, np_out in zip(outs_ort, outs_np):
        assert np.allclose(ort_out, np_out, rtol=1e-03, atol=1e-05)
