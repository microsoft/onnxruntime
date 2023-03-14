import numpy as np
import tempfile
import os

import onnx
from onnx import TensorProto, helper

import onnxruntime as ort
import pytest


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

    def forward_np(self, X, W, R, B=None, H0=None, C0=None, P=None):
        """Computes the LSTM forward pass using numpy.

        The computation follows the following rules:
        - it = sigmoid(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        - ft = sigmoid(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        - ct = tanh(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        - Ct = ft (.) Ct-1 + it (.) ct
        - ot = sigmoid(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        - Ht = ot (.) h(Ct)


        Args:
            X (np.array): the input tensor of shape (sequence_length, batch_size, input_size)
            W (np.array): the weight tensor of shape (num_directions, 4 * hidden_size, input_size)
            R (np.array): the recurrence weight tensor of shape (num_directions, 4 * hidden_size, hidden_size)
            B (np.array, optional): the bias tensor of shape (num_directions, 8 * hidden_size). Defaults to None.
            H0 (np.array, optional): the initial hidden state tensor of shape (num_directions, batch_size, hidden_size). Defaults to None.
            C0 (np.array, optional): the initial cell state tensor of shape (num_directions, batch_size, hidden_size). Defaults to None.
            P (np.array, optional): the peephole weight tensor of shape (num_directions, 3 * hidden_size). Defaults to None.

        Returns:
            HAll (np.array): all hidden states tensor of shape (sequence_length, num_directions, batch_size, hidden_size)
            HFinal (np.array): the final hidden state tensor of shape (num_directions, batch_size, hidden_size)
            CFinal (np.array): the final cell state tensor of shape (num_directions, batch_size, hidden_size)
            CAll (np.array): all cell states tensor of shape (sequence_length, num_directions, batch_size, hidden_size)
            IOFC (np.array): all intermediate values of the gates tensor of shape (sequence_length, num_directions, batch_size, 4 * hidden_size)
        """
        HAll = np.zeros((self._sequence_length, self._num_directions, self._batch_size, self._hidden_size), np.float32)
        Ht = np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
        Ct = np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
        CAll = np.zeros((self._sequence_length, self._num_directions, self._batch_size, self._hidden_size), np.float32)
        IOFC = np.zeros(
            (self._sequence_length, self._num_directions, self._batch_size, 4 * self._hidden_size), np.float32
        )
        for idx in range(self._batch_size):
            Hprev = H0[0, idx, :] if H0 is not None else np.zeros((self._hidden_size), np.float32)
            Cprev = C0[0, idx, :] if C0 is not None else np.zeros((self._hidden_size), np.float32)
            for t in range(self._sequence_length):
                Xt = X[t, idx, :]
                Wi = np.squeeze(W[:, : self._hidden_size, :], axis=0)
                Wo = np.squeeze(W[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                Wf = np.squeeze(W[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                Wc = np.squeeze(W[:, 3 * self._hidden_size :, :], axis=0)

                Ri = np.squeeze(R[:, : self._hidden_size, :], axis=0)
                Ro = np.squeeze(R[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                Rf = np.squeeze(R[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                Rc = np.squeeze(R[:, 3 * self._hidden_size :, :], axis=0)

                WBi = (
                    np.squeeze(B[:, : self._hidden_size], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                WBo = (
                    np.squeeze(B[:, self._hidden_size : 2 * self._hidden_size], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                WBf = (
                    np.squeeze(B[:, 2 * self._hidden_size : 3 * self._hidden_size], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                WBc = (
                    np.squeeze(B[:, 3 * self._hidden_size : 4 * self._hidden_size], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                RBi = (
                    np.squeeze(B[:, 4 * self._hidden_size : 5 * self._hidden_size], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                RBo = (
                    np.squeeze(B[:, 5 * self._hidden_size : 6 * self._hidden_size], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                RBf = (
                    np.squeeze(B[:, 6 * self._hidden_size : 7 * self._hidden_size], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                RBc = (
                    np.squeeze(B[:, 7 * self._hidden_size :], axis=0)
                    if B is not None
                    else np.zeros((self._hidden_size), np.float32)
                )

                Pi = (
                    np.squeeze(P[:, : self._hidden_size], axis=0)
                    if P is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                Po = (
                    np.squeeze(P[:, self._hidden_size : 2 * self._hidden_size], axis=0)
                    if P is not None
                    else np.zeros((self._hidden_size), np.float32)
                )
                Pf = (
                    np.squeeze(P[:, 2 * self._hidden_size : 3 * self._hidden_size], axis=0)
                    if P is not None
                    else np.zeros((self._hidden_size), np.float32)
                )

                it = np.dot(Xt, Wi.T) + np.dot(Hprev, Ri.T) + WBi + RBi + Pi * Cprev
                ft = np.dot(Xt, Wf.T) + np.dot(Hprev, Rf.T) + WBf + RBf + Pf * Cprev
                ct = np.dot(Xt, Wc.T) + np.dot(Hprev, Rc.T) + WBc + RBc

                it = sigmoid(it)
                ft = sigmoid(ft)
                ct = np.tanh(ct)

                Ct[0, idx, :] = ft * Cprev + it * ct

                ot = np.dot(Xt, Wo.T) + np.dot(Hprev, Ro.T) + WBo + RBo + Po * Ct[0, idx, :]
                ot = sigmoid(ot)

                IOFC[t, 0, idx, : self._hidden_size] = it
                IOFC[t, 0, idx, self._hidden_size : 2 * self._hidden_size] = ot
                IOFC[t, 0, idx, 2 * self._hidden_size : 3 * self._hidden_size] = ft
                IOFC[t, 0, idx, 3 * self._hidden_size :] = ct

                Ht[0, idx, :] = ot * np.tanh(Ct[0, idx, :])

                HAll[t, 0, idx, :] = Ht[0, idx, :]
                CAll[t, 0, idx, :] = Ct[0, idx, :]

                Hprev = Ht[0, idx, :]
                Cprev = Ct[0, idx, :]

        return HAll, Ht, Ct, CAll, IOFC

    def forward_ort(self, X, W, R, B=None, H0=None, C0=None, P=None):
        """Run LSTM forward pass using ONNX Runtime."""
        ort_inputs = {"X": X, "W": W, "R": R}
        if B is not None:
            ort_inputs["B"] = B
        if H0 is not None:
            ort_inputs["H0"] = H0
        if C0 is not None:
            ort_inputs["C0"] = C0
        if P is not None:
            ort_inputs["P"] = P

        ort_outs = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "lstm.onnx")
            onnx.save(self._forward_model, model_path)
            ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            ort_outs = ort_session.run(None, ort_inputs)

        return ort_outs

    def forward_graph(self, B=True, SL=False, H0=True, C0=True, P=False):
        """Create a graph for LSTM forward pass."""
        X = helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        W = helper.make_tensor_value_info(
            "W", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self.input_size]
        )
        R = helper.make_tensor_value_info(
            "R", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self._hidden_size]
        )
        B = (
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [self._num_directions, 8 * self._hidden_size])
            if B
            else None
        )
        SL = helper.make_tensor_value_info("SL", TensorProto.INT64, [self._batch_size]) if SL else None
        H0 = (
            helper.make_tensor_value_info(
                "H0", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if H0
            else None
        )
        C0 = (
            helper.make_tensor_value_info(
                "C0", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if C0
            else None
        )
        P = (
            helper.make_tensor_value_info("P", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size])
            if P
            else None
        )

        HAll = helper.make_tensor_value_info(
            "HAll",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        Ht = helper.make_tensor_value_info(
            "Ht", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
        )
        Ct = helper.make_tensor_value_info(
            "Ct", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
        )
        CAll = helper.make_tensor_value_info(
            "CAll",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        IOFC = helper.make_tensor_value_info(
            "IOFC",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, 4 * self._hidden_size],
        )

        lstm = helper.make_node(
            "LSTMTraining",
            inputs=[
                "X",
                "W",
                "R",
                "B" if B else "",
                "SL" if SL else "",
                "H0" if H0 else "",
                "C0" if C0 else "",
                "P" if P else "",
            ],
            outputs=["HAll", "Ht", "Ct", "CAll", "IOFC"],
            domain="com.microsoft",
            hidden_size=self._hidden_size,
        )

        graph = helper.make_graph(
            [lstm],
            "lstm",
            [gi for gi in [X, W, R, B, SL, H0, C0, P] if gi],
            [HAll, Ht, Ct, CAll, IOFC],
        )

        self._forward_model = helper.make_model(
            graph,
            producer_name="orttraining",
            opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)],
        )

        return self._forward_model

    def backward_np(self, X, W, R, B, H0, C0, P, HAll, CAll, IOFC, dHAll, dHt=None, dCt=None):
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
            X (np.ndarray): input tensor of shape (sequence_length, batch_size, input_size)
            W (np.ndarray): weight tensor of shape (num_directions, 4 * hidden_size, input_size)
            R (np.ndarray): recurrence weight tensor of shape (num_directions, 4 * hidden_size, hidden_size)
            B (np.ndarray): bias tensor of shape (num_directions, 8 * hidden_size)
            H0 (np.ndarray): initial hidden state tensor of shape (num_directions, batch_size, hidden_size)
            C0 (np.ndarray): initial cell state tensor of shape (num_directions, batch_size, hidden_size)
            P (np.ndarray): peephole weight tensor of shape (num_directions, 3 * hidden_size)
            HAll (np.ndarray): output tensor of shape (sequence_length, num_directions, batch_size, hidden_size)
            CAll (np.ndarray): cell state tensor of shape (sequence_length, num_directions, batch_size, hidden_size)
            IOFC (np.ndarray): input, output, forget, and cell gate tensor of shape (sequence_length, num_directions, batch_size, 4 * hidden_size)
            dHAll (np.ndarray): gradient of HAll
            dHt (np.ndarray): gradient of Ht
            dCt (np.ndarray): gradient of Ct

        Returns:
            tuple[np.ndarray]: gradients of X, W, R, B, H0, C0, P
        """
        dX = np.zeros((self._sequence_length, self._batch_size, self.input_size), np.float32)
        dW = np.zeros((self._num_directions, 4 * self._hidden_size, self.input_size), np.float32)
        dR = np.zeros((self._num_directions, 4 * self._hidden_size, self._hidden_size), np.float32)
        dB = np.zeros((self._num_directions, 8 * self._hidden_size), np.float32) if B is not None else None
        dH0 = (
            np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
            if H0 is not None
            else None
        )
        dC0 = (
            np.zeros((self._num_directions, self._batch_size, self._hidden_size), np.float32)
            if C0 is not None
            else None
        )
        dP = np.zeros((self._num_directions, 3 * self._hidden_size), np.float32) if P is not None else None

        for idx in range(self._batch_size):
            dH = dHt[0, idx, :] if dHt is not None else np.zeros((self._hidden_size), np.float32)
            dC = dCt[0, idx, :] if dCt is not None else np.zeros((self._hidden_size), np.float32)
            for t in reversed(range(self._sequence_length)):
                Xt = X[t, idx, :]
                Wi = np.squeeze(W[:, : self._hidden_size, :], axis=0)
                Wo = np.squeeze(W[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                Wf = np.squeeze(W[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                Wc = np.squeeze(W[:, 3 * self._hidden_size :, :], axis=0)

                Ri = np.squeeze(R[:, : self._hidden_size, :], axis=0)
                Ro = np.squeeze(R[:, self._hidden_size : 2 * self._hidden_size, :], axis=0)
                Rf = np.squeeze(R[:, 2 * self._hidden_size : 3 * self._hidden_size, :], axis=0)
                Rc = np.squeeze(R[:, 3 * self._hidden_size :, :], axis=0)

                it = IOFC[t, 0, idx, : self._hidden_size]
                ot = IOFC[t, 0, idx, self._hidden_size : 2 * self._hidden_size]
                ft = IOFC[t, 0, idx, 2 * self._hidden_size : 3 * self._hidden_size]
                ct = IOFC[t, 0, idx, 3 * self._hidden_size :]

                dH += dHAll[t, 0, idx, :]
                dCt2 = dH * ot * (1 - np.tanh(CAll[t, 0, idx, :]) ** 2)
                dC += dCt2
                dCtminus1 = dC * ft
                if t == 0 and dC0 is not None:
                    dC0[0, idx, :] = dCtminus1

                dit = dC * ct
                dot = dH * np.tanh(CAll[t, 0, idx, :])
                dft = dC * (CAll[t - 1, 0, idx, :] if t > 0 else C0[0, idx, :])
                dct = dC * it

                dai = dit * it * (1 - it)
                dao = dot * ot * (1 - ot)
                daf = dft * ft * (1 - ft)
                dac = dct * (1 - ct**2)

                dX[t, idx, :] = np.dot(dai, Wi) + np.dot(dao, Wo) + np.dot(daf, Wf) + np.dot(dac, Wc)
                dH = np.dot(dai, Ri) + np.dot(dao, Ro) + np.dot(daf, Rf) + np.dot(dac, Rc)
                if t == 0 and dH0 is not None:
                    dH0[0, idx, :] = dH

                dW[0, : self._hidden_size, :] += np.dot(np.expand_dims(dai, axis=0).T, np.expand_dims(Xt, axis=0))
                dW[0, self._hidden_size : 2 * self._hidden_size, :] += np.dot(
                    np.expand_dims(dao, axis=0).T, np.expand_dims(Xt, axis=0)
                )
                dW[0, 2 * self._hidden_size : 3 * self._hidden_size, :] += np.dot(
                    np.expand_dims(daf, axis=0).T, np.expand_dims(Xt, axis=0)
                )
                dW[0, 3 * self._hidden_size :, :] += np.dot(np.expand_dims(dac, axis=0).T, np.expand_dims(Xt, axis=0))

                Htminus1 = HAll[t - 1, 0, idx, :] if t > 0 else H0[0, idx, :]
                dR[0, : self._hidden_size, :] += np.dot(np.expand_dims(dai, axis=0).T, np.expand_dims(Htminus1, axis=0))
                dR[0, self._hidden_size : 2 * self._hidden_size, :] += np.dot(
                    np.expand_dims(dao, axis=0).T, np.expand_dims(Htminus1, axis=0)
                )
                dR[0, 2 * self._hidden_size : 3 * self._hidden_size, :] += np.dot(
                    np.expand_dims(daf, axis=0).T, np.expand_dims(Htminus1, axis=0)
                )
                dR[0, 3 * self._hidden_size :, :] += np.dot(
                    np.expand_dims(dac, axis=0).T, np.expand_dims(Htminus1, axis=0)
                )

                if dB is not None:
                    dB[0, : self._hidden_size] += dai
                    dB[0, self._hidden_size : 2 * self._hidden_size] += dao
                    dB[0, 2 * self._hidden_size : 3 * self._hidden_size] += daf
                    dB[0, 3 * self._hidden_size : 4 * self._hidden_size] += dac
                    dB[0, 4 * self._hidden_size : 5 * self._hidden_size] += dai
                    dB[0, 5 * self._hidden_size : 6 * self._hidden_size] += dao
                    dB[0, 6 * self._hidden_size : 7 * self._hidden_size] += daf
                    dB[0, 7 * self._hidden_size :] += dac

                if dP is not None:
                    dP[0, : self._hidden_size] += dai * (CAll[t - 1, 0, idx, :] if t > 0 else C0[0, idx, :])
                    dP[0, self._hidden_size : 2 * self._hidden_size] += dao * CAll[t, 0, idx, :]
                    dP[0, 2 * self._hidden_size : 3 * self._hidden_size] += daf * (
                        CAll[t - 1, 0, idx, :] if t > 0 else C0[0, idx, :]
                    )

                dC = dCtminus1

        return tuple([out for out in [dX, dW, dR, dB, dH0, dC0, dP] if out is not None])

    def backward_graph(self, B=True, SL=False, H0=True, C0=True, P=False, Ht=False, Ct=False):
        """Generate the ONNX graph for the backward pass of the LSTM operator."""
        X = helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        W = helper.make_tensor_value_info(
            "W", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self.input_size]
        )
        R = helper.make_tensor_value_info(
            "R", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self._hidden_size]
        )
        B = (
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [self._num_directions, 8 * self._hidden_size])
            if B
            else None
        )
        SL = helper.make_tensor_value_info("SL", TensorProto.INT64, [self._batch_size]) if SL else None
        H0 = (
            helper.make_tensor_value_info(
                "H0", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if H0
            else None
        )
        C0 = (
            helper.make_tensor_value_info(
                "C0", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if C0
            else None
        )
        P = (
            helper.make_tensor_value_info("P", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size])
            if P
            else None
        )

        HAll = helper.make_tensor_value_info(
            "HAll",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        Ht = (
            helper.make_tensor_value_info(
                "Ht", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if Ht
            else None
        )
        Ct = (
            helper.make_tensor_value_info(
                "Ct", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if Ct
            else None
        )
        CAll = helper.make_tensor_value_info(
            "CAll",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        IOFC = helper.make_tensor_value_info(
            "IOFC",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, 4 * self._hidden_size],
        )
        dHAll = helper.make_tensor_value_info(
            "dHAll",
            TensorProto.FLOAT,
            [self._sequence_length, self._num_directions, self._batch_size, self._hidden_size],
        )
        dHt = (
            helper.make_tensor_value_info(
                "dHt", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if Ht
            else None
        )
        dCt = (
            helper.make_tensor_value_info(
                "dCt", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if Ct
            else None
        )

        dX = helper.make_tensor_value_info(
            "dX", TensorProto.FLOAT, [self._sequence_length, self._batch_size, self.input_size]
        )
        dW = helper.make_tensor_value_info(
            "dW", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self.input_size]
        )
        dR = helper.make_tensor_value_info(
            "dR", TensorProto.FLOAT, [self._num_directions, 4 * self._hidden_size, self._hidden_size]
        )
        dB = (
            helper.make_tensor_value_info("dB", TensorProto.FLOAT, [self._num_directions, 8 * self._hidden_size])
            if B
            else None
        )
        dH0 = (
            helper.make_tensor_value_info(
                "dH0", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if H0
            else None
        )
        dC0 = (
            helper.make_tensor_value_info(
                "dC0", TensorProto.FLOAT, [self._num_directions, self._batch_size, self._hidden_size]
            )
            if C0
            else None
        )
        dP = (
            helper.make_tensor_value_info("dP", TensorProto.FLOAT, [self._num_directions, 3 * self._hidden_size])
            if P
            else None
        )

        lstm = helper.make_node(
            "LSTMGrad",
            inputs=[
                "X",
                "W",
                "R",
                "B" if B else "",
                "SL" if SL else "",
                "H0" if H0 else "",
                "C0" if C0 else "",
                "P" if P else "",
                "HAll" if HAll else "",
                "CAll" if CAll else "",
                "IOFC" if IOFC else "",
                "dHAll" if dHAll else "",
                "dHt" if dHt else "",
                "dCt" if dCt else "",
            ],
            outputs=[
                "dX",
                "dW",
                "dR",
                "dB" if dB else "",
                "dH0" if dH0 else "",
                "dC0" if dC0 else "",
                "dP" if dP else "",
            ],
            domain="com.microsoft",
            hidden_size=self._hidden_size,
        )

        graph = helper.make_graph(
            [lstm],
            "lstm",
            [gi for gi in [X, W, R, B, SL, H0, C0, P, HAll, CAll, IOFC, dHAll, dHt, dCt] if gi],
            [go for go in [dX, dW, dR, dB, dH0, dC0, dP] if go],
        )

        self._backward_model = helper.make_model(
            graph,
            producer_name="orttraining",
            opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)],
        )

        return self._backward_model

    def backward_ort(
        self, X, W, R, B=None, H0=None, C0=None, P=None, HAll=None, CAll=None, IOFC=None, dHAll=None, dHt=None, dCt=None
    ):
        """Run LSTM backward using ONNX Runtime.

        Users must call backward_graph before calling this function.
        """
        ort_inputs = {"X": X, "W": W, "R": R, "HAll": HAll, "CAll": CAll, "IOFC": IOFC, "dHAll": dHAll}
        if B is not None:
            ort_inputs["B"] = B
        if H0 is not None:
            ort_inputs["H0"] = H0
        if C0 is not None:
            ort_inputs["C0"] = C0
        if P is not None:
            ort_inputs["P"] = P
        if dHt is not None:
            ort_inputs["dHt"] = dHt
        if dCt is not None:
            ort_inputs["dCt"] = dCt

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
    _ = lstm.forward_graph(B=True, SL=False, H0=True, C0=True, P=True)

    X = np.random.rand(sequence_length, batch_size, input_size).astype(np.float32)
    W = np.random.rand(num_directions, 4 * hidden_size, input_size).astype(np.float32)
    R = np.random.rand(num_directions, 4 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)
    H0 = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    C0 = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    P = np.random.rand(num_directions, 3 * hidden_size).astype(np.float32)

    outs_ort = lstm.forward_ort(X, W, R, B, H0, C0, P)
    outs_np = lstm.forward_np(X, W, R, B, H0, C0, P)

    for ort_out, np_out in zip(outs_ort, outs_np):
        assert np.allclose(ort_out, np_out, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize("sequence_length", [2, 4, 16, 32])
@pytest.mark.parametrize("batch_size", [2, 4, 32])
@pytest.mark.parametrize("input_size", [2, 4, 32])
@pytest.mark.parametrize("hidden_size", [2, 4, 32])
def test_lstm_backward(sequence_length, batch_size, input_size, hidden_size):
    num_directions = 1

    lstm = LSTM(sequence_length, batch_size, input_size, hidden_size)
    _ = lstm.backward_graph(B=True, SL=False, H0=True, C0=True, P=True, Ht=True, Ct=True)

    X = np.random.rand(sequence_length, batch_size, input_size).astype(np.float32)
    W = np.random.rand(num_directions, 4 * hidden_size, input_size).astype(np.float32)
    R = np.random.rand(num_directions, 4 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)
    H0 = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    C0 = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    P = np.random.rand(num_directions, 3 * hidden_size).astype(np.float32)

    HAll = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    CAll = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    IOFC = np.random.rand(sequence_length, num_directions, batch_size, 4 * hidden_size).astype(np.float32)
    dHAll = np.random.rand(sequence_length, num_directions, batch_size, hidden_size).astype(np.float32)
    dHt = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)
    dCt = np.random.rand(num_directions, batch_size, hidden_size).astype(np.float32)

    outs_ort = lstm.backward_ort(X, W, R, B, H0, C0, P, HAll, CAll, IOFC, dHAll, dHt, dCt)
    outs_np = lstm.backward_np(X, W, R, B, H0, C0, P, HAll, CAll, IOFC, dHAll, dHt, dCt)

    for ort_out, np_out in zip(outs_ort, outs_np):
        assert np.allclose(ort_out, np_out, rtol=1e-03, atol=1e-05)
