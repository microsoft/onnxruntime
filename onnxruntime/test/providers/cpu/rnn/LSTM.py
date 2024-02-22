# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from typing import Any, Tuple  # noqa: F401

import numpy as np  # type: ignore

# import onnx
# from ..base import Base
# from . import expect

DebugOutput = True
np.set_printoptions(suppress=True)  # , precision=16, floatmode='maxprec')


def print_with_shape(name, a, force_output=False):
    if force_output or DebugOutput:
        print(name + " [shape: ", a.shape, "]\n", a)


def print_results(Y, Y_h, Y_c):
    print("*************************")
    print_with_shape("Y", Y, True)
    print("---------")
    print_with_shape("Y_h", Y_h, True)
    print("---------")
    print_with_shape("Y_c", Y_c, True)
    print("*************************")


class LSTM_Helper:  # noqa: N801
    def __init__(self, **params):  # type: (*Any) -> None
        required_inputs = ["X", "W", "R"]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        X = params["X"]  # noqa: N806
        W = params["W"]  # noqa: N806
        R = params["R"]  # noqa: N806

        num_directions = W.shape[0]
        X.shape[0]
        batch_size = X.shape[1]
        hidden_size = R.shape[-1]

        B = (  # noqa: N806
            params["B"]
            if "B" in params
            else np.zeros(num_directions * 8 * hidden_size).reshape(num_directions, 8 * hidden_size)
        )
        P = (  # noqa: N806
            params["P"]
            if "P" in params
            else np.zeros(num_directions * 3 * hidden_size).reshape(num_directions, 3 * hidden_size)
        )
        h_0 = (
            params["initial_h"]
            if "initial_h" in params
            else np.zeros((num_directions, batch_size, hidden_size)).reshape(num_directions, batch_size, hidden_size)
        )
        c_0 = (
            params["initial_c"]
            if "initial_c" in params
            else np.zeros((num_directions, batch_size, hidden_size)).reshape(num_directions, batch_size, hidden_size)
        )

        f = params.get("f", ActivationFuncs.sigmoid)
        g = params.get("g", ActivationFuncs.tanh)
        h = params.get("h", ActivationFuncs.tanh)
        input_forget = params.get("input_forget", False)
        clip = params.get("clip", 9999.0)

        self.direction = params.get("direction", "forward")

        if num_directions == 1:
            if self.direction == "forward":
                self.one = OneDirectionLSTM(X, W, R, B, P, h_0, c_0, f, g, h, input_forget, clip)
            else:
                # flip input so we process in reverse
                self.one = OneDirectionLSTM(np.flip(X, 0), W, R, B, P, h_0, c_0, f, g, h, input_forget, clip)

            self.two = None

        else:
            # split the inputs which have per direction rows
            Wfw, Wbw = np.vsplit(W, 2)  # noqa: N806
            Rfw, Rbw = np.vsplit(R, 2)  # noqa: N806
            Bfw, Bbw = np.vsplit(B, 2)  # noqa: N806
            Pfw, Pbw = np.vsplit(P, 2)  # noqa: N806
            h_0fw, h_0bw = np.vsplit(h_0, 2)
            c_0fw, c_0bw = np.vsplit(c_0, 2)

            self.one = OneDirectionLSTM(X, Wfw, Rfw, Bfw, Pfw, h_0fw, c_0fw, f, g, h, input_forget, clip)
            self.two = OneDirectionLSTM(
                np.flip(X, 0),
                Wbw,
                Rbw,
                Bbw,
                Pfw,
                h_0bw,
                c_0fw,
                f,
                g,
                h,
                input_forget,
                clip,
            )

    def run(self):
        if self.direction == "bidirectional":
            f_output, f_Y_h, f_Y_c = self.one.execute()  # noqa: N806
            r_output, r_Y_h, r_Y_c = self.two.execute()  # noqa: N806

            # flip reverse output it matches the original input order
            r_output_orig_input_order = np.flip(r_output, 0)

            # create merged output by merging the forward and reverse rows for seq_length
            # 0 rows, 2 directions, batch size, hidden_size
            seq_length = f_output.shape[0]
            batch_size = f_output.shape[2]
            hidden_size = f_output.shape[3]

            output = np.empty((0, 2, batch_size, hidden_size), np.float32)
            # Y_h = np.empty((0, 2, batch_size, hidden_size), np.float32)
            # Y_c = np.empty((0, 2, hidden_size, hidden_size), np.float32)
            for x in range(0, seq_length):
                output = np.append(output, f_output[x])
                output = np.append(output, r_output_orig_input_order[x])

            output = output.reshape(seq_length, 2, batch_size, hidden_size)

            Y_h = np.append(f_Y_h, r_Y_h)  # noqa: N806
            Y_c = np.append(f_Y_c, r_Y_c)  # noqa: N806

        else:
            output, Y_h, Y_c = self.one.execute()  # noqa: N806
            if self.direction == "reverse":
                # flip so it's back in the original order of the inputs
                output = np.flip(output, 0)

        return output, Y_h, Y_c


class ActivationFuncs:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)


class OneDirectionLSTM:
    def __init__(
        self,
        X,
        W,
        R,
        B,
        P,
        initial_h,
        initial_c,
        f=ActivationFuncs.sigmoid,
        g=ActivationFuncs.tanh,
        h=ActivationFuncs.tanh,
        input_forget=False,
        clip=9999.0,
    ):
        self.X = X
        # remove num_directions axis for W, R, B, P, H_0, C_0
        self.W = np.squeeze(W, axis=0)
        self.R = np.squeeze(R, axis=0)
        self.B = np.squeeze(B, axis=0)
        self.P = np.squeeze(P, axis=0)
        self.h_0 = np.squeeze(initial_h, axis=0)
        self.c_0 = np.squeeze(initial_c, axis=0)

        print_with_shape("X", self.X)
        print_with_shape("W", self.W)
        print_with_shape("R", self.R)
        print_with_shape("B", self.B)
        print_with_shape("P", self.P)
        print_with_shape("h_0", self.h_0)
        print_with_shape("c_0", self.c_0)

        self.f = f
        self.g = g
        self.h = h
        self.input_forget = input_forget
        self.clip = clip

    def execute(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        [p_i, p_o, p_f] = np.split(self.P, 3)
        h_list = []

        H_t = self.h_0  # noqa: N806
        C_t = self.c_0  # noqa: N806

        for x in np.split(self.X, self.X.shape[0], axis=0):
            print_with_shape("Xt1", x)

            # gates = np.dot(x, np.transpose(self.W)) + np.dot(H_t, np.transpose(self.R)) + np.add(*np.split(self.B, 2))

            print_with_shape("W^T", np.transpose(self.W))
            # t0 == t-1, t1 == current
            Xt1_W = np.dot(x, np.transpose(self.W))  # noqa: N806
            print_with_shape("Xt1_W^T", Xt1_W)
            Ht0_R = np.dot(H_t, np.transpose(self.R))  # noqa: N806
            print_with_shape("Ht-1*R", Ht0_R)
            WbRb = np.add(*np.split(self.B, 2))  # noqa: N806
            print_with_shape("Wb + Rb", WbRb)
            gates = Xt1_W + Ht0_R + WbRb

            # input to it, ft, ct, ot
            it_in, ot_in, ft_in, ct_in = np.split(gates, 4, -1)
            print_with_shape("it_in", it_in)
            print_with_shape("ot_in", ot_in)
            print_with_shape("ft_in", ft_in)
            print_with_shape("ct_in", ct_in)

            i = self.f(np.clip((it_in + p_i * C_t), -self.clip, self.clip))
            if self.input_forget:
                f = 1.0 - i  # this is what ONNXRuntime does
            else:
                f = self.f(np.clip((ft_in + p_f * C_t), -self.clip, self.clip))
            c = self.g(np.clip(ct_in, -self.clip, self.clip))
            C = f * C_t + i * c  # noqa: N806
            o = self.f(np.clip((ot_in + p_o * C), -self.clip, self.clip))
            H = o * self.h(C)  # noqa: N806
            h_list.append(H)
            H_t = H  # noqa: N806
            C_t = C  # noqa: N806

            print_with_shape("i", i)
            print_with_shape("f", f)
            print_with_shape("c", c)
            print_with_shape("o", o)
            print_with_shape("C", C)
            print_with_shape("H", i)

        concatenated = np.concatenate(h_list)
        output = np.expand_dims(concatenated, 1)
        return output, h_list[-1], C


class LSTM:  # Base):
    @staticmethod
    def SimpleWeightsNoBiasTwoRows(direction):  # type: () -> None  # noqa: N802
        print(LSTM.SimpleWeightsNoBiasTwoRows.__name__ + " direction=" + direction)

        input_size = 1
        hidden_size = 3
        number_of_gates = 4

        input = np.array([[[1.0], [2.0]], [[10.0], [11.0]]]).astype(np.float32)

        W = (  # noqa: N806
            np.array([0.1, 0.2, 0.3, 0.4, 1, 2, 3, 4, 10, 11, 12, 13])
            .astype(np.float32)
            .reshape(1, number_of_gates * hidden_size, input_size)
        )

        weight_scale = 0.1
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        if direction == "bidirectional":
            W = np.tile(W, (2, 1)).reshape(2, number_of_gates * hidden_size, input_size)  # noqa: N806
            R = np.tile(R, (2, 1)).reshape(2, number_of_gates * hidden_size, hidden_size)  # noqa: N806

        lstm = LSTM_Helper(X=input, W=W, R=R, direction=direction)

        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        # expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_defaults')
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def LargeBatchWithClip(clip):  # noqa: N802
        print(LSTM.LargeBatchWithClip.__name__ + " clip=" + str(clip))

        seq_length = 2
        batch_size = 32
        input_size = 1
        hidden_size = 3
        number_of_gates = 4

        # sequentialvalues from 1 to 32
        input = (
            np.array(range(1, seq_length * batch_size + 1, 1))
            .astype(np.float32)
            .reshape(seq_length, batch_size, input_size)
        )

        W = (  # noqa: N806
            np.array([0.1, 0.2, 0.3, 0.4, 1, 2, 3, 4, 10, 11, 12, 13])
            .astype(np.float32)
            .reshape(1, number_of_gates * hidden_size, input_size)
        )

        weight_scale = 0.1
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        lstm = LSTM_Helper(X=input, W=W, R=R, clip=clip)

        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def BatchParallelFalseSeqLengthGreaterThanOne():  # noqa: N802
        print(LSTM.BatchParallelFalseSeqLengthGreaterThanOne.__name__)

        seq_length = 2
        batch_size = 1
        input_size = 1
        hidden_size = 2
        number_of_gates = 4

        input = np.array([1, 2]).astype(np.float32).reshape(seq_length, batch_size, input_size)

        W = (  # noqa: N806
            np.array([0.1, 0.2, 0.3, 0.4, 1, 2, 3, 4])
            .astype(np.float32)
            .reshape(1, number_of_gates * hidden_size, input_size)
        )

        weight_scale = 0.1
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        lstm = LSTM_Helper(X=input, W=W, R=R)

        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def export_initial_bias():  # type: () -> None
        print(LSTM.export_initial_bias.__name__)

        input = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(np.float32)

        input_size = 3
        hidden_size = 4
        weight_scale = 0.1
        custom_bias = 0.1
        number_of_gates = 4

        # node = onnx.helper.make_node(
        #     'LSTM',
        #     inputs=['X', 'W', 'R', 'B'],
        #     outputs=['', 'Y'],
        #     hidden_size=hidden_size
        # )

        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)  # noqa: N806
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        # Adding custom bias
        W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)  # noqa: N806
        R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)  # noqa: N806
        B = np.concatenate((W_B, R_B), 1)  # noqa: N806

        lstm = LSTM_Helper(X=input, W=W, R=R, B=B)
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)
        # expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_lstm_with_initial_bias')

    @staticmethod
    def export_peepholes():  # type: () -> None
        input = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]).astype(np.float32)

        input_size = 4
        hidden_size = 3
        weight_scale = 0.1
        number_of_gates = 4
        number_of_peepholes = 3

        # node = onnx.helper.make_node(
        #     'LSTM',
        #     inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
        #     outputs=['', 'Y'],
        #     hidden_size=hidden_size
        # )

        # Initializing Inputs
        W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)  # noqa: N806
        R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806
        B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)  # noqa: N806
        np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
        init_h = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
        init_c = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
        P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(np.float32)  # noqa: N806

        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)
        # expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[Y_h.astype(np.float32)],
        #        name='test_lstm_with_peepholes')


class ONNXRuntimeTestContext:
    hidden_size = 2
    input_size = 2

    @staticmethod
    def OneDirectionWeights():  # noqa: N802
        num_directions = 1
        hidden_size = ONNXRuntimeTestContext.hidden_size
        input_size = ONNXRuntimeTestContext.input_size

        W = (  # noqa: N806
            np.array(
                [
                    -0.494659,
                    0.0453352,
                    -0.487793,
                    0.417264,
                    -0.0175329,
                    0.489074,
                    -0.446013,
                    0.414029,
                    -0.0091708,
                    -0.255364,
                    -0.106952,
                    -0.266717,
                    -0.0888852,
                    -0.428709,
                    -0.283349,
                    0.208792,
                ]
            )
            .reshape(num_directions, 4 * hidden_size, input_size)
            .astype(np.float32)
        )

        R = (  # noqa: N806
            np.array(
                [
                    0.146626,
                    -0.0620289,
                    -0.0815302,
                    0.100482,
                    -0.219535,
                    -0.306635,
                    -0.28515,
                    -0.314112,
                    -0.228172,
                    0.405972,
                    0.31576,
                    0.281487,
                    -0.394864,
                    0.42111,
                    -0.386624,
                    -0.390225,
                ]
            )
            .reshape(num_directions, 4 * hidden_size, hidden_size)
            .astype(np.float32)
        )

        P = (  # noqa: N806
            np.array([0.2345, 0.5235, 0.4378, 0.3475, 0.8927, 0.3456])
            .reshape(num_directions, 3 * hidden_size)
            .astype(np.float32)
        )

        # // [8*hidden]
        B = (  # noqa: N806
            np.array(
                [
                    0.381619,
                    0.0323954,
                    -0.14449,
                    0.420804,
                    -0.258721,
                    0.45056,
                    -0.250755,
                    0.0967895,
                    # peephole bias
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            .reshape(num_directions, 8 * hidden_size)
            .astype(np.float32)
        )

        return W, R, B, P

    @staticmethod
    def BidirectionalWeights():  # noqa: N802
        hidden_size = ONNXRuntimeTestContext.hidden_size
        input_size = ONNXRuntimeTestContext.input_size

        W1, R1, B1, P1 = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806

        W = np.tile(W1, (2, 1)).reshape(2, 4 * hidden_size, input_size)  # noqa: N806
        R = np.tile(R1, (2, 1)).reshape(2, 4 * hidden_size, hidden_size)  # noqa: N806
        B = np.tile(B1, (2, 1))  # noqa: N806
        P = np.tile(P1, (2, 1))  # noqa: N806

        return W, R, B, P

    @staticmethod
    def DefaultInput():  # noqa: N802
        seq_length = 2
        batch_size = 1
        input_size = 2

        input = (
            np.array([-0.455351, -0.276391, -0.185934, -0.269585])
            .reshape(seq_length, batch_size, input_size)
            .astype(np.float32)
        )

        return input


class ONNXRuntimeUnitTests:
    @staticmethod
    def ONNXRuntime_TestLSTMBidirectionalBasic():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMBidirectionalBasic.__name__)

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.BidirectionalWeights()  # noqa: N806
        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, direction="bidirectional")
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMForwardNoBiasUsePeepholes():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardNoBiasUsePeepholes.__name__)
        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        lstm = LSTM_Helper(X=input, W=W, R=R, P=P)  # no bias
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMForwardInputForget():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardInputForget.__name__)

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, input_forget=True)
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMForwardClip():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardClip.__name__)

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, clip=0.1)
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMBackward():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMBackward.__name__)

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, direction="reverse")
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMForwardHiddenState():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardHiddenState.__name__)

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        initial_h = np.array([0.34, 0.72]).reshape(1, 1, 2).astype(np.float32)
        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, initial_h=initial_h)
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMForwardCellState():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardCellState.__name__)

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        initial_h = np.array([0.34, 0.72]).reshape(1, 1, 2).astype(np.float32)
        initial_c = np.array([0.63, 0.21]).reshape(1, 1, 2).astype(np.float32)
        lstm = LSTM_Helper(X=input, W=W, R=R, B=B, initial_h=initial_h, initial_c=initial_c)
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMActivation():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMActivation.__name__)

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        lstm = LSTM_Helper(
            X=input,
            W=W,
            R=R,
            B=B,
            f=ActivationFuncs.tanh,
            g=ActivationFuncs.sigmoid,
            h=ActivationFuncs.tanh,
        )
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMBatchReallocation():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMBatchReallocation.__name__)
        seq_length = 2
        batch_size = 1
        input_size = 2

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        lstm = LSTM_Helper(
            X=input,
            W=W,
            R=R,
            B=B,
            f=ActivationFuncs.tanh,
            g=ActivationFuncs.sigmoid,
            h=ActivationFuncs.tanh,
        )
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)
        print("===============")

        batch_size = 3
        input = (
            np.array(
                [
                    -0.455351,
                    -0.476391,
                    -0.555351,
                    -0.376391,
                    -0.655351,
                    -0.276391,
                    -0.185934,
                    -0.869585,
                    -0.285934,
                    -0.769585,
                    -0.385934,
                    -0.669585,
                ]
            )
            .reshape(seq_length, batch_size, input_size)
            .astype(np.float32)
        )

        W, R, B, P = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        lstm = LSTM_Helper(
            X=input,
            W=W,
            R=R,
            B=B,
            f=ActivationFuncs.tanh,
            g=ActivationFuncs.sigmoid,
            h=ActivationFuncs.tanh,
        )
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

    @staticmethod
    def ONNXRuntime_TestLSTMOutputWrite():  # noqa: N802
        print(ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMOutputWrite.__name__)
        seq_length = 2
        batch_size = 1
        input_size = 2

        input = ONNXRuntimeTestContext.DefaultInput()
        W, R, B, P = ONNXRuntimeTestContext.BidirectionalWeights()  # noqa: N806
        lstm = LSTM_Helper(
            X=input,
            W=W,
            R=R,
            B=B,
            direction="bidirectional",
            f=ActivationFuncs.tanh,
            g=ActivationFuncs.sigmoid,
            h=ActivationFuncs.tanh,
        )
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)

        print("===============")

        batch_size = 3
        input = (
            np.array(
                [
                    -0.455351,
                    -0.776391,
                    -0.355351,
                    -0.576391,
                    -0.255351,
                    -0.376391,
                    -0.185934,
                    -0.169585,
                    -0.285934,
                    -0.469585,
                    -0.385934,
                    -0.669585,
                ]
            )
            .reshape(seq_length, batch_size, input_size)
            .astype(np.float32)
        )

        W, R, B, P = ONNXRuntimeTestContext.BidirectionalWeights()  # noqa: N806
        lstm = LSTM_Helper(
            X=input,
            W=W,
            R=R,
            B=B,
            direction="bidirectional",
            f=ActivationFuncs.tanh,
            g=ActivationFuncs.sigmoid,
            h=ActivationFuncs.tanh,
        )
        Y, Y_h, Y_c = lstm.run()  # noqa: N806
        print_results(Y, Y_h, Y_c)


DebugOutput = False
LSTM.SimpleWeightsNoBiasTwoRows("forward")
LSTM.SimpleWeightsNoBiasTwoRows("reverse")
LSTM.SimpleWeightsNoBiasTwoRows("bidirectional")
LSTM.LargeBatchWithClip(99999.0)  # too large to affect output
LSTM.LargeBatchWithClip(4.0)
LSTM.BatchParallelFalseSeqLengthGreaterThanOne()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMBidirectionalBasic()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardNoBiasUsePeepholes()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardInputForget()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardClip()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMBackward()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardHiddenState()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMForwardCellState()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMActivation()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMBatchReallocation()
ONNXRuntimeUnitTests.ONNXRuntime_TestLSTMOutputWrite()
