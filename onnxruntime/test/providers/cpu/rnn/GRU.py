# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

DebugOutput = False
np.set_printoptions(suppress=True)  # , precision=16, floatmode='maxprec')


def print_with_shape(name, a, force_output=False):
    if force_output or DebugOutput:
        print(name + " [shape: ", a.shape, "]\n", a)


def print_results(Y):
    print("*************************")
    print_with_shape("Y", Y, True)
    print("*************************")


class GRU_Helper:  # noqa: N801
    def __init__(self, **params):
        # Match the ONNXRuntime/CNTK behavior
        # If False use the python from the ONNX spec
        self.match_onnxruntime = True

        required_inputs = ["X", "W", "R"]
        for i in required_inputs:
            assert i in params, f"Missing Required Input: {i}"

        num_directions = params["W"].shape[0]
        params["X"].shape[0]

        hidden_size = params["R"].shape[-1]
        batch_size = params["X"].shape[1]

        X = params["X"]  # noqa: N806
        W = params["W"]  # noqa: N806
        R = params["R"]  # noqa: N806
        B = (  # noqa: N806
            params["B"]
            if "B" in params
            else np.zeros(num_directions * 6 * hidden_size).reshape(num_directions, 6 * hidden_size)
        )
        H_0 = (  # noqa: N806
            params["initial_h"]
            if "initial_h" in params
            else np.zeros((num_directions, batch_size, hidden_size)).reshape(num_directions, batch_size, hidden_size)
        )
        LBR = params.get("linear_before_reset", 0)  # noqa: N806
        self.direction = params.get("direction", "forward")

        if num_directions == 1:
            if self.direction == "forward":
                self.one = OneDirectionGRU(X, W, R, B, H_0, LBR)
            else:
                # flip input so we process in reverse
                self.one = OneDirectionGRU(np.flip(X, 0), W, R, B, H_0, LBR)

            self.two = None

        else:
            # split the inputs which have per direction rows
            Wfw, Wbw = np.vsplit(W, 2)  # noqa: N806
            Rfw, Rbw = np.vsplit(R, 2)  # noqa: N806
            Bfw, Bbw = np.vsplit(B, 2)  # noqa: N806
            H_0fw, H_0bw = np.vsplit(H_0, 2)  # noqa: N806

            self.one = OneDirectionGRU(X, Wfw, Rfw, Bfw, H_0fw, LBR)
            self.two = OneDirectionGRU(np.flip(X, 0), Wbw, Rbw, Bbw, H_0bw, LBR)

    def run(self):
        if self.direction == "bidirectional":
            f_output = self.one.execute()
            r_output = self.two.execute()

            # flip reverse output it matches the original input order
            r_output_orig_input_order = np.flip(r_output, 0)

            # create merged output by merging the forward and reverse rows for seq_length
            # 0 rows, 2 directions, batch size, hidden_size
            seq_length = f_output.shape[0]
            batch_size = f_output.shape[2]
            hidden_size = f_output.shape[3]

            output = np.empty((0, 2, batch_size, hidden_size), np.float32)
            for x in range(0, seq_length):
                output = np.append(output, f_output[x])
                output = np.append(output, r_output_orig_input_order[x])

            output = output.reshape(seq_length, 2, batch_size, hidden_size)
        else:
            output = self.one.execute()
            if self.direction == "reverse":
                # flip so it's back in the original order of the inputs
                output = np.flip(output, 0)

        return output


class OneDirectionGRU:
    def __init__(self, X, W, R, B, initial_h, LBR):
        self.X = X
        # remove num_directions axis for W, R, B, H_0
        self.W = np.squeeze(W, axis=0)
        self.R = np.squeeze(R, axis=0)
        self.B = np.squeeze(B, axis=0)
        self.H_0 = np.squeeze(initial_h, axis=0)
        self.LBR = LBR

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, x):
        return np.tanh(x)

    def execute(self):
        print_with_shape("X", self.X)

        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)

        # print_with_shape("w_z", w_z)
        # print_with_shape("w_r", w_r)
        # print_with_shape("w_h", w_h)

        # print_with_shape("r_z", r_z)
        # print_with_shape("r_r", r_r)
        # print_with_shape("r_h", r_h)

        # print_with_shape("w_bz", w_bz)
        # print_with_shape("w_br", w_br)
        # print_with_shape("w_bh", w_bh)
        # print_with_shape("r_bz", r_bz)
        # print_with_shape("r_br", r_br)
        # print_with_shape("r_bh", r_bh)

        self.X.shape[0]
        num_directions = 1
        hidden_size = self.R.shape[-1]
        batch_size = self.X.shape[1]

        output = np.empty((0, num_directions, batch_size, hidden_size), np.float32)

        for row in self.X:
            z = self.f(np.dot(row, np.transpose(w_z)) + np.dot(self.H_0, np.transpose(r_z)) + w_bz + r_bz)
            r = self.f(np.dot(row, np.transpose(w_r)) + np.dot(self.H_0, np.transpose(r_r)) + w_br + r_br)
            h_default = self.g(np.dot(row, np.transpose(w_h)) + np.dot(r * self.H_0, np.transpose(r_h)) + w_bh + r_bh)
            h_linear = self.g(np.dot(row, np.transpose(w_h)) + r * (np.dot(self.H_0, np.transpose(r_h)) + r_bh) + w_bh)

            h = h_linear if self.LBR else h_default

            print_with_shape("z", z)
            print_with_shape("r", r)
            print_with_shape("h", h)

            H = (1 - z) * h + z * self.H_0  # noqa: N806

            print_with_shape("H", H)
            output = np.append(output, H.reshape(1, 1, batch_size, hidden_size), axis=0)

            self.H_0 = H

        return output


class ONNXRuntimeTestContext:
    @staticmethod
    def OneDirectionWeights():  # noqa: N802
        hidden_size = 2

        W = np.array(  # noqa: N806
            [
                [
                    [-0.494659, 0.0453352],  # Wz
                    [-0.487793, 0.417264],
                    [-0.0091708, -0.255364],  # Wr
                    [-0.106952, -0.266717],
                    [-0.0888852, -0.428709],  # Wh
                    [-0.283349, 0.208792],
                ]
            ]
        ).astype(np.float32)

        R = np.array(  # noqa: N806
            [
                [
                    [0.146626, -0.0620289],  # Rz
                    [-0.0815302, 0.100482],
                    [-0.228172, 0.405972],  # Rr
                    [0.31576, 0.281487],
                    [-0.394864, 0.42111],  # Rh
                    [-0.386624, -0.390225],
                ]
            ]
        ).astype(np.float32)

        W_B = np.array(  # noqa: N806
            [
                [
                    0.381619,
                    0.0323954,
                    -0.258721,
                    0.45056,
                    -0.250755,
                    0.0967895,
                ]
            ]
        ).astype(  # Wbz  # Wbr
            np.float32
        )  # Wbh
        R_B = np.zeros((1, 3 * hidden_size)).astype(np.float32)  # noqa: N806
        B = np.concatenate((W_B, R_B), axis=1)  # noqa: N806

        return W, R, B

    @staticmethod
    def BidirectionalWeights():  # noqa: N802
        W1, R1, B1 = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806

        hidden_size = R1.shape[-1]
        input_size = W1.shape[-1]

        W = np.tile(W1, (2, 1)).reshape(2, 3 * hidden_size, input_size)  # noqa: N806
        R = np.tile(R1, (2, 1)).reshape(2, 3 * hidden_size, hidden_size)  # noqa: N806
        B = np.tile(B1, (2, 1))  # noqa: N806

        return W, R, B


# replicate ONNXRuntime unit tests inputs to validate output
class GRU_ONNXRuntimeUnitTests:  # noqa: N801
    @staticmethod
    def ForwardDefaultActivationsSimpleWeightsNoBiasTwoRows():  # noqa: N802
        print(GRU_ONNXRuntimeUnitTests.ForwardDefaultActivationsSimpleWeightsNoBiasTwoRows.__name__)

        seq_length = 2
        batch_size = 2
        input_size = 1
        hidden_size = 3
        input = np.array([1.0, 2.0, 10.0, 11.0]).astype(np.float32).reshape(seq_length, batch_size, input_size)

        W = (  # noqa: N806
            np.array([0.1, 0.2, 0.3, 1, 2, 3, 10, 11, 12]).astype(np.float32).reshape(1, 3 * hidden_size, input_size)
        )

        weight_scale = 0.1
        R = weight_scale * np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        gru = GRU_Helper(X=input, W=W, R=R, direction="forward")
        fw_output = gru.run()
        print_results(fw_output)

    @staticmethod
    def ReverseDefaultActivationsSimpleWeightsNoBiasTwoRows():  # noqa: N802
        print(GRU_ONNXRuntimeUnitTests.ReverseDefaultActivationsSimpleWeightsNoBiasTwoRows.__name__)

        input_size = 1
        hidden_size = 3
        input = np.array([[[1.0], [2.0]], [[10.0], [11.0]]]).astype(np.float32)

        W = (  # noqa: N806
            np.array([0.1, 0.2, 0.3, 1, 2, 3, 10, 11, 12]).astype(np.float32).reshape(1, 3 * hidden_size, input_size)
        )

        weight_scale = 0.1
        R = weight_scale * np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        gru = GRU_Helper(X=input, W=W, R=R, direction="reverse")
        fw_output = gru.run()
        print_results(fw_output)

    @staticmethod
    def BidirectionalDefaultActivationsSimpleWeightsNoBias(linear_before_reset=0):  # noqa: N802
        print(
            GRU_ONNXRuntimeUnitTests.BidirectionalDefaultActivationsSimpleWeightsNoBias.__name__
            + ".linear_before_reset="
            + str(linear_before_reset)
        )

        input_size = 1
        hidden_size = 3

        if linear_before_reset:
            input = np.array([[[1.0], [2.0], [3.0]], [[10.0], [11.0], [12.0]]]).astype(np.float32)
        else:
            input = np.array([[[1.0], [2.0]], [[10.0], [11.0]]]).astype(np.float32)

        W = (  # noqa: N806
            np.array([0.1, 0.2, 0.3, 1, 2, 3, 10, 11, 12]).astype(np.float32).reshape(1, 3 * hidden_size, input_size)
        )

        weight_scale = 0.1
        R = weight_scale * np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        # duplicate the W and R inputs so we use the same values for both forward and reverse
        gru = GRU_Helper(
            X=input,
            W=np.tile(W, (2, 1)).reshape(2, 3 * hidden_size, input_size),
            R=np.tile(R, (2, 1)).reshape(2, 3 * hidden_size, hidden_size),
            direction="bidirectional",
            linear_before_reset=linear_before_reset,
        )

        fw_output = gru.run()
        print_results(fw_output)

    @staticmethod
    def DefaultActivationsSimpleWeightsWithBias(rows=2, direction="forward", linear_before_reset=0):  # noqa: N802
        print(
            GRU_ONNXRuntimeUnitTests.DefaultActivationsSimpleWeightsWithBias.__name__
            + " batch_parallel="
            + str(rows != 1)
            + " direction="
            + direction
            + " linear_before_reset="
            + str(linear_before_reset)
        )

        seq_length = 2
        batch_size = rows
        input_size = 1
        hidden_size = 3

        if batch_size == 1:
            input = [-0.1, -0.3]
        else:
            input = [-0.1, 0.2, -0.3, 0.4]

        input = np.array(input).astype(np.float32).reshape(seq_length, batch_size, input_size)

        W = (  # noqa: N806
            np.array([0.1, 0.2, 0.3, 0.2, 0.3, 0.1, 0.3, 0.1, 0.2])
            .astype(np.float32)
            .reshape(1, 3 * hidden_size, input_size)
        )

        weight_scale = 0.1
        R = weight_scale * np.ones((1, 3 * hidden_size, hidden_size)).astype(np.float32)  # noqa: N806

        # Wb[zrh] Rb[zrh]
        B = (  # noqa: N806
            np.array(
                [
                    -0.01,
                    0.1,
                    0.01,
                    -0.2,
                    -0.02,
                    0.02,
                    0.3,
                    -0.3,
                    -0.3,
                    -0.03,
                    0.5,
                    -0.7,
                    0.05,
                    -0.7,
                    0.3,
                    0.07,
                    -0.03,
                    0.5,
                ]
            )
            .astype(np.float32)
            .reshape(1, 6 * hidden_size)
        )

        gru = GRU_Helper(
            X=input,
            W=W,
            R=R,
            B=B,
            direction=direction,
            linear_before_reset=linear_before_reset,
        )
        fw_output = gru.run()
        print_results(fw_output)

    @staticmethod
    def ForwardDefaultActivationsSimpleWeightsWithBiasBatchParallel():  # noqa: N802
        GRU_ONNXRuntimeUnitTests.DefaultActivationsSimpleWeightsWithBias()

    @staticmethod
    def ForwardDefaultActivationsSimpleWeightsWithBiasBatchParallelLinearBeforeReset():  # noqa: N802
        GRU_ONNXRuntimeUnitTests.DefaultActivationsSimpleWeightsWithBias(linear_before_reset=1)

    @staticmethod
    def ReverseDefaultActivationsSimpleWeightsWithBiasBatchParallelLinearBeforeReset():  # noqa: N802
        GRU_ONNXRuntimeUnitTests.DefaultActivationsSimpleWeightsWithBias(direction="reverse", linear_before_reset=1)

    @staticmethod
    def ForwardDefaultActivationsSimpleWeightsWithBiasLinearBeforeReset():  # noqa: N802
        GRU_ONNXRuntimeUnitTests.DefaultActivationsSimpleWeightsWithBias(rows=1, linear_before_reset=1)

    @staticmethod
    def ReverseDefaultActivationsSimpleWeightsWithBiasLinearBeforeReset():  # noqa: N802
        GRU_ONNXRuntimeUnitTests.DefaultActivationsSimpleWeightsWithBias(
            rows=1, direction="reverse", linear_before_reset=1
        )

    @staticmethod
    def Legacy_TestGRUOpForwardBasic():  # noqa: N802
        print(GRU_ONNXRuntimeUnitTests.Legacy_TestGRUOpForwardBasic.__name__)

        input = np.array([[[-0.455351, -0.276391]], [[-0.185934, -0.269585]]]).astype(np.float32)

        W, R, B = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        gru = GRU_Helper(X=input, W=W, R=R, B=B)
        output = gru.run()
        print_results(output)

    @staticmethod
    def Legacy_TestGRUOpBackwardBasic():  # noqa: N802
        print(GRU_ONNXRuntimeUnitTests.Legacy_TestGRUOpBackwardBasic.__name__)

        input = np.array([[[-0.185934, -0.269585]], [[-0.455351, -0.276391]]]).astype(np.float32)

        W, R, B = ONNXRuntimeTestContext.OneDirectionWeights()  # noqa: N806
        gru = GRU_Helper(X=input, W=W, R=R, B=B, direction="reverse")
        output = gru.run()
        print_results(output)

    @staticmethod
    def Legacy_TestGRUOpBidirectionalBasic():  # noqa: N802
        print(GRU_ONNXRuntimeUnitTests.Legacy_TestGRUOpBidirectionalBasic.__name__)

        input = np.array([[[-0.455351, -0.276391]], [[-0.185934, -0.269585]]]).astype(np.float32)

        W, R, B = ONNXRuntimeTestContext.BidirectionalWeights()  # noqa: N806
        gru = GRU_Helper(X=input, W=W, R=R, B=B, direction="bidirectional")
        output = gru.run()
        print_results(output)


GRU_ONNXRuntimeUnitTests.ForwardDefaultActivationsSimpleWeightsNoBiasTwoRows()
GRU_ONNXRuntimeUnitTests.ReverseDefaultActivationsSimpleWeightsNoBiasTwoRows()
GRU_ONNXRuntimeUnitTests.BidirectionalDefaultActivationsSimpleWeightsNoBias()
GRU_ONNXRuntimeUnitTests.BidirectionalDefaultActivationsSimpleWeightsNoBias(linear_before_reset=1)

GRU_ONNXRuntimeUnitTests.ForwardDefaultActivationsSimpleWeightsWithBiasBatchParallel()
GRU_ONNXRuntimeUnitTests.ForwardDefaultActivationsSimpleWeightsWithBiasBatchParallelLinearBeforeReset()
GRU_ONNXRuntimeUnitTests.ReverseDefaultActivationsSimpleWeightsWithBiasBatchParallelLinearBeforeReset()
GRU_ONNXRuntimeUnitTests.ForwardDefaultActivationsSimpleWeightsWithBiasLinearBeforeReset()
GRU_ONNXRuntimeUnitTests.ReverseDefaultActivationsSimpleWeightsWithBiasLinearBeforeReset()

GRU_ONNXRuntimeUnitTests.Legacy_TestGRUOpForwardBasic()
GRU_ONNXRuntimeUnitTests.Legacy_TestGRUOpBackwardBasic()
GRU_ONNXRuntimeUnitTests.Legacy_TestGRUOpBidirectionalBasic()
