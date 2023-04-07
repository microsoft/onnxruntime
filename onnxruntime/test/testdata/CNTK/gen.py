# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import cntk as C
import numpy as np
import onnx
from onnx import numpy_helper

model_file = "model.onnx"
data_dir = "test_data_set_0"


def SaveTensorProto(file_path, variable, data, name):  # noqa: N802
    # ONNX input shape always has sequence axis as the first dimension, if sequence axis exists
    if len(variable.dynamic_axes) == 2:
        data = data.transpose((1, 0, *tuple(range(2, len(data.shape)))))
    tp = numpy_helper.from_array(data, name if name else variable.uid)
    onnx.save_tensor(tp, file_path)


def SaveData(test_data_dir, prefix, variables, data_list, name_replacements=None):  # noqa: N802
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
    for (i, d), v in zip(enumerate(data_list), variables):
        SaveTensorProto(
            os.path.join(test_data_dir, f"{prefix}_{i}.pb"),
            v,
            d,
            name_replacements[v.uid] if name_replacements else None,
        )


def Save(dir, func, feed, outputs):  # noqa: N802
    if not os.path.exists(dir):
        os.makedirs(dir)
    onnx_file = os.path.join(dir, model_file)
    func.save(onnx_file, C.ModelFormat.ONNX)

    # onnx model may have different name for RNN initial states as inputs
    cntk_to_actual_names = {}
    model = onnx.load(onnx_file)
    for actual_input in model.graph.input:
        actual_input_name = actual_input.name
        for cntk_input in func.arguments:
            cntk_name = cntk_input.uid
            if actual_input_name.startswith(cntk_name):
                cntk_to_actual_names[cntk_name] = actual_input_name

    if type(feed) is not dict:
        feed = {func.arguments[0]: feed}

    if type(outputs) is not dict:
        outputs = {func.outputs[0]: outputs}

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    SaveData(
        test_data_dir,
        "input",
        func.arguments,
        [feed[var] for var in func.arguments],
        cntk_to_actual_names,
    )
    SaveData(test_data_dir, "output", func.outputs, [outputs[var] for var in func.outputs])


def GenSimple():  # noqa: N802
    x = C.input_variable(
        (
            1,
            3,
        )
    )  # TODO: fix CNTK exporter bug with shape (3,)
    y = C.layers.Embedding(2)(x) + C.parameter((-1,))
    data_x = np.random.rand(1, *x.shape).astype(np.float32)
    data_y = y.eval(data_x)
    Save("test_simple", y, data_x, data_y)


def GenSharedWeights():  # noqa: N802
    x = C.input_variable(
        (
            1,
            3,
        )
    )
    y = C.layers.Embedding(2)(x)
    y = y + y.parameters[0]
    data_x = np.random.rand(1, *x.shape).astype(np.float32)
    data_y = y.eval(data_x)
    Save("test_shared_weights", y, data_x, data_y)


def GenSimpleMNIST():  # noqa: N802
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200

    feature = C.input_variable(input_dim, np.float32)

    scaled_input = C.element_times(C.constant(0.00390625, shape=(input_dim,)), feature)

    z = C.layers.Sequential(
        [
            C.layers.For(
                range(num_hidden_layers),
                lambda i: C.layers.Dense(hidden_layers_dim, activation=C.relu),
            ),
            C.layers.Dense(num_output_classes),
        ]
    )(scaled_input)

    model = C.softmax(z)

    data_feature = np.random.rand(1, *feature.shape).astype(np.float32)
    data_output = model.eval(data_feature)
    Save("test_simpleMNIST", model, data_feature, data_output)


def GenMatMul_1k():  # noqa: N802
    feature = C.input_variable(
        (
            1024,
            1024,
        ),
        np.float32,
    )
    model = C.times(feature, C.parameter((1024, 1024), init=C.glorot_uniform()))

    data_feature = np.random.rand(1, *feature.shape).astype(np.float32)
    data_output = model.eval(data_feature)
    Save("test_MatMul_1k", model, data_feature, data_output)


def LSTM(cell_dim, use_scan=True):  # noqa: N802
    # we now create an LSTM_cell function and call it with the input and placeholders
    LSTM_cell = C.layers.LSTM(cell_dim)  # noqa: N806

    @C.Function
    def func(dh, dc, input):
        LSTM_func = LSTM_cell(dh, dc, input)  # noqa: N806
        if use_scan:
            LSTM_func_root = C.as_composite(LSTM_func.outputs[0].owner.block_root)  # noqa: N806
            args = LSTM_func_root.arguments
            LSTM_func = LSTM_func_root.clone(  # noqa: N806
                C.CloneMethod.share, {args[0]: input, args[1]: dh, args[2]: dc}
            )
        return LSTM_func

    return func


def GenLSTMx4(use_scan):  # noqa: N802
    feature = C.sequence.input_variable((128,), np.float32)
    lstm1 = C.layers.Recurrence(LSTM(512, use_scan))(feature)
    lstm2_fw = C.layers.Recurrence(LSTM(512, use_scan))(lstm1)
    lstm2_bw = C.layers.Recurrence(LSTM(512, use_scan), go_backwards=True)(lstm1)
    lstm2 = C.splice(lstm2_fw, lstm2_bw, axis=0)
    lstm3_fw = C.layers.Recurrence(LSTM(512, use_scan))(lstm2)
    lstm3_bw = C.layers.Recurrence(LSTM(512, use_scan), go_backwards=True)(lstm2)
    lstm3 = C.splice(lstm3_fw, lstm3_bw, axis=0)
    lstm4 = C.layers.Recurrence(LSTM(512, use_scan))(lstm3)
    model = lstm4

    postfix = "Scan" if use_scan else "LSTM"

    data_feature = np.random.rand(1, 64, 128).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature))
    Save("test_LSTMx4_" + postfix, model, data_feature, data_output)


def GenScan():  # noqa: N802
    np.random.seed(0)
    feature = C.sequence.input_variable((3,), np.float32)
    model = C.layers.For(range(4), lambda: C.layers.Recurrence(LSTM(2, use_scan=True)))(feature)

    data_feature = np.random.rand(2, 5, 3).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature))

    Save("test_Scan", model, data_feature, data_output)

    # Currently CNTK only outputs batch == 1, do some editing
    in_mp = onnx.load("test_Scan/model.onnx")
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.graph.ClearField("initializer")

    # change LSTM init_c/h into inputs to support truncated sequence
    # as batch dimension is unknown on those data when building model
    # note here we assume init_c/h starts from 0
    # if not the case, user need to manually broadcast it for feed
    num_inputs = 1
    for i in in_mp.graph.initializer:
        if i.name.startswith("Constant"):
            shape = i.dims
            shape[0] = 2
            aa = np.zeros(shape, dtype=np.float32)
            tp = numpy_helper.from_array(aa, i.name)
            with open("test_Scan/test_data_set_0/input_" + str(num_inputs) + ".pb", "wb") as ff:
                ff.write(tp.SerializeToString())
            num_inputs = num_inputs + 1
        else:
            out_mp.graph.initializer.add().CopyFrom(i)

    for vi in list(out_mp.graph.input) + list(out_mp.graph.output) + list(out_mp.graph.value_info):
        dim = vi.type.tensor_type.shape.dim
        dim[len(dim) - 2].dim_param = "batch"

    for n in out_mp.graph.node:
        if n.op_type == "Scan":
            body = [attr for attr in n.attribute if attr.name == "body"][0]
            for vi in list(body.g.input) + list(body.g.output) + list(body.g.value_info):
                dim = vi.type.tensor_type.shape.dim
                dim[0].dim_param = "batch"

    onnx.save(out_mp, "test_Scan/model.onnx", "wb")


def GenSimpleScan():  # noqa: N802
    feature = C.sequence.input_variable((128,), np.float32)
    param = C.parameter(shape=(1,), dtype=np.float32)
    scan = C.layers.Recurrence(lambda h, x: x + h + param)(feature)
    model = C.sequence.reduce_sum(scan)
    data_feature = np.random.rand(1, 64, 128).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature), dtype=np.float32)
    Save("test_SimpleScan", model, data_feature, data_output)


def GenGRU():  # noqa: N802
    feature = C.sequence.input_variable((64,), np.float32)
    gru_fw = C.layers.Recurrence(C.layers.GRU(128))(feature)
    gru_bw = C.layers.Recurrence(C.layers.GRU(128), go_backwards=True)(feature)
    model = C.splice(gru_fw, gru_bw, axis=0)
    data_feature = np.random.rand(1, 16, 64).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature))
    Save("test_GRU", model, data_feature, data_output)


def GenRNN():  # noqa: N802
    feature = C.sequence.input_variable((64,), np.float32)
    model = C.optimized_rnnstack(
        feature,
        C.parameter(
            (
                C.InferredDimension,
                64,
            ),
            init=C.glorot_uniform(),
        ),
        128,
        2,
        True,
        "rnnReLU",
    )
    data_feature = np.random.rand(1, 16, 64).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature))
    Save("test_RNN", model, data_feature, data_output)


if __name__ == "__main__":
    np.random.seed(0)
    GenSimple()
    GenSharedWeights()
    GenSimpleMNIST()
    GenMatMul_1k()
    GenLSTMx4(use_scan=True)
    GenLSTMx4(use_scan=False)
    GenSimpleScan()
    GenScan()
    GenGRU()
    GenRNN()
