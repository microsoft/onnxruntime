# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import cntk as C
import numpy as np
import onnx
import os

model_file = 'model.onnx'
data_dir = 'test_data_set_0'

def SaveTensorProto(file_path, variable, data, name):
    tp = onnx.TensorProto()
    tp.name = name if name else variable.uid
    # ONNX input shape always has sequence axis as the first dimension, if sequence axis exists
    for i in range(len(variable.dynamic_axes)):
        tp.dims.append(data.shape[len(variable.dynamic_axes) - 1 - i])
    for (i,d) in enumerate(variable.shape):
        tp.dims.append(d) if d > 0 else tp.dims.append(data.shape[len(data.shape)-len(variable.shape)+i])
    tp.data_type = onnx.TensorProto.FLOAT
    tp.raw_data = data.tobytes()
    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())

def SaveData(test_data_dir, prefix, variables, data_list, name_replacements=None):
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
    for (i, d), v in zip(enumerate(data_list), variables):
        SaveTensorProto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), v, d, name_replacements[v.uid] if name_replacements else None)

def Save(dir, func, feed, outputs):
    if not os.path.exists(dir):
        os.makedirs(dir)
    onnx_file = os.path.join(dir,model_file)
    func.save(onnx_file, C.ModelFormat.ONNX)

    # onnx model may have different name for RNN initial states as inputs
    cntk_to_actual_names = {}
    with open(onnx_file, 'rb') as ff:
        sf = ff.read()
    model = onnx.ModelProto()
    model.ParseFromString(sf)
    for actual_input in model.graph.input:
        actual_input_name = actual_input.name
        for cntk_input in func.arguments:
            cntk_name = cntk_input.uid
            if actual_input_name.startswith(cntk_name):
                cntk_to_actual_names[cntk_name] = actual_input_name

    if type(feed) is not dict:
       feed = {func.arguments[0]:feed}

    if type(outputs) is not dict:
       outputs = {func.outputs[0]:outputs}

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    SaveData(test_data_dir, 'input', func.arguments, [feed[var] for var in func.arguments], cntk_to_actual_names)
    SaveData(test_data_dir, 'output', func.outputs, [outputs[var] for var in func.outputs])

def GenSimple():
    x = C.input_variable((1,3,)) # TODO: fix CNTK exporter bug with shape (3,)
    y = C.layers.Embedding(2)(x) + C.parameter((-1,))
    data_x = np.random.rand(1,*x.shape).astype(np.float32)
    data_y = y.eval(data_x)
    Save('test_simple', y, data_x, data_y)

def GenSharedWeights():
    x = C.input_variable((1,3,))
    y = C.layers.Embedding(2)(x)
    y = y + y.parameters[0]
    data_x = np.random.rand(1,*x.shape).astype(np.float32)
    data_y = y.eval(data_x)
    Save('test_shared_weights', y, data_x, data_y)
    
def GenSimpleMNIST():
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200

    feature = C.input_variable(input_dim, np.float32)

    scaled_input = C.element_times(C.constant(0.00390625, shape=(input_dim,)), feature)

    z = C.layers.Sequential([C.layers.For(range(num_hidden_layers), lambda i: C.layers.Dense(hidden_layers_dim, activation=C.relu)),
                    C.layers.Dense(num_output_classes)])(scaled_input)

    model = C.softmax(z)

    data_feature = np.random.rand(1,*feature.shape).astype(np.float32)
    data_output = model.eval(data_feature)
    Save('test_simpleMNIST', model, data_feature, data_output)

def GenMatMul_1k():
    feature = C.input_variable((1024, 1024,), np.float32)
    model = C.times(feature, C.parameter((1024,1024), init=C.glorot_uniform()))

    data_feature = np.random.rand(1,*feature.shape).astype(np.float32)
    data_output = model.eval(data_feature)
    Save('test_MatMul_1k', model, data_feature, data_output)

def LSTM(cell_dim, use_scan=True):
    # we now create an LSTM_cell function and call it with the input and placeholders
    LSTM_cell = C.layers.LSTM(cell_dim)

    @C.Function
    def func(dh, dc, input):
        LSTM_func = LSTM_cell(dh, dc, input)
        if use_scan:
            LSTM_func_root = C.as_composite(LSTM_func.outputs[0].owner.block_root)
            args = LSTM_func_root.arguments
            LSTM_func = LSTM_func_root.clone(C.CloneMethod.share, {args[0]:input, args[1]:dh, args[2]:dc})
        return LSTM_func

    return func

def GenLSTMx4(use_scan):
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

    postfix = 'Scan' if use_scan else 'LSTM'

    data_feature = np.random.rand(1,64,128).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature))
    Save('test_LSTMx4_' + postfix, model, data_feature, data_output)
    
def GenScan():
    np.random.seed(0)
    feature = C.sequence.input_variable((3,), np.float32)
    model = C.layers.For(range(4), lambda : C.layers.Recurrence(LSTM(2, use_scan=True)))(feature)

    data_feature = np.random.rand(1,5,3).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature))

    # print values for test as ground truth
    print("Scan input\n", data_feature, "\nScan output\n", data_output)

    Save('test_Scan', model, data_feature, data_output)

def GenSimpleScan():
    feature = C.sequence.input_variable((128,), np.float32)
    param = C.parameter(shape=(1,), dtype=np.float32)
    scan = C.layers.Recurrence(lambda h, x: x + h + param)(feature)
    model = C.sequence.reduce_sum(scan)
    data_feature = np.random.rand(1,64,128).astype(np.float32)
    data_output = np.asarray(model.eval(data_feature), dtype=np.float32)
    Save('test_SimpleScan', model, data_feature, data_output)

def GenLCBLSTM():
    nc_len = 16
    nr_len = 16
    in_dim = 128
    cell_dim = 256
    batch_size = 1
    input = C.sequence.input_variable((in_dim,))
    init_h = C.input_variable((cell_dim,))
    init_c = C.input_variable((cell_dim,))

    fwd_cell = LSTM(cell_dim, use_scan=True)
    fwd_hc = C.layers.RecurrenceFrom(fwd_cell, go_backwards=False, return_full_state=True)(init_h, init_c, input)

    fwd_h_nc = C.sequence.slice(fwd_hc[0], -nr_len-1, -nr_len)
    fwd_c_nc = C.sequence.slice(fwd_hc[1], -nr_len-1, -nr_len)

    bwd_cell = LSTM(cell_dim, use_scan=True)
    bwd = C.layers.Recurrence(bwd_cell, go_backwards=True)(input)

    nr = C.splice(fwd_hc[0], bwd)
    # workaround CNTK bug in slice output name by + 0
    model = C.combine([C.sequence.reduce_sum(nr), fwd_h_nc + 0, fwd_c_nc + 0])
    
    input_data = np.random.rand(batch_size, nc_len+nr_len, in_dim).astype(np.float32)
    init_h_data = np.random.rand(batch_size, cell_dim).astype(np.float32)
    init_c_data = np.random.rand(batch_size, cell_dim).astype(np.float32)
    feed = {input:input_data, init_h:init_h_data, init_c:init_c_data}
    data_output = model.eval(feed)
    Save('test_LCBLSTM', model, feed, data_output)

if __name__=='__main__':
    np.random.seed(0)
    GenSimple()
    GenSharedWeights()
    GenSimpleMNIST()
    GenMatMul_1k()
    GenLSTMx4(use_scan=True)
    GenLSTMx4(use_scan=False)
    GenSimpleScan()
    GenLCBLSTM()
    GenScan()
