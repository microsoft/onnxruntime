# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
import multiprocessing
import numpy as np
import onnx
# use lines below when building ONNX Runtime from source with --enable_pybind
#import sys
#sys.path.append(r'X:\Repos\Lotus\build\Windows\Release\Release')
#sys.path.append('/repos/Lotus/build/Linux/Release')
import onnxruntime
from onnx import helper, numpy_helper
from onnx import shape_inference
from onnx import IR_VERSION
import os
from timeit import default_timer as timer

def generate_model(rnn_type, input_dim, hidden_dim, bidirectional, layers, model_name, batch_one=True, has_seq_len=False, onnx_opset_ver=7):
    model = onnx.ModelProto()
    model.ir_version = IR_VERSION
    
    opset = model.opset_import.add()
    opset.domain == 'onnx'
    opset.version = onnx_opset_ver
    num_directions = 2 if bidirectional else 1

    X = 'input'
    model.graph.input.add().CopyFrom(helper.make_tensor_value_info(X, onnx.TensorProto.FLOAT, ['s', 1 if batch_one else 'b', input_dim]))
    model.graph.initializer.add().CopyFrom(numpy_helper.from_array(np.asarray([0, 0, -1], dtype=np.int64), 'shape'))

    if has_seq_len:
        seq_len = 'seq_len'
        model.graph.input.add().CopyFrom(helper.make_tensor_value_info(seq_len, onnx.TensorProto.INT32, [1 if batch_one else 'b',]))

    gates = {'lstm':4, 'gru':3, 'rnn':1}[rnn_type]
    for i in range(layers):
        layer_input_dim = (input_dim if i == 0 else hidden_dim * num_directions)
        model.graph.initializer.add().CopyFrom(numpy_helper.from_array(np.random.rand(num_directions, gates*hidden_dim, layer_input_dim).astype(np.float32), 'W'+str(i)))
        model.graph.initializer.add().CopyFrom(numpy_helper.from_array(np.random.rand(num_directions, gates*hidden_dim, hidden_dim).astype(np.float32), 'R'+str(i)))
        model.graph.initializer.add().CopyFrom(numpy_helper.from_array(np.random.rand(num_directions, 2*gates*hidden_dim).astype(np.float32), 'B'+str(i)))
        layer_inputs = [X, 'W'+str(i), 'R'+str(i), 'B'+str(i)]
        if has_seq_len:
            layer_inputs += [seq_len]
        layer_outputs = ['layer_output_'+str(i)]
        model.graph.node.add().CopyFrom(helper.make_node(rnn_type.upper(), layer_inputs, layer_outputs, rnn_type+str(i), hidden_size=hidden_dim, direction='bidirectional' if bidirectional else 'forward'))
        model.graph.node.add().CopyFrom(helper.make_node('Transpose', layer_outputs, ['transposed_output_'+str(i)], 'transpose'+str(i), perm=[0,2,1,3]))
        model.graph.node.add().CopyFrom(helper.make_node('Reshape', ['transposed_output_'+str(i), 'shape'], ['reshaped_output_'+str(i)], 'reshape'+str(i)))
        X = 'reshaped_output_'+str(i)
    model.graph.output.add().CopyFrom(helper.make_tensor_value_info(X, onnx.TensorProto.FLOAT, ['s', 'b', hidden_dim * num_directions]))
    model = shape_inference.infer_shapes(model)
    onnx.save(model, model_name)

def perf_run(sess, feeds, min_counts=5, min_duration_seconds=10):
    # warm up
    sess.run([], feeds)

    start = timer()
    run = True
    count = 0
    per_iter_cost = []
    while run:
        iter_start = timer()
        sess.run([], feeds)
        end = timer()
        count = count + 1
        per_iter_cost.append(end - iter_start)
        if end - start >= min_duration_seconds and count >= min_counts:
            run = False
    return count, (end - start), per_iter_cost

def top_n_avg(per_iter_cost, n):
    # following the perf test methodology in [timeit](https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat)
    per_iter_cost.sort()
    return sum(per_iter_cost[:n]) * 1000 / n

def get_num_threads():
    return os.environ['OMP_NUM_THREADS'] if 'OMP_NUM_THREADS' in os.environ else None

def set_num_threads(num_threads):
    if num_threads:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
    else:
        del os.environ['OMP_NUM_THREADS']

class ScopedSetNumThreads:
    def __init__(self, num_threads):
        self.saved_num_threads_ = get_num_threads()
        self.num_threads_ = num_threads

    def __enter__(self):
        set_num_threads(self.num_threads_)

    def __exit__(self, type, value, tb):
        set_num_threads(self.saved_num_threads_)

def perf_test(rnn_type, num_threads, input_dim, hidden_dim, bidirectional, layers, seq_len, batch_size, top_n=5, min_duration_seconds=10):
    model_name = '{}_i{}_h{}_{}_l{}_{}.onnx'.format(rnn_type, input_dim, hidden_dim,
                                                    'bi' if bidirectional else '',
                                                    layers,
                                                    'batched' if batch_size > 1 else 'no_batch')

    generate_model(rnn_type, input_dim, hidden_dim, bidirectional, layers, model_name, batch_size == 1)
    feeds = {'input':np.random.rand(seq_len, batch_size, input_dim).astype(np.float32)}

    # run original model in CPU provider, using all threads
    # there are some local thread pool inside LSTM/GRU CPU kernel
    # that cannot be controlled by OMP or intra_op_num_threads
    sess = onnxruntime.InferenceSession(model_name, providers=['CPUExecutionProvider'])
    count, duration, per_iter_cost = perf_run(sess, feeds, min_counts=top_n, min_duration_seconds=min_duration_seconds)
    avg_rnn = top_n_avg(per_iter_cost, top_n)
    print('perf_rnn (with default threads) {}: run for {} iterations, top {} avg {:.3f} ms'.format(model_name, count, top_n, avg_rnn))

    # run converted model in Nuphar, using specified threads
    with ScopedSetNumThreads(num_threads) as scoped_set_num_threads:
        # run Scan model converted from original in Nuphar
        from .model_editor import convert_to_scan_model
        from ..tools.symbolic_shape_infer import SymbolicShapeInference
        scan_model_name = os.path.splitext(model_name)[0] + '_scan.onnx'
        convert_to_scan_model(model_name, scan_model_name)
        # note that symbolic shape inference is needed because model has symbolic batch dim, thus init_state is ConstantOfShape
        onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(scan_model_name)), scan_model_name)
        sess = onnxruntime.InferenceSession(scan_model_name, providers=onnxruntime.get_available_providers())
        count, duration, per_iter_cost = perf_run(sess, feeds, min_counts=top_n, min_duration_seconds=min_duration_seconds)
        avg_scan = top_n_avg(per_iter_cost, top_n)
        print('perf_scan (with {} threads) {}: run for {} iterations, top {} avg {:.3f} ms'.format(num_threads, scan_model_name, count, top_n, avg_scan))

        # quantize Scan model to int8 and run in Nuphar
        from .model_quantizer import convert_matmul_model
        int8_model_name = os.path.splitext(model_name)[0] + '_int8.onnx'
        convert_matmul_model(scan_model_name, int8_model_name)
        onnx.save(SymbolicShapeInference.infer_shapes(onnx.load(int8_model_name)), int8_model_name)
        sess = onnxruntime.InferenceSession(int8_model_name, providers=onnxruntime.get_available_providers())
        count, duration, per_iter_cost = perf_run(sess, feeds, min_counts=top_n, min_duration_seconds=min_duration_seconds)
        avg_int8 = top_n_avg(per_iter_cost, top_n)
        print('perf_int8 (with {} threads) {}: run for {} iterations, top {} avg {:.3f} ms'.format(num_threads, int8_model_name, count, top_n, avg_int8))

    return avg_rnn, avg_scan, avg_int8

def perf_test_auto(auto_file):
    # generate reports in csv format
    with open('single_thread_' + auto_file + '.csv', 'w') as f:
        print('single thread test: unidirection 4-layer lstm/gru/rnn with input_dim=128 batch_size=1', file=f)
        print('rnn_type,hidden,seq_len,avg_rnn,avg_nuphar_fp,avg_nuphar_int8,speedup_fp,speedup_int8', file=f)
        for rnn_type in ['lstm', 'gru', 'rnn']:
            for hidden_dim in [32, 128, 1024, 2048]:
                for seq_len in [1, 16, 32, 64]:
                    avg_rnn, avg_scan, avg_int8 = perf_test(rnn_type, 1, 128, hidden_dim, False, 4, seq_len, 1)
                    print('{},{},{},{},{},{},{},{}'.format(rnn_type,hidden_dim, seq_len, avg_rnn, avg_scan, avg_int8, avg_rnn/avg_scan, avg_rnn/avg_int8), file=f)

    with open('multi_thread_' + auto_file + '.csv', 'w') as f:
        print('multi-thread test: unidirection 4-layer lstm/gru/rnn with input_dim=128 seq_len=32 batch_size=1', file=f)
        print('rnn_type,threads,hidden,avg_rnn,avg_nuphar_fp,avg_nuphar_int8,speedup_fp,speedup_int8', file=f)
        for rnn_type in ['lstm', 'gru', 'rnn']:
            for num_threads in [1, 2, 4]:
                for hidden_dim in [32, 128, 1024, 2048]:
                    avg_rnn, avg_scan, avg_int8 = perf_test(rnn_type, num_threads, 128, hidden_dim, False, 4, seq_len, 1)
                    print('{},{},{},{},{},{},{},{}'.format(rnn_type,num_threads, hidden_dim, avg_rnn, avg_scan, avg_int8, avg_rnn/avg_scan, avg_rnn/avg_int8), file=f)

    with open('batch_single_thread_' + auto_file + '.csv', 'w') as f:
        print('single thread test: unidirection 4-layer lstm/gru/rnn with input_dim=128 hidden_dim=1024', file=f)
        print('rnn_type,seq_len,batch_size,avg_rnn,avg_nuphar_fp,avg_nuphar_int8,speedup_fp,speedup_int8', file=f)
        for rnn_type in ['lstm', 'gru', 'rnn']:
            for seq_len in [1, 16, 32, 64]:
                for batch_size in [1, 4, 16, 64]:
                    avg_rnn, avg_scan, avg_int8 = perf_test(rnn_type, 1, 128, 1024, False, 4, seq_len, batch_size)
                    print('{},{},{},{},{},{},{},{}'.format(rnn_type,seq_len, batch_size, avg_rnn, avg_scan, avg_int8, avg_rnn/avg_scan, avg_rnn/avg_int8), file=f)

    with open('batch_multi_thread_' + auto_file + '.csv', 'w') as f:
        print('batch thread test: unidirection 4-layer lstm/gru/rnn with input_dim=128 hidden_dim=1024 seq_len=32', file=f)
        print('rnn_type,threads,batch_size,avg_rnn,avg_nuphar_fp,avg_nuphar_int8,speedup_fp,speedup_int8', file=f)
        for rnn_type in ['lstm', 'gru', 'rnn']:
            for num_threads in [1, 2, 4]:
                for batch_size in [1, 4, 16, 64]:
                    avg_rnn, avg_scan, avg_int8 = perf_test(rnn_type, num_threads, 128, 1024, False, 4, 32, batch_size)
                    print('{},{},{},{},{},{},{},{}'.format(rnn_type,num_threads, batch_size, avg_rnn, avg_scan, avg_int8, avg_rnn/avg_scan, avg_rnn/avg_int8), file=f)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_type', help='Type of rnn, one of lstm/gru/rnn', choices=['lstm', 'gru', 'rnn'], default='lstm')
  parser.add_argument('--input_dim', help='Input size of lstm/gru/rnn', type=int, default=128)
  parser.add_argument('--hidden_dim', help='Hidden size of lstm/gru/rnn', type=int, default=1024)
  parser.add_argument('--bidirectional', help='Use bidirectional', action='store_true', default=False)
  parser.add_argument('--layers', help='Number of layers', type=int, default=4)
  parser.add_argument('--seq_len', help='Sequence length', type=int, default=32)
  parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
  parser.add_argument('--num_threads', help='Number of MKL threads', type=int, default=multiprocessing.cpu_count())
  parser.add_argument('--top_n', help='Fastest N samples to compute average time', type=int, default=5)
  parser.add_argument('--auto', help='Auto_name (usually CPU type) for auto test to generate (batch_)single|multithread_<auto_name>.csv files', default=None)
  return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.auto:
        perf_test_auto(args.auto)
    else:
        print('Testing model: ', args.rnn_type.upper())
        print('  input_dim: ', args.input_dim)
        print('  hidden_dim: ', args.hidden_dim)
        if args.bidirectional:
            print('  bidirectional')
        print('  layers: ', args.layers)
        cpu_count = multiprocessing.cpu_count()
        num_threads = max(min(args.num_threads, cpu_count), 1)
        print('Test setup')
        print('  cpu_count: ', cpu_count)
        print('  num_threads: ', num_threads)
        print('  seq_len: ', args.seq_len)
        print('  batch_size: ', args.batch_size)
        perf_test(args.rnn_type, num_threads, args.input_dim, args.hidden_dim, args.bidirectional, args.layers, args.seq_len, args.batch_size, args.top_n)
