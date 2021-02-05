# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as onnxrt
from helper import get_name
import os
from onnxruntime.nuphar.rnn_benchmark import perf_test, generate_model
from onnxruntime.nuphar.model_tools import validate_with_ort, run_shape_inference
import shutil
import sys
import subprocess
import tarfile
import unittest
import urllib.request

def reference_gemm(a, b, c, alpha, beta, transA, transB):
    a = a if transA == 0 else a.T
    b = b if transB == 0 else b.T
    y = alpha * np.dot(a, b) + beta * c
    return y

def set_gemm_node_attrs(attrs, config):
    if config['alpha'] != 1.0:
        attrs['alpha'] = config['alpha']
    if config['beta'] != 1.0:
        attrs['beta'] = config['beta']
    if config['transA']:
        attrs['transA'] = 1
    if config['transB']:
        attrs['transB'] = 1

def generate_gemm_inputs_initializers(graph, config, added_inputs_initializers={}, extend=False):
    M = config['M']
    K = config['K']
    N = config['N']

    shape_a = [K, M] if config['transA'] else [M, K]
    shape_b = [N, K] if config['transB'] else [K, N]
    shape_c = [M, N]

    # when A/B/C are graph input of the main graph which contains
    # a Scan node, then they need an extra 'seq' dimension
    input_shape_a = ['seq'] + shape_a if extend else shape_a
    input_shape_b = ['seq'] + shape_b if extend else shape_b
    input_shape_c = ['seq'] + shape_c if extend else shape_c

    np.random.seed(12345)
    a = np.random.ranf(shape_a).astype(np.float32)
    b = np.random.ranf(shape_b).astype(np.float32)
    c = np.random.ranf(shape_c).astype(np.float32) if config['withC'] else np.array(0)

    init_a = a if config['initA'] else None
    init_b = b if config['initB'] else None
    init_c = c if config['initC'] else None

    A = config['A']
    B = config['B']
    C = config['C']

    # A is an initializer
    if A in added_inputs_initializers:
        a = added_inputs_initializers[A]
    else:
        added_inputs_initializers[A] = a
        if init_a is not None:
            graph.initializer.add().CopyFrom(numpy_helper.from_array(init_a, A))
        else:
            graph.input.add().CopyFrom(helper.make_tensor_value_info(A,
                                                                     onnx.TensorProto.FLOAT,
                                                                     input_shape_a))

    # B is an initializer
    if B in added_inputs_initializers:
        b = added_inputs_initializers[B]
    else:
        added_inputs_initializers[B] = b
        if init_b is not None:
            graph.initializer.add().CopyFrom(numpy_helper.from_array(init_b, B))
        else:
            graph.input.add().CopyFrom(helper.make_tensor_value_info(B,
                                                                     onnx.TensorProto.FLOAT,
                                                                     input_shape_b))

    if config['withC']:
        if C in added_inputs_initializers:
            c = added_inputs_initializers[C]
        else:
            added_inputs_initializers[C] = c
            if init_c is not None:
                graph.initializer.add().CopyFrom(numpy_helper.from_array(init_c, C))
            else:
                graph.input.add().CopyFrom(helper.make_tensor_value_info(C,
                                                                         onnx.TensorProto.FLOAT,
                                                                         input_shape_c))

    return (a, b, c)

def generate_gemm_model(model_name, config):
    model = onnx.ModelProto()
    model.ir_version = onnx.IR_VERSION
    opset = model.opset_import.add()
    opset.version = 11

    added_inputs_initializers = {}
    (a, b, c) = generate_gemm_inputs_initializers(model.graph, config, added_inputs_initializers)

    node_inputs = [config['A'], config['B']]
    if config['withC']:
        node_inputs.append(config['C'])

    attrs = {}
    set_gemm_node_attrs(attrs, config)
    node = helper.make_node('Gemm', node_inputs, [config['Y']], config['node_name'], **attrs)
    model.graph.node.add().CopyFrom(node)

    shape_output = [config['M'], config['N']]
    model.graph.output.add().CopyFrom(helper.make_tensor_value_info(config['Y'],
                                                                    onnx.TensorProto.FLOAT,
                                                                    shape_output))

    # compute reference output
    y = reference_gemm(a, b, c, config['alpha'], config['beta'], config['transA'], config['transB'])

    onnx.save(model, model_name)
    return (a, b, c, y)

def generate_gemm_node_subgraph(scan_body, scan_node_inputs, postfix, config, added_inputs):
    M = config['M']
    K = config['K']
    N = config['N']

    shape_a = [K, M] if config['transA'] else [M, K]
    shape_b = [N, K] if config['transB'] else [K, N]

    A = config['A']
    B = config['B']

    gemm_node_inputs = []
    # A comes from the outer graph if it's an initializer
    if config['initA']:
        gemm_node_inputs.append(A)
    else:
        gemm_node_inputs.append(A + postfix)
        if A not in added_inputs:
            added_inputs[A] = 1
            scan_node_inputs.append(A)
            scan_body.input.add().CopyFrom(helper.make_tensor_value_info(A + postfix,
                                                                         onnx.TensorProto.FLOAT,
                                                                         shape_a))

    # B comes from the outer graph if it's an initializer
    if config['initB']:
        gemm_node_inputs.append(B)
    else:
        gemm_node_inputs.append(B + postfix)
        if B not in added_inputs:
            added_inputs[B] = 1
            scan_node_inputs.append(B)
            scan_body.input.add().CopyFrom(helper.make_tensor_value_info(B + postfix,
                                                                         onnx.TensorProto.FLOAT,
                                                                         shape_b))

    # C comes from Scan state
    if config['withC']:
        gemm_node_inputs.append('in_' + config['C'] + postfix)

    attrs = {}
    set_gemm_node_attrs(attrs, config)
    node = helper.make_node('Gemm',
                            gemm_node_inputs,
                            [config['Y'] + postfix],
                            config['node_name'],
                            **attrs)
    scan_body.node.add().CopyFrom(node)

def generate_gemm_scan_model(model_name, config1, config2):
    model = onnx.ModelProto()
    model.ir_version = onnx.IR_VERSION
    opset = model.opset_import.add()
    opset.version = 11

    # Based on the given configs, we would have a model like below:
    # Main graph, where C is an initializer and passed as the input state for the Scan:
    #      C  input_1A input_2A
    #       \     |    /
    #        \    |   /
    #           Scan
    #             |
    #           output
    #
    # Scan's subgraph, where out_C is the output state of the Scan
    # input_1A  B  C  input_2A B  C
    #     \     | /       \    | /
    #      \    |/         \   |/
    #      Gemm_1           Gemm_2
    #           \          /
    #            \        /
    #               Sub
    #              /   \
    #           out_C  output
    #
    # config1 and config2 configure alpha/beta/transA/transB for Gemm_1 and Gemm_2, respectively.

    scan_body = onnx.GraphProto()
    scan_body.name = 'gemm_subgraph'

    shape_c1 = [config1['M'], config1['N']]
    shape_c2 = [config2['M'], config2['N']]
    assert shape_c1 == shape_c2
    C1 = config1['C']
    C2 = config2['C']

    scan_node_inputs = []
    postfix = '_subgraph'
    states_cnt = 0
    # make sure we create state inputs first
    if config1['withC']:
        assert config1['initC']
        states_cnt = states_cnt + 1
        scan_node_inputs.append(C1)
        scan_body.input.add().CopyFrom(helper.make_tensor_value_info('in_' + C1 + postfix,
                                                                     onnx.TensorProto.FLOAT,
                                                                     shape_c1))
    if config2['withC'] and C1 != C2:
        assert config2['initC']
        states_cnt = states_cnt + 1
        scan_node_inputs.append(C2)
        scan_body.input.add().CopyFrom(helper.make_tensor_value_info('in_' + C2 + postfix,
                                                                     onnx.TensorProto.FLOAT,
                                                                     shape_c2))

    added_inputs_subgraph = {}
    generate_gemm_node_subgraph(scan_body,
                                scan_node_inputs,
                                postfix,
                                config1,
                                added_inputs_subgraph)
    generate_gemm_node_subgraph(scan_body,
                                scan_node_inputs,
                                postfix,
                                config2,
                                added_inputs_subgraph)

    sub_output = 'sub_output' + postfix
    # create a Sub op instead of Add to break the MatMul-to-Gemm rewriting rule
    # performed by the ort optimizer
    sub_node = helper.make_node('Sub',
                                [config1['Y'] + postfix, config2['Y'] + postfix],
                                [sub_output],
                                'sub_node')
    scan_body.node.add().CopyFrom(sub_node)

    scan_node_outputs = []
    # create state outputs
    if config1['withC']:
        id_node1 = onnx.helper.make_node('Identity',
                                         [sub_output],
                                         ['out_' + C1 + postfix],
                                         'id_node1')
        scan_body.node.add().CopyFrom(id_node1)
        scan_body.output.add().CopyFrom(helper.make_tensor_value_info('out_' + C1 + postfix,
                                                                      onnx.TensorProto.FLOAT,
                                                                      shape_c1))
        scan_node_outputs.append('out_' + C1)

    if config2['withC'] and C1 != C2:
        id_node2 = onnx.helper.make_node('Identity',
                                         [sub_output],
                                         ['out_' + C2 + postfix],
                                         'id_node2')
        scan_body.node.add().CopyFrom(id_node2)
        scan_body.output.add().CopyFrom(helper.make_tensor_value_info('out_' + C2 + postfix,
                                                                      onnx.TensorProto.FLOAT,
                                                                      shape_c2))
        scan_node_outputs.append('out_' + C2)

    # scan subgraph output
    scan_body.output.add().CopyFrom(helper.make_tensor_value_info(sub_output,
                                                                  onnx.TensorProto.FLOAT,
                                                                  shape_c1))
    scan_node_outputs.append('scan_output')

    # create scan node
    inputs_cnt = len(scan_node_inputs) - states_cnt
    assert inputs_cnt > 0

    scan_node = onnx.helper.make_node('Scan',
                                      scan_node_inputs,
                                      scan_node_outputs,
                                      'scan_node',
                                      num_scan_inputs=inputs_cnt,
                                      body=scan_body)
    model.graph.node.add().CopyFrom(scan_node)

    added_inputs_initializers = {}
    # main graph inputs and initializers
    (a1, b1, c1) = generate_gemm_inputs_initializers(model.graph,
                                                     config1,
                                                     added_inputs_initializers,
                                                     extend=True)
    (a2, b2, c2) = generate_gemm_inputs_initializers(model.graph,
                                                     config2,
                                                     added_inputs_initializers,
                                                     extend=True)

    shape_output = ['seq', config1['M'], config1['N']]
    # main graph outputs
    model.graph.output.add().CopyFrom(helper.make_tensor_value_info('scan_output',
                                                                    onnx.TensorProto.FLOAT,
                                                                    shape_output))
    onnx.save(model, model_name)
    return (a1, b1, c1, a2, b2, c2)

def set_gemm_model_inputs(config, test_inputs, a, b, c):
    if not config['initA']:
        test_inputs[config['A']] = a
    if not config['initB']:
        test_inputs[config['B']] = b
    if config['withC'] and not config['initC']:
        test_inputs[config['C']] = c


def make_providers(nuphar_settings):
    return [
        ('NupharExecutionProvider', {
            'nuphar_settings': nuphar_settings
        }),
        'CPUExecutionProvider',
    ]


class TestNuphar(unittest.TestCase):

    def test_bidaf(self):
        cwd = os.getcwd()

        bidaf_dir_src = '/build/models/opset9/test_bidaf'

        bidaf_dir = os.path.join(cwd, 'bidaf')
        shutil.copytree(bidaf_dir_src, bidaf_dir)

        bidaf_dir = os.path.join(cwd, 'bidaf')
        bidaf_model = os.path.join(bidaf_dir, 'model.onnx')
        run_shape_inference(bidaf_model, bidaf_model)
        bidaf_scan_model = os.path.join(bidaf_dir, 'bidaf_scan.onnx')
        bidaf_opt_scan_model = os.path.join(bidaf_dir, 'bidaf_opt_scan.onnx')
        bidaf_int8_scan_only_model = os.path.join(bidaf_dir, 'bidaf_int8_scan_only.onnx')
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.nuphar.model_editor', '--input', bidaf_model, '--output',
            bidaf_scan_model, '--mode', 'to_scan'
        ],
                       check=True,
                       cwd=cwd)
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.nuphar.model_editor', '--input', bidaf_scan_model, '--output',
            bidaf_opt_scan_model, '--mode', 'opt_inproj'
        ],
                       check=True,
                       cwd=cwd)
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.nuphar.model_quantizer', '--input', bidaf_opt_scan_model, '--output',
            bidaf_int8_scan_only_model, '--only_for_scan'
        ],
                       check=True,
                       cwd=cwd)

        # run onnx_test_runner to verify results
        # use -M to disable memory pattern
        onnx_test_runner = os.path.join(cwd, 'onnx_test_runner')
        subprocess.run([onnx_test_runner, '-e', 'nuphar', '-M', '-c', '1', '-j', '1', '-n', 'bidaf', cwd], check=True, cwd=cwd)

        # test AOT on the quantized model
        if os.name not in ['nt', 'posix']:
            return  # don't run the rest of test if AOT is not supported

        cache_dir = os.path.join(cwd, 'nuphar_cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

        # prepare feed
        feed = {}
        for i in range(4):
            tp = onnx.load_tensor(os.path.join(bidaf_dir, 'test_data_set_0', 'input_{}.pb'.format(i)))
            feed[tp.name] = numpy_helper.to_array(tp)

        for model in [bidaf_opt_scan_model, bidaf_int8_scan_only_model]:
            nuphar_settings = 'nuphar_cache_path:{}'.format(cache_dir)
            for isa in ['avx', 'avx2', 'avx512']:
                # JIT cache happens when initializing session
                sess = onnxrt.InferenceSession(
                    model, providers=make_providers(nuphar_settings + ', nuphar_codegen_target:' + isa))

            cache_dir_content = os.listdir(cache_dir)
            assert len(cache_dir_content) == 1
            cache_versioned_dir = os.path.join(cache_dir, cache_dir_content[0])
            so_name = os.path.basename(model) + '.so'
            subprocess.run([
                sys.executable, '-m', 'onnxruntime.nuphar.create_shared', '--input_dir', cache_versioned_dir,
                '--output_name', so_name
            ],
                           check=True)

            nuphar_settings = 'nuphar_cache_path:{}, nuphar_cache_so_name:{}, nuphar_cache_force_no_jit:{}'.format(
                cache_dir, so_name, 'on')
            sess = onnxrt.InferenceSession(model, providers=make_providers(nuphar_settings))
            sess.run([], feed)

            # test avx
            nuphar_settings = 'nuphar_cache_path:{}, nuphar_cache_so_name:{}, nuphar_cache_force_no_jit:{}, nuphar_codegen_target:{}'.format(
                cache_dir, so_name, 'on', 'avx')
            sess = onnxrt.InferenceSession(model, providers=make_providers(nuphar_settings))
            sess.run([], feed)

    def test_bert_squad(self):
        cwd = os.getcwd()

        # run symbolic shape inference on this model
        # set int_max to 1,000,000 to simplify symbol computes for things like min(1000000, seq_len) -> seq_len
        bert_squad_dir_src = '/build/models/opset10/BERT_Squad'
        bert_squad_dir = os.path.join(cwd, 'BERT_Squad')
        shutil.copytree(bert_squad_dir_src, bert_squad_dir)

        bert_squad_model = os.path.join(bert_squad_dir, 'bertsquad10.onnx')
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.tools.symbolic_shape_infer', '--input', bert_squad_model, '--output',
            bert_squad_model, '--auto_merge', '--int_max=1000000'
        ],
                       check=True,
                       cwd=cwd)

        # run onnx_test_runner to verify results
        onnx_test_runner = os.path.join(cwd, 'onnx_test_runner')
        subprocess.run([onnx_test_runner, '-e', 'nuphar', '-n', 'BERT_Squad', cwd], check=True, cwd=cwd)

        # run onnxruntime_perf_test, note that nuphar currently is not integrated with ORT thread pool, so set -x 1 to avoid thread confliction with OpenMP
        onnxruntime_perf_test = os.path.join(cwd, 'onnxruntime_perf_test')
        subprocess.run([onnxruntime_perf_test, '-e', 'nuphar', '-x', '1', '-t', '20', bert_squad_model, '1.txt'],
                       check=True,
                       cwd=cwd)

    def test_rnn_benchmark(self):
        # make sure benchmarking scripts works
        # note: quantized model requires AVX2, otherwise it might be slow
        avg_rnn, avg_scan, avg_int8 = perf_test('lstm',
                                                num_threads=1,
                                                input_dim=128,
                                                hidden_dim=1024,
                                                bidirectional=True,
                                                layers=1,
                                                seq_len=16,
                                                batch_size=1,
                                                min_duration_seconds=1)
        avg_rnn, avg_scan, avg_int8 = perf_test('gru',
                                                num_threads=1,
                                                input_dim=128,
                                                hidden_dim=1024,
                                                bidirectional=False,
                                                layers=2,
                                                seq_len=16,
                                                batch_size=3,
                                                min_duration_seconds=1)
        avg_rnn, avg_scan, avg_int8 = perf_test('rnn',
                                                num_threads=1,
                                                input_dim=128,
                                                hidden_dim=1024,
                                                bidirectional=False,
                                                layers=3,
                                                seq_len=16,
                                                batch_size=2,
                                                min_duration_seconds=1)

    def test_batch_scan(self):
        input_dim = 3
        hidden_dim = 5
        bidirectional = False
        layers = 3

        lstm_model_name = 'test_batch_rnn_lstm.onnx'
        # create an LSTM model for generating baseline data
        generate_model('lstm',
                       input_dim,
                       hidden_dim,
                       bidirectional,
                       layers,
                       lstm_model_name,
                       batch_one=False,
                       has_seq_len=True)

        seq_len = 8
        batch_size = 2
        # prepare input
        data_input = (np.random.rand(seq_len, batch_size, input_dim) * 2 - 1).astype(np.float32)
        data_seq_len = np.random.randint(1, seq_len, size=(batch_size,), dtype=np.int32)

        # run lstm as baseline
        sess = onnxrt.InferenceSession(lstm_model_name)
        first_lstm_data_output = sess.run([], {'input': data_input[:, 0:1, :], 'seq_len': data_seq_len[0:1]})

        lstm_data_output = []
        lstm_data_output = first_lstm_data_output

        for b in range(1, batch_size):
            lstm_data_output = lstm_data_output + sess.run([], {
                'input': data_input[:, b:(b + 1), :],
                'seq_len': data_seq_len[b:(b + 1)]
            })
        lstm_data_output = np.concatenate(lstm_data_output, axis=1)

        # generate a batch scan model
        scan_model_name = 'test_batch_rnn_scan.onnx'
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.nuphar.model_editor', '--input', lstm_model_name, '--output',
            scan_model_name, '--mode', 'to_scan'
        ],
                       check=True)

        # run scan_batch with batch size 1
        sess = onnxrt.InferenceSession(scan_model_name)
        scan_batch_data_output = sess.run([], {'input': data_input[:, 0:1, :], 'seq_len': data_seq_len[0:1]})
        assert np.allclose(first_lstm_data_output, scan_batch_data_output)

        # run scan_batch with batch size 2
        scan_batch_data_output = sess.run([], {'input': data_input, 'seq_len': data_seq_len})
        assert np.allclose(lstm_data_output, scan_batch_data_output)

        # run scan_batch with batch size 1 again
        scan_batch_data_output = sess.run([], {'input': data_input[:, 0:1, :], 'seq_len': data_seq_len[0:1]})
        assert np.allclose(first_lstm_data_output, scan_batch_data_output)

    def test_gemm_to_matmul(self):
        gemm_model_name_prefix = "gemm_model"
        matmul_model_name_prefix = "matmul_model"
        common_config = {
            'node_name':'GemmNode',
            'A':'inputA', 'B':'inputB', 'C':'inputC', 'Y':'output', 'M':2, 'K':3, 'N':4, 'withC':False,
            'initA':False, 'initB':False, 'initC':False, 'alpha':1.0, 'beta':1.0, 'transA':0, 'transB':0
        }
        test_configs = [
            {},
            {'transA':1},
            {'transB':1},
            {'transA':1, 'transB':1},
            {'withC':True},
            {'withC':True, 'initC':True},
            {'initA':True},
            {'initB':True},
            {'initA':True, 'initB':True},
            {'initA':True, 'transA':1},
            {'initB':True, 'transB':1},
            {'initA':True, 'transA':1, 'initB':True, 'transB':1},
            {'alpha':2.2},
            {'transA':1, 'alpha':2.2},
            {'initA':True, 'transA':1, 'alpha':2.2},
            {'withC':True, 'beta':3.3},
            {'withC':True, 'initC':True, 'beta':3.3},
            {'initA':True, 'transA':1, 'alpha':2.2, 'withC':True, 'initC':True, 'beta':3.3},
            {'transA':1, 'transB':1, 'alpha':2.2, 'withC':True, 'beta':3.3},
            {'transA':1, 'transB':1, 'alpha':2.2, 'withC':True, 'initC':True, 'beta':3.3}
        ]

        for i, config in enumerate(test_configs):
            running_config = common_config.copy()
            running_config.update(config)
            gemm_model_name = gemm_model_name_prefix+str(i)+'.onnx'
            matmul_model_name = matmul_model_name_prefix+str(i)+'.onnx'
            a, b, c, expected_y = generate_gemm_model(gemm_model_name, running_config)
            subprocess.run([
                sys.executable, '-m', 'onnxruntime.nuphar.model_editor',
                '--input', gemm_model_name,
                '--output', matmul_model_name, '--mode', 'gemm_to_matmul'
            ], check=True)

            sess = onnxrt.InferenceSession(matmul_model_name)
            test_inputs = {}
            set_gemm_model_inputs(running_config, test_inputs, a, b, c)
            actual_y = sess.run([], test_inputs)
            assert np.allclose(expected_y, actual_y)

    def test_gemm_to_matmul_with_scan(self):
        gemm_model_name_prefix = "gemm_scan_model"
        matmul_model_name_prefix = "matmul_scan_model"

        common_config = {
            'M':2, 'K':3, 'N':4, 'withC':False, 'initA':False, 'initB':False, 'initC':False,
            'alpha':1.0, 'beta':1.0, 'transA':0, 'transB':0
        }
        common_config1 = common_config.copy()
        common_config1.update({
            'node_name':'GemmNode1', 'A':'input1A', 'B':'input1B', 'C':'input1C', 'Y':'output1'
        })
        common_config2 = common_config.copy()
        common_config2.update({
            'node_name':'GemmNode2', 'A':'input2A', 'B':'input2B', 'C':'input2C', 'Y':'output2'
        })
        test_configs = [
            ({}, {}),
            ({'transA':1}, {'transB':1}),
            ({'transA':1, 'transB':1}, {'transA':1, 'transB':1}),
            ({'alpha':2.2}, {'alpha':3.3}),
            ({'transA':1, 'transB':1, 'alpha':2.2}, {'transA':1, 'transB':1, 'alpha':3.3}),
            ({'withC':True, 'initC':True}, {}),
            ({'withC':True, 'initC':True}, {'withC':True, 'initC':True}),
            ({'transA':1, 'transB':1, 'alpha':2.2, 'withC':True, 'initC':True, 'beta':1.2},
             {'transA':1, 'transB':1, 'alpha':3.3, 'withC':True, 'initC':True, 'beta':4.1}),
            ({'initA':True}, {}),
            ({'initA':True}, {'initB':True}),
            # FIXME: enable the test below after we fix some likely issue in graph partitioner
            #({'initA':True, 'initB':True}, {}),
            #({'initA':True, 'initB':True, 'transA':1}, {'initA':True, 'transB':1}),
            #({'initA':True, 'transA':1, 'transB':1, 'alpha':2.2},
            # {'initB':True, 'transA':1, 'transB':1, 'alpha':3.3}),
            #({'initA':True, 'transA':1, 'transB':1, 'alpha':2.2, 'withC':True, 'initC':True},
            # {'initB':True, 'transA':1, 'transB':1, 'alpha':3.3}),
            #({'initA':True, 'transA':1, 'transB':1, 'alpha':2.2, 'withC':True, 'initC':True, 'beta':1.2},
            # {'initB':True, 'transA':1, 'transB':1, 'alpha':3.3, 'withC':True, 'initC':True, 'beta':4.2}),
            ({'A':'inputA', 'initA':True}, {'A':'inputA', 'initA':True}),
            ({'B':'inputB', 'initB':True}, {'B':'inputB', 'initB':True}),
            ({'C':'inputC', 'withC':True, 'initC':True}, {'C':'inputC', 'withC':True, 'initC':True}),
            ({'transA':1, 'alpha':1.2, 'B':'inputB', 'initB':True, 'C':'inputC', 'withC':True, 'initC':True},
             {'transA':1, 'alpha':2.2, 'B':'inputB', 'initB':True, 'C':'inputC', 'withC':True, 'initC':True}),
            ({'transB':1, 'alpha':1.2, 'B':'inputB'}, {'transB':1, 'alpha':2.2, 'B':'inputB'}),
            ({'transA':1, 'alpha':1.2, 'A':'inputA', 'B':'inputB'},
             {'transA':1, 'alpha':2.2, 'A':'inputA', 'B':'inputB'}),
            ({'transA':1, 'alpha':1.2, 'A':'inputA', 'B':'inputB', 'C':'inputC1', 'withC':True, 'initC':True},
             {'transA':1, 'alpha':2.2, 'A':'inputA', 'B':'inputB', 'C':'inputC2', 'withC':True, 'initC':True}),
        ]

        for i, config in enumerate(test_configs):
            config1, config2 = config
            running_config1 = common_config1.copy()
            running_config1.update(config1)
            running_config2 = common_config2.copy()
            running_config2.update(config2)

            gemm_model_name = gemm_model_name_prefix+str(i)+'.onnx'
            matmul_model_name = matmul_model_name_prefix+str(i)+'.onnx'
            a1, b1, c1, a2, b2, c2 = generate_gemm_scan_model(gemm_model_name,
                                                              running_config1,
                                                              running_config2)

            a1 = a1.reshape((1, ) + a1.shape)
            b1 = b1.reshape((1, ) + b1.shape)
            a2 = a2.reshape((1, ) + a2.shape)
            b2 = b2.reshape((1, ) + b2.shape)
            sess = onnxrt.InferenceSession(gemm_model_name)

            test_inputs = {}
            set_gemm_model_inputs(running_config1, test_inputs, a1, b1, c1)
            set_gemm_model_inputs(running_config2, test_inputs, a2, b2, c2)

            # run before model editing
            expected_y = sess.run([], test_inputs)

            subprocess.run([
                sys.executable, '-m', 'onnxruntime.nuphar.model_editor',
                '--input', gemm_model_name,
                '--output', matmul_model_name, '--mode', 'gemm_to_matmul'
            ], check=True)

            # run after model editing
            sess = onnxrt.InferenceSession(matmul_model_name)
            actual_y = sess.run([], test_inputs)

            assert np.allclose(expected_y, actual_y, atol=1e-7)
            print("finished " + matmul_model_name)

    def test_loop_to_scan(self):
        loop_model_filename = get_name("nuphar_tiny_model_with_loop_shape_infered.onnx")
        scan_model_filename = "nuphar_tiny_model_with_loop_shape_infered_converted_to_scan.onnx"
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.nuphar.model_editor',
            '--input', loop_model_filename,
            '--output', scan_model_filename, '--mode', 'loop_to_scan'
        ], check=True)

        validate_with_ort(loop_model_filename, scan_model_filename)

    def test_loop_to_scan_with_inconvertible_loop(self):
        # nuphar_onnx_test_loop11_inconvertible_loop.onnx contains a Loop op with dynamic loop count.
        # This Loop op cannot be converted to a Scan op.
        # Set --keep_unconvertible_loop_ops option so conversion will not fail due to unconvertible loop ops.
        loop_model_filename = get_name("nuphar_onnx_test_loop11_inconvertible_loop.onnx")
        scan_model_filename = "nuphar_onnx_test_loop11_inconvertible_loop_unchanged.onnx"
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.nuphar.model_editor',
            '--input', loop_model_filename,
            '--output', scan_model_filename, '--mode', 'loop_to_scan',
            '--keep_unconvertible_loop_ops'
        ], check=True)

        # onnxruntime is failing with:
        # onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 :
        # FAIL : Non-zero status code returned while running Loop node. Name:''
        # Status Message: Inconsistent shape in loop output for output.  Expected:{1} Got:{0}
        # skip validate_with_ort for now
        # validate_with_ort(loop_model_filename, scan_model_filename)

    def test_loop_to_scan_tool(self):
        loop_model_filename = get_name("nuphar_tiny_model_with_loop_shape_infered.onnx")
        scan_model_filename = "nuphar_tiny_model_with_loop_shape_infered_converted_to_scan.onnx"
        subprocess.run([
            sys.executable, '-m', 'onnxruntime.nuphar.model_tools',
            '--input', loop_model_filename,
            '--output', scan_model_filename,
            '--tool', 'convert_loop_to_scan_and_validate',
            '--symbolic_dims', 'sequence=30'
        ], check=True)

        validate_with_ort(loop_model_filename, scan_model_filename)

if __name__ == '__main__':
    unittest.main()
