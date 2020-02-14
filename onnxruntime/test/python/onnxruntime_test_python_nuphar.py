# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as onnxrt
import os
from onnxruntime.nuphar.rnn_benchmark import perf_test, generate_model
from pathlib import Path
import shutil
import sys
import subprocess
import tarfile
import unittest
import urllib.request

class TestNuphar(unittest.TestCase):
    def test_bidaf(self):
        # download BiDAF model
        cwd = os.getcwd()
        bidaf_url = 'https://onnxzoo.blob.core.windows.net/models/opset_9/bidaf/bidaf.tar.gz'
        cache_dir = os.path.join(os.path.expanduser("~"), '.cache','onnxruntime')
        os.makedirs(cache_dir, exist_ok=True)
        bidaf_local = os.path.join(cache_dir, 'bidaf.tar.gz')
        if not os.path.exists(bidaf_local):
            urllib.request.urlretrieve(bidaf_url, bidaf_local)
        with tarfile.open(bidaf_local, 'r') as f:
            f.extractall(cwd)

        # verify accuracy of quantized model
        bidaf_dir = os.path.join(cwd, 'bidaf')
        bidaf_model = os.path.join(bidaf_dir, 'bidaf.onnx')
        bidaf_scan_model = os.path.join(bidaf_dir, 'bidaf_scan.onnx')
        bidaf_opt_scan_model = os.path.join(bidaf_dir, 'bidaf_opt_scan.onnx')
        bidaf_int8_scan_only_model = os.path.join(bidaf_dir, 'bidaf_int8_scan_only.onnx')
        subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.model_editor', '--input', bidaf_model, '--output', bidaf_scan_model, '--mode', 'to_scan'], check=True, cwd=cwd)
        subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.model_editor', '--input', bidaf_scan_model, '--output', bidaf_opt_scan_model, '--mode', 'opt_inproj'], check=True, cwd=cwd)
        subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.model_quantizer', '--input', bidaf_opt_scan_model, '--output', bidaf_int8_scan_only_model, '--only_for_scan'], check=True, cwd=cwd)

        # run onnx_test_runner to verify results
        # use -M to disable memory pattern
        onnx_test_runner = os.path.join(cwd, 'onnx_test_runner')
        subprocess.run([onnx_test_runner, '-e', 'nuphar', '-M', '-n', 'bidaf', cwd], check=True, cwd=cwd)

        # test AOT on the quantized model
        if os.name not in ['nt', 'posix']:
            return # don't run the rest of test if AOT is not supported

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
                onnxrt.capi._pybind_state.set_nuphar_settings(nuphar_settings + ', nuphar_codegen_target:' + isa)
                sess = onnxrt.InferenceSession(model) # JIT cache happens when initializing session

            cache_dir_content = os.listdir(cache_dir)
            assert len(cache_dir_content) == 1
            cache_versioned_dir = os.path.join(cache_dir, cache_dir_content[0])
            so_name = os.path.basename(model) + '.so'
            subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.create_shared', '--input_dir', cache_versioned_dir, '--output_name', so_name], check=True)

            nuphar_settings = 'nuphar_cache_path:{}, nuphar_cache_so_name:{}, nuphar_cache_force_no_jit:{}'.format(cache_dir, so_name, 'on')
            onnxrt.capi._pybind_state.set_nuphar_settings(nuphar_settings)
            sess = onnxrt.InferenceSession(model)
            sess.run([], feed)

            # test avx
            nuphar_settings = 'nuphar_cache_path:{}, nuphar_cache_so_name:{}, nuphar_cache_force_no_jit:{}, nuphar_codegen_target:{}'.format(cache_dir, so_name, 'on', 'avx')
            onnxrt.capi._pybind_state.set_nuphar_settings(nuphar_settings)
            sess = onnxrt.InferenceSession(model)
            sess.run([], feed)


    def test_bert_squad(self):
        # download BERT_squad model
        cwd = os.getcwd()
        bert_squad_url = 'https://onnxzoo.blob.core.windows.net/models/opset_10/bert_squad/download_sample_10.tar.gz'
        cache_dir = os.path.join(os.path.expanduser("~"), '.cache','onnxruntime')
        os.makedirs(cache_dir, exist_ok=True)
        bert_squad_local = os.path.join(cache_dir, 'bert_squad.tar.gz')
        if not os.path.exists(bert_squad_local):
            urllib.request.urlretrieve(bert_squad_url, bert_squad_local)
        with tarfile.open(bert_squad_local, 'r') as f:
            f.extractall(cwd)

        # run symbolic shape inference on this model
        # set int_max to 1,000,000 to simplify symbol computes for things like min(1000000, seq_len) -> seq_len
        bert_squad_dir = os.path.join(cwd, 'download_sample_10')
        bert_squad_model = os.path.join(bert_squad_dir, 'bertsquad10.onnx')
        subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.symbolic_shape_infer', '--input', bert_squad_model, '--output', bert_squad_model, '--auto_merge', '--int_max=1000000'], check=True, cwd=cwd)

        # run onnx_test_runner to verify results
        onnx_test_runner = os.path.join(cwd, 'onnx_test_runner')
        subprocess.run([onnx_test_runner, '-e', 'nuphar', '-n', 'download_sample_10', cwd], check=True, cwd=cwd)

        # run onnxruntime_perf_test, note that nuphar currently is not integrated with ORT thread pool, so set -x 1 to avoid thread confliction with OpenMP
        onnxruntime_perf_test = os.path.join(cwd, 'onnxruntime_perf_test')
        subprocess.run([onnxruntime_perf_test, '-e', 'nuphar', '-x', '1', '-t', '20', bert_squad_model, '1.txt'], check=True, cwd=cwd)


    def test_rnn_benchmark(self):
        # make sure benchmarking scripts works
        # note: quantized model requires AVX2, otherwise it might be slow
        avg_rnn, avg_scan, avg_int8 = perf_test('lstm', num_threads=1,
                                                input_dim=128, hidden_dim=1024, bidirectional=True,
                                                layers=1, seq_len=16, batch_size=1,
                                                min_duration_seconds=1)
        avg_rnn, avg_scan, avg_int8 = perf_test('gru', num_threads=1,
                                                input_dim=128, hidden_dim=1024, bidirectional=False,
                                                layers=2, seq_len=16, batch_size=3,
                                                min_duration_seconds=1)
        avg_rnn, avg_scan, avg_int8 = perf_test('rnn', num_threads=1,
                                                input_dim=128, hidden_dim=1024, bidirectional=False,
                                                layers=3, seq_len=16, batch_size=2,
                                                min_duration_seconds=1)


    def test_batch_scan(self):
        input_dim = 3
        hidden_dim = 5
        bidirectional = False
        layers = 3
 
        lstm_model_name = 'test_batch_rnn_lstm.onnx'
        # create an LSTM model for generating baseline data
        generate_model('lstm', input_dim, hidden_dim, bidirectional, layers, lstm_model_name, batch_one=False, has_seq_len=True)

        seq_len = 8
        batch_size = 2
        # prepare input
        data_input = (np.random.rand(seq_len, batch_size, input_dim) * 2 - 1).astype(np.float32)
        data_seq_len = np.random.randint(1, seq_len, size=(batch_size,), dtype=np.int32)

        # run lstm as baseline
        sess = onnxrt.InferenceSession(lstm_model_name)
        first_lstm_data_output = sess.run([], {'input':data_input[:,0:1,:], 'seq_len':data_seq_len[0:1]})

        lstm_data_output = []
        lstm_data_output = first_lstm_data_output

        for b in range(1, batch_size):
            lstm_data_output = lstm_data_output + sess.run([], {'input':data_input[:,b:(b+1),:], 'seq_len':data_seq_len[b:(b+1)]})
        lstm_data_output = np.concatenate(lstm_data_output, axis=1)

        # generate a batch scan model
        scan_model_name = 'test_batch_rnn_scan.onnx'
        subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.model_editor', '--input', lstm_model_name, '--output', scan_model_name, '--mode', 'to_scan'], check=True)

        # run scan_batch with batch size 1
        sess = onnxrt.InferenceSession(scan_model_name)
        scan_batch_data_output = sess.run([], {'input':data_input[:,0:1,:], 'seq_len':data_seq_len[0:1]})
        assert np.allclose(first_lstm_data_output, scan_batch_data_output)

        # run scan_batch with batch size 2
        scan_batch_data_output = sess.run([], {'input':data_input, 'seq_len':data_seq_len})
        assert np.allclose(lstm_data_output, scan_batch_data_output)

        # run scan_batch with batch size 1 again
        scan_batch_data_output = sess.run([], {'input':data_input[:,0:1,:], 'seq_len':data_seq_len[0:1]})
        assert np.allclose(first_lstm_data_output, scan_batch_data_output)


    def test_symbolic_shape_infer(self):
        cwd = os.getcwd()
        test_model_dir = os.path.join(cwd, '..', 'models')
        for filename in Path(test_model_dir).rglob('*.onnx'):
            if filename.name.startswith('.'):
                continue # skip some bad model files
            subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.symbolic_shape_infer', '--input', str(filename), '--auto_merge', '--int_max=100000', '--guess_output_rank'], check=True, cwd=cwd)


if __name__ == '__main__':
    unittest.main()
