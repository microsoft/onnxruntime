# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as onnxrt
import os
from onnxruntime.nuphar.rnn_benchmark import perf_test
import sys
import subprocess
import tarfile
from timeit import default_timer as timer
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
        # use -j 1 -c 1 to run one model/session at a time when running multiple models
        onnx_test_runner = os.path.join(cwd, 'onnx_test_runner')
        subprocess.run([onnx_test_runner, '-e', 'nuphar', '-M', '-c', '1', '-j', '1', '-n', 'bidaf', cwd], check=True, cwd=cwd)

        # test AOT on the quantized model
        cache_dir = os.path.join(cwd, 'nuphar_cache')
        if os.path.exists(cache_dir):
            for sub_dir in os.listdir(cache_dir):
                full_sub_dir = os.path.join(cache_dir, sub_dir)
                if os.path.isdir(full_sub_dir):
                    for f in os.listdir(full_sub_dir):
                        os.remove(os.path.join(full_sub_dir, f))
        else:
            os.makedirs(cache_dir)

        # prepare feed
        feed = {}
        for i in range(4):
            tp = onnx.load_tensor(os.path.join(bidaf_dir, 'test_data_set_0', 'input_{}.pb'.format(i)))
            feed[tp.name] = numpy_helper.to_array(tp)

        # force codegen_target to be avx
        nuphar_settings = 'nuphar_codegen_target:avx'
        onnxrt.capi._pybind_state.set_nuphar_settings(nuphar_settings)
        sess = onnxrt.InferenceSession(bidaf_int8_scan_only_model)
        assert 'NupharExecutionProvider' in sess.get_providers()
        output = sess.run([], feed)

        nuphar_settings = 'nuphar_cache_path:{}'.format(cache_dir)
        onnxrt.capi._pybind_state.set_nuphar_settings(nuphar_settings)
        sess = onnxrt.InferenceSession(bidaf_int8_scan_only_model) # JIT cache happens when initializing session
        assert 'NupharExecutionProvider' in sess.get_providers()
        output = sess.run([], feed)

        cache_dir_content = os.listdir(cache_dir)
        assert len(cache_dir_content) == 1
        cache_versioned_dir = os.path.join(cache_dir, cache_dir_content[0])
        so_name = 'bidaf.so'
        if os.name in ['nt', 'posix'] : # Windows or Linux
            subprocess.run([sys.executable, '-m', 'onnxruntime.nuphar.create_shared', '--input_dir', cache_versioned_dir, '--output_name', so_name], check=True)
        else:
            return # don't run the rest of test if AOT is not supported

        nuphar_settings = 'nuphar_cache_path:{}, nuphar_cache_so_name:{}, nuphar_cache_force_no_jit:{}'.format(cache_dir, so_name, 'on')
        onnxrt.capi._pybind_state.set_nuphar_settings(nuphar_settings)
        sess = onnxrt.InferenceSession(bidaf_int8_scan_only_model) # JIT cache happens when initializing session
        assert 'NupharExecutionProvider' in sess.get_providers()
        sess.run([], feed)


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

if __name__ == '__main__':
    unittest.main()
