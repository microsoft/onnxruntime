# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import onnxruntime as onnxrt

class TestNuphar(unittest.TestCase):
    def test_nuphar_scripts(self):
        if not 'NUPHAR' in onnxrt.get_device():
            return

        # make sure benchmarking scripts works
        # note: quantized model requires AVX2, otherwise it might be slow

        from rnn_benchmark import perf_test
        upper_bound = 1.05 # allow some head rooms
        avg_rnn, avg_scan, avg_int8 = perf_test('lstm', num_threads=1,
                                                input_dim=128, hidden_dim=1024, bidirectional=True,
                                                layers=1, seq_len=16, batch_size=1)
        self.assertLess(avg_scan, avg_rnn * upper_bound)
        avg_rnn, avg_scan, avg_int8 = perf_test('gru', num_threads=1,
                                                input_dim=128, hidden_dim=1024, bidirectional=False,
                                                layers=2, seq_len=16, batch_size=3)
        self.assertLess(avg_scan, avg_rnn * upper_bound)
        avg_rnn, avg_scan, avg_int8 = perf_test('rnn', num_threads=1,
                                                input_dim=128, hidden_dim=1024, bidirectional=False,
                                                layers=3, seq_len=16, batch_size=2)
        self.assertLess(avg_scan, avg_rnn * upper_bound)

    def test_nuphar_bidaf(self):
        if not 'NUPHAR' in onnxrt.get_device():
            return

        import subprocess
        from timeit import default_timer as timer
        import os
        import urllib.request
        import tarfile
        cwd = os.getcwd()
        bidaf_url = 'https://onnxzoo.blob.core.windows.net/models/opset_9/bidaf/bidaf.tar.gz'
        bidaf_local = os.path.join(cwd, 'bidaf.tar.gz')
        if not os.path.exists(bidaf_local):
            urllib.request.urlretrieve(bidaf_url, bidaf_local)
            with tarfile.open(bidaf_local, 'r') as f:
                f.extractall(cwd)

        # verify accuracy of quantized model
        from model_editor import convert_to_scan_model
        from model_quantizer import convert_matmul_model
        from symbolic_shape_infer import SymbolicShapeInference
        bidaf_model = os.path.join(cwd, 'bidaf', 'bidaf.onnx')
        bidaf_scan_model = os.path.join(cwd, 'bidaf', 'bidaf_scan.onnx')
        bidaf_int8_scan_only_model = os.path.join(cwd, 'bidaf', 'bidaf_int8_scan_only.onnx')
        convert_to_scan_model(bidaf_model, bidaf_scan_model)
        SymbolicShapeInference.infer_shapes(bidaf_scan_model, bidaf_scan_model)
        convert_matmul_model(bidaf_scan_model, bidaf_int8_scan_only_model, only_for_scan=True)
        SymbolicShapeInference.infer_shapes(bidaf_int8_scan_only_model, bidaf_int8_scan_only_model)
        
        onnx_test_runner = os.path.join(cwd, 'onnx_test_runner')
        subprocess.run([onnx_test_runner, '-e', 'nuphar', '-n', 'bidaf', cwd], check=True, cwd=cwd)

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

        os.environ['NUPHAR_CACHE_PATH'] = cache_dir
        onnxruntime_perf_test = os.path.join(cwd, 'onnxruntime_perf_test')
        non_jit_repeats = 1
        jit_repeats = 10

        start = timer()
        subprocess.run([onnxruntime_perf_test, '-e', 'nuphar', '-r', str(non_jit_repeats), bidaf_int8_scan_only_model, 'bidaf_log.txt'], check=True, cwd=cwd)
        non_jit_time = timer() - start

        cache_dir_content = os.listdir(cache_dir)
        assert len(cache_dir_content) == 1
        cache_versioned_dir = os.path.join(cache_dir, cache_dir_content[0])
        if os.name == 'nt': # Windows
            subprocess.run(['cmd', '/k', os.path.join(cwd, 'create_shared.cmd'), cache_versioned_dir], check=True, cwd=cwd)
        elif os.name == 'posix': #Linux
            subprocess.run(['bash', os.path.join(cwd, 'create_shared.sh'), '-c', cache_versioned_dir], check=True, cwd=cwd)
        else:
            return # don't run the rest of test if AOT is not supported

        start = timer()
        subprocess.run([onnxruntime_perf_test, '-e', 'nuphar', '-r', str(jit_repeats), bidaf_int8_scan_only_model, 'bidaf_log.txt'], check=True)
        jit_time = timer() - start
        self.assertLess(jit_time, non_jit_time)
        del os.environ['NUPHAR_CACHE_PATH']

if __name__ == '__main__':
    unittest.main()
