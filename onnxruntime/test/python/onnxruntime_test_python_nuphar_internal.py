# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
from model_editor_internal import PairDescription
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as onnxrt
import os
from rnn_benchmark import generate_model
import shutil
import subprocess
import sys
import unittest

# list of internal models with test data
# CNTK models can be converted using onnxruntime/core/providers/nuphar/scripts/cntk_converter.py
tests = {'https://lotus.blob.core.windows.net/internal-models/model.lstm.90.8bit.zip' : '68188753f9bb63ef269fca26bc4ca48f',
         'https://lotus.blob.core.windows.net/internal-models/model.lt.lcblstm.simplified.cntk.8bit.zip' : '15afada6e03a053562510d7fb6a38982',
         'https://lotus.blob.core.windows.net/internal-models/model.lstm.se.40.converted3.zip' : '546d33e5da4b6be085304822c3de6f6c',
         'https://lotus.blob.core.windows.net/internal-models/model.dbn.am.zip' : '0964d42a1ae8086e15f6e30f78b674c8',
         'https://lotus.blob.core.windows.net/internal-models/bert_tiny.zip' : 'c2b7a28f3cff55f529dee554f4367b5d'}

cwd = os.getcwd()
unzip_base_dir = os.path.join(cwd, 'internal-models')
download_models_dir = os.path.join(unzip_base_dir, 'models')
models_dir = os.path.join(unzip_base_dir, 'testcases')

def prepare_test_data():
    # use download_test_data from build.py
    # To test locally without downloading, please put zip file under following folder:
    #   On Windows: %USERPROFILE%\.cache\onnxruntime
    #   On Linux: ~/.cache/onnxruntime
    # Test models would be unzipped to <cwd>/internal-models/models/
    # 7z.exe (Windows) or unzip (Linux) should be added to PATH for unzipping
    assert 'BUILD_PY_PATH' in os.environ
    shutil.copyfile(os.environ['BUILD_PY_PATH'], os.path.join(cwd, 'build_py.py'))
    from build_py import download_test_data

    assert 'AZURE_SAS_KEY' in os.environ
    azure_sas_key = os.environ['AZURE_SAS_KEY']
    shutil.rmtree(models_dir, ignore_errors=True)
    os.makedirs(models_dir, exist_ok=True)
    for url, checksum in tests.items():
        download_test_data(unzip_base_dir, url, checksum, azure_sas_key)
        # note that download_test_data removes download_models_dir before extract
        # so move the models to a different folder
        for model_name in os.listdir(download_models_dir):
            shutil.move(os.path.join(download_models_dir, model_name), models_dir)

def generate_random_feeds(mp, seq_len, batch_size):
    from symbolic_shape_infer import get_shape_from_type_proto
    graph_initializers = [i.name for i in mp.graph.initializer]
    real_inputs = [i for i in mp.graph.input if i.name not in graph_initializers]
    feeds = {}
    np.random.seed(42)
    for i, vi in enumerate(real_inputs):
        shape = [batch_size if d == 'batch' else (seq_len if type(d) == str else d) for d in get_shape_from_type_proto(vi.type)]
        dtype = vi.type.tensor_type.elem_type
        assert dtype == onnx.TensorProto.FLOAT
        feeds[vi.name] = np.random.rand(*shape).astype(np.float32)
    return feeds

def concatenate_on_batch(name_and_values, batched_name_and_values):
    for name, value in name_and_values.items():
        if name in batched_name_and_values:
            batched_name_and_values[name] = np.concatenate([batched_name_and_values[name], value], axis=-2)
        else:
            batched_name_and_values[name] = value

class TestNupharInternal(unittest.TestCase):
    def test_0_0_truncated_batch(self):
        # generate 2-layer unidirectional LSTM models
        input_dim = 3
        hidden_dim = 5
        bidirectional = False
        layers = 2
        lstm_no_batch_model = 'lstm_no_batch.onnx'
        scan_no_batch_model = 'scan_no_batch.onnx'
        scan_batch_model = 'scan_batch.onnx'
        scan_fixed_batch_model = 'scan_fixed_batch.onnx'
        scan_batch_truncated_model = 'scan_batch_truncated.onnx'
        scan_fixed_batch_truncated_model = 'scan_fixed_batch_truncated.onnx'

        batch_size = 4

        generate_model('lstm', input_dim, hidden_dim, bidirectional, layers, lstm_no_batch_model, batch_one=True, has_seq_len=True)
        subprocess.run([sys.executable, 'model_editor.py', '--input', lstm_no_batch_model, '--output', scan_no_batch_model, '--mode', 'to_scan'], check=True)
        subprocess.run([sys.executable, 'model_editor_internal.py', '--input', scan_no_batch_model, '--output', scan_batch_model, '--mode', 'add_batch'], check=True)
        subprocess.run([sys.executable, 'model_editor_internal.py', '--input', scan_no_batch_model, '--output', scan_batch_truncated_model, '--mode', 'add_truncated_batch'], check=True)
        subprocess.run([sys.executable, 'model_editor_internal.py', '--input', scan_no_batch_model, '--output', scan_fixed_batch_model, '--mode', 'add_batch', '--batch_size', str(batch_size)], check=True)
        subprocess.run([sys.executable, 'model_editor_internal.py', '--input', scan_no_batch_model, '--output', scan_fixed_batch_truncated_model, '--mode', 'add_truncated_batch', '--batch_size', str(batch_size)], check=True)

        seq_len = 8
        data_input = (np.random.rand(seq_len, batch_size, input_dim) * 2 - 1).astype(np.float32)
        data_seq_len = np.random.randint(1, seq_len, size=(batch_size,), dtype=np.int32)

        # run lstm as baseline
        sess = onnxrt.InferenceSession(lstm_no_batch_model)
        lstm_data_output = []
        for b in range(batch_size):
            lstm_data_output = lstm_data_output + sess.run([], {'input':data_input[:,b:(b+1),:], 'seq_len':data_seq_len[b:(b+1)]})
        lstm_data_output = np.concatenate(lstm_data_output, axis=1)

        # run scan_no_batch, compare baseline
        sess = onnxrt.InferenceSession(scan_no_batch_model)
        scan_no_batch_data_output = []
        for b in range(batch_size):
            scan_no_batch_data_output = scan_no_batch_data_output + sess.run([], {'input':data_input[:,b:(b+1),:], 'seq_len':data_seq_len[b:(b+1)]})
        scan_no_batch_data_output = np.concatenate(scan_no_batch_data_output, axis=1)
        assert np.allclose(lstm_data_output, scan_no_batch_data_output)

        # run scan_batch
        for m in [scan_batch_model, scan_fixed_batch_model]:
            sess = onnxrt.InferenceSession(m)
            scan_batch_data_output = sess.run([], {'input':data_input, 'seq_len':data_seq_len})
            assert np.allclose(lstm_data_output, scan_batch_data_output)

        # run scan batch truncated
        for m in [scan_batch_truncated_model, scan_fixed_batch_truncated_model]:
            sess = onnxrt.InferenceSession(m)
            truncated_seq_lens = [2,3,3]
            assert sum(truncated_seq_lens) == seq_len
            seq_offset = [0]
            for tsl in truncated_seq_lens:
                seq_offset.append(seq_offset[-1] + tsl)
            input_names = [i.name for i in sess.get_inputs()]
            num_scan_states = layers*2 # LSTM has two scan states each layer
            scan_states = [np.zeros((batch_size,hidden_dim), dtype=np.float32)]*num_scan_states
            scan_batch_data_output = []
            for i,tsl in enumerate(truncated_seq_lens):
                feed = []
                feed.append(data_input[seq_offset[i]:seq_offset[i+1],...]) # input
                feed.append(np.clip(data_seq_len - seq_offset[i], 0, tsl).astype(np.int32)) # seq_len
                feed = feed + scan_states
                feed.append(np.asarray([i == 0]*batch_size, dtype=np.bool).reshape(-1,1)) # ResetSequence
                truncated_output = sess.run([], dict(zip(input_names, feed)))
                scan_batch_data_output = scan_batch_data_output + [truncated_output[0]]
                scan_states = truncated_output[1:]
            scan_batch_data_output = np.concatenate(scan_batch_data_output, axis=0)
            assert np.allclose(lstm_data_output, scan_batch_data_output)

    def test_0_1_latency_control(self):
        # generate 2-layer unidirectional LSTM models
        input_dim = 3
        hidden_dim = 5
        bidirectional = False
        layers = 2
        lstm_model = 'test_lstm.onnx'
        scan_model = 'test_lstm_scan.onnx'
        lc_scan_model = 'test_lstm_lc.onnx'

        generate_model('lstm', input_dim, hidden_dim, bidirectional, layers, lstm_model, batch_one=True, has_seq_len=False)
        subprocess.run([sys.executable, 'model_editor.py', '--input', lstm_model, '--output', scan_model, '--mode', 'to_scan'], check=True)
        subprocess.run([sys.executable, 'model_editor_internal.py', '--input', scan_model, '--output', scan_model, '--mode', 'enable_truncated'], check=True)
        subprocess.run([sys.executable, 'model_editor_internal.py', '--input', scan_model, '--output', lc_scan_model, '--mode', 'add_lc'], check=True)

        seq_len = 8
        data_input = (np.random.rand(seq_len, 1, input_dim) * 2 - 1).astype(np.float32)

        # run scan as baseline for full sequence
        sess = onnxrt.InferenceSession(scan_model)
        lstm_data_output = sess.run([], {'input':data_input})[0].reshape(seq_len,1,hidden_dim)

        # run truncated
        scan_mp = onnx.load(scan_model)
        pair_desc = PairDescription()
        pair_desc.parse_from_string(scan_mp.graph.doc_string)
        state_output_input_pairs = pair_desc.get_pairs(PairDescription.PairType.output_state_2_input_state)
        truncated_cfgs = ((0, 3), (3, 6), (6, 8)) # start/end
        feeds = {}
        output_names = [o.name for o in sess.get_outputs()]
        for (start, end) in truncated_cfgs:
            feeds.update({'input':data_input[start:end]})
            scan_outputs = sess.run([], feeds)
            for output_idx, output_name in enumerate(output_names):
                if output_idx == 0:  # skip output[0] in feeds as it's not state
                    continue
                feeds[state_output_input_pairs[output_name]] = scan_outputs[output_idx]
            assert np.allclose(scan_outputs[0], lstm_data_output[start:end])

        # run lc_scan, and compare baseline
        sess = onnxrt.InferenceSession(lc_scan_model)
        lc_mp = onnx.load(lc_scan_model)
        pair_desc = PairDescription()
        pair_desc.parse_from_string(lc_mp.graph.doc_string)
        state_output_input_pairs = pair_desc.get_pairs(PairDescription.PairType.output_state_2_input_state)

        # LC = seq_len - 1, the same as truncated
        feeds = {}
        output_names = [o.name for o in sess.get_outputs()]
        for (start, end) in truncated_cfgs:
            feeds['input'] = data_input[start:end]
            feeds['FutureContextLength_LC_Position'] = np.full((), end - start - 1, dtype=np.int32)
            scan_outputs = sess.run([], feeds)
            for output_idx, output_name in enumerate(output_names):
                if output_idx == 0:  # skip output[0] in feeds as it's not state
                    continue
                feeds[state_output_input_pairs[output_name]] = scan_outputs[output_idx]
            assert np.allclose(scan_outputs[0], lstm_data_output[start:end])

        # start/end/lc_pos for runs with non-zero LC
        lc_cfgs = ((0, 4, 1),
                   (2, 6, 1),
                   (4, 8, 1))

        feeds = {}
        output_names = [o.name for o in sess.get_outputs()]
        for (start, end, lc_pos) in lc_cfgs:
            feeds.update({'input':data_input[start:end], 'FutureContextLength_LC_Position':np.full((), lc_pos, dtype=np.int32)})
            lc_outputs = sess.run([], feeds)
            for output_idx, output_name in enumerate(output_names):
                if output_idx == 0:  # skip output[0] in feeds as it's not state
                    continue
                feeds[state_output_input_pairs[output_name]] = lc_outputs[output_idx]
            assert np.allclose(lc_outputs[0], lstm_data_output[start:end])

    def test_1_basic(self):
        models_to_test = os.listdir(models_dir)
        models_to_test.remove('model.dbn.am') # this model has accuracy mismatch on big negative numbers like -77, needs further investigation
        for model_name in models_to_test:
            subprocess.run([os.path.join(cwd, 'onnx_test_runner'), '-e', 'nuphar', '-n', model_name, models_dir], cwd=cwd, check=True)

    def test_2_add_batch(self):
        models_to_test = os.listdir(models_dir)
        models_to_test.remove('bert_tiny') # BERT model already have batch
        for model_name in models_to_test:
            model_dir = os.path.join(models_dir, model_name)
            assert os.path.isdir(model_dir)
            base_model = os.path.join(model_dir, 'model.onnx')
            seq_len = 40
            batch_size = 8
            batch_feeds = {}
            batch_outputs = {}
            base_mp = onnx.load(base_model)
            base_sess = onnxrt.InferenceSession(base_model)
            for i in range(batch_size):
                feeds = generate_random_feeds(base_mp, seq_len, 1)
                outputs = dict(zip([o.name for o in base_mp.graph.output], base_sess.run([], feeds)))
                concatenate_on_batch(feeds, batch_feeds)
                concatenate_on_batch(outputs, batch_outputs)

            batch_model = os.path.join(model_dir, 'model_batch.onnx')
            subprocess.run([sys.executable, 'model_editor_internal.py', '--input', base_model, '--output', batch_model, '--mode', 'add_batch'], check=True)
            batch_sess = onnxrt.InferenceSession(batch_model)
            real_outputs = batch_sess.run([], batch_feeds)
            # compare probability from LogScaledLikelihood
            def _softmax(arr):
                return np.exp(arr - np.amax(arr))
            for i,o in enumerate(base_mp.graph.output):
                results_match = np.allclose(_softmax(real_outputs[i]), _softmax(batch_outputs[o.name]), rtol=5e-5, atol=5e-5)
                if not results_match:
                    print('max abs error: ', np.amax(np.abs(_softmax(real_outputs[i]) - _softmax(batch_outputs[o.name]))))
                    assert results_match

    def test_3_add_lc_batch_quantize(self):
        for model_name in ['model.lstm.90.8bit', 'model.lt.lcblstm.simplified.cntk.8bit']:
            model_dir = os.path.join(models_dir, model_name)
            assert os.path.isdir(model_dir)
            base_model = os.path.join(model_dir, 'model.onnx')
            lc_model = os.path.join(model_dir, 'model_lc.onnx')
            lc_int8_model = os.path.join(model_dir, 'model_lc_int8.onnx')
            batch_lc_model = os.path.join(model_dir, 'model_lc_batch.onnx')
            batch_lc_int8_model = os.path.join(model_dir, 'model_lc_batch_int8.onnx')
            subprocess.run([sys.executable, 'model_editor_internal.py', '--input', base_model, '--output', lc_model, '--mode', 'add_lc'], check=True)
            subprocess.run([sys.executable, 'model_quantizer.py', '--input', lc_model, '--output', lc_int8_model], check=True)
            subprocess.run([sys.executable, 'model_editor_internal.py', '--input', lc_model, '--output', batch_lc_model, '--mode', 'add_truncated_batch'], check=True)
            subprocess.run([sys.executable, 'model_quantizer.py', '--input', batch_lc_model, '--output', batch_lc_int8_model], check=True)

            # check pairs and inputs/outputs for consistency
            base_mp = onnx.load(base_model)
            final_mp = onnx.load(batch_lc_int8_model)
            pair_desc = PairDescription()
            pair_desc.parse_from_string(final_mp.graph.doc_string)
            state_output_to_input_pairs = pair_desc.get_pairs(PairDescription.PairType.output_state_2_input_state)
            past_value_output_to_input_pairs = pair_desc.get_pairs(PairDescription.PairType.past_value_output_2_past_value_input)
            # make sure FutureContextLength is in Output->Input pairs
            assert 'FutureContextLength' in state_output_to_input_pairs
            # pairs should be 1-to-1 mapping
            assert len(state_output_to_input_pairs) == len(set(state_output_to_input_pairs.values()))
            assert len(past_value_output_to_input_pairs) == len(set(past_value_output_to_input_pairs.values()))
            # inputs other than ResetSequence, FutureContextLength_LC_Position and original ones, should be in pairs
            # note not count in 'FutureContextLength' in pairs
            original_initializers = [i.name for i in base_mp.graph.initializer]
            num_original_required_inputs = len([vi for vi in base_mp.graph.input if vi.name not in original_initializers])
            num_additional_inputs = 2 #ResetSequence, FutureContextLength_LC_Position
            assert len(state_output_to_input_pairs) - 1 + len(past_value_output_to_input_pairs) == len(final_mp.graph.input) - num_additional_inputs - num_original_required_inputs
            for o, i in list(state_output_to_input_pairs.items()) + list(past_value_output_to_input_pairs.items()):
                if o == 'FutureContextLength':
                    continue
                i = i.split(':')[0]
                assert [vi for vi in final_mp.graph.input if vi.name == i]
                assert [vi for vi in final_mp.graph.output if vi.name == o]

            # only test if the model can be converted and loaded
            onnxrt.InferenceSession(lc_int8_model)
            onnxrt.InferenceSession(batch_lc_int8_model)

    def test_4_add_past_value_batch_quantize(self):
        for model_name in ['model.lstm.se.40.converted3', 'model.dbn.am']:
            model_dir = os.path.join(models_dir, model_name)
            assert os.path.isdir(model_dir)
            base_model = os.path.join(model_dir, 'model.onnx')
            past_value_model = os.path.join(model_dir, 'model_pv.onnx')
            past_value_int8_model = os.path.join(model_dir, 'model_pv_int8.onnx')
            batch_model = os.path.join(model_dir, 'model_batch.onnx')
            batch_int8_model = os.path.join(model_dir, 'model_batch_int8.onnx')
            subprocess.run([sys.executable, 'model_editor_internal.py', '--input', base_model, '--output', past_value_model, '--mode', 'enable_truncated'], check=True)
            subprocess.run([sys.executable, 'model_quantizer.py', '--input', past_value_model, '--output', past_value_int8_model], check=True)
            subprocess.run([sys.executable, 'model_editor_internal.py', '--input', past_value_model, '--output', batch_model, '--mode', 'add_truncated_batch'], check=True)
            subprocess.run([sys.executable, 'model_quantizer.py', '--input', batch_model, '--output', batch_int8_model], check=True)

            # check pairs and inputs/outputs for consistency
            base_mp = onnx.load(base_model)
            final_mp = onnx.load(batch_int8_model)
            pair_desc = PairDescription()
            pair_desc.parse_from_string(final_mp.graph.doc_string)
            state_output_to_input_pairs = pair_desc.get_pairs(PairDescription.PairType.output_state_2_input_state)
            past_value_output_to_input_pairs = pair_desc.get_pairs(PairDescription.PairType.past_value_output_2_past_value_input)
            # pairs should be 1-to-1 mapping
            assert len(state_output_to_input_pairs) == len(set(state_output_to_input_pairs.values()))
            assert len(past_value_output_to_input_pairs) == len(set(past_value_output_to_input_pairs.values()))
            # inputs other than ResetSequence and original ones, should be in pairs
            original_initializers = [i.name for i in base_mp.graph.initializer]
            num_original_required_inputs = len([vi for vi in base_mp.graph.input if vi.name not in original_initializers])
            num_additional_inputs = 1 # ResetSequence
            assert len(state_output_to_input_pairs) + len(past_value_output_to_input_pairs) == len(final_mp.graph.input) - num_additional_inputs - num_original_required_inputs
            for o, i in list(state_output_to_input_pairs.items()) + list(past_value_output_to_input_pairs.items()):
                i = i.split(':')[0]
                assert [vi for vi in final_mp.graph.input if vi.name == i]
                assert [vi for vi in final_mp.graph.output if vi.name == o]

            # only test if the model can be converted and loaded
            onnxrt.InferenceSession(past_value_int8_model)
            onnxrt.InferenceSession(batch_int8_model)

if __name__ == '__main__':
    prepare_test_data()
    unittest.main()
