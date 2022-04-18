from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
import time
import numpy as np

def get_perf(ort_sess, ort_inputs, repeat = 10):
    ort_sess.run(None, ort_inputs)
    t0 = time.time()
    for i in range(repeat):
        print(i)
        ort_sess.run(None, ort_inputs)
    total_time = time.time() - t0
    return total_time / repeat * 1000

def generate_encoder_input(batch, encoder_sequence):
    ort_inputs = {}
    ort_inputs['input_ids'] = np.random.randint(low=0, high=250104 - 1, size=(batch, encoder_sequence), dtype=np.int32)
    ort_inputs['attention_mask'] = np.ones([batch, encoder_sequence], dtype=np.int32)
    return ort_inputs

def generate_decoder_input(batch, decoder_sequence, current_sequence):
    ort_inputs = {}
    ort_inputs['last_hidden_state'] = np.float32(np.random.uniform(-1, 1, (batch, decoder_sequence, 1024)))
    ort_inputs['decoder_input_ids'] = np.random.randint(low=0, high=250104 - 1, size=(batch, current_sequence), dtype=np.int32)
    ort_inputs['attention_mask'] = np.ones([batch, decoder_sequence], dtype=np.int32)
    return ort_inputs

def generate_beam_search_input(batch, encoder_sequence):
    ort_inputs = {
      "input_ids": np.random.randint(low=0, high=250104 - 1, size=(batch, encoder_sequence), dtype=np.int32),
      "max_length": np.array([50], dtype=np.int32),
      "min_length": np.array([1], dtype=np.int32),
      "num_beams": np.array([5], dtype=np.int32),
      "num_return_sequences": np.array([1], dtype=np.int32),
      "temperature": np.array([1], dtype=np.float32),
      "length_penalty": np.array([1], dtype=np.float32),
      "repetition_penalty": np.array([1], dtype=np.float32)
    }
    return ort_inputs

def inference(model_path, ort_inputs):
    sess_options = SessionOptions()
    sess_options.log_severity_level = 4
    sess = InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    perf_number = get_perf(sess, ort_inputs)
    print("Model:", model_path, "CPU", "ave perf time(ms)", perf_number)

encoder_model = 'zcode_encoder/encoder.onnx'
decoder_model = 'zcode_decoder/decoder.onnx'
bs_model = 'zcode_beamsearch/beam_search_zcode.onnx'

inference(bs_model, generate_beam_search_input(1, 128))
inference(encoder_model, generate_encoder_input(1, 128))
for len in range(1):
    inference(decoder_model, generate_decoder_input(1, 128, len + 1))

