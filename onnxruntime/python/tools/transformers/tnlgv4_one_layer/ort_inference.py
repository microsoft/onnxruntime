# create an ort session
import onnxruntime as ort
import numpy as np
np.random.seed(0)

onnx_path = "opt/tnlgv4_one_layer_opt.onnx"
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0
ort_session = ort.InferenceSession(onnx_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

batch_size = 1
seq_len = 1024
past_seq_len = 1
total_seq_len = seq_len + past_seq_len

ort_inputs = {ort_session.get_inputs()[0].name: np.random.randint(low=0, high=100351, size=(batch_size, seq_len), dtype=np.int64),
              ort_session.get_inputs()[1].name: np.ones((batch_size, total_seq_len), dtype = np.int32),
              ort_session.get_inputs()[2].name: np.random.randn(2, batch_size, 32, past_seq_len, 128).astype(np.float16),
}

ort_outputs = ort_session.run(None, ort_inputs)
print("logits:", ort_outputs[0])

# TODO: add a test case to compare the logits with the original model
# TODO: use iobinding to benchmark the per-token latency