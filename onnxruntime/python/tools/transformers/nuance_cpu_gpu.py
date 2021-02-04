import os
import psutil
import time
import numpy as np

onnx_model_path = "./onnx_model/model.onnx"

def get_input(batch_size = 1, seq_length = 256):
    # Bert Inputs
    input_ids = np.random.uniform(1, 10, size=(batch_size, seq_length)).astype('int32')
    input_mask = np.ones((batch_size, seq_length), dtype="int32")
    segment_ids = np.zeros((batch_size, seq_length), dtype="int32")
    # External Feature
    n_external_feature = 250
    external_features = np.random.uniform(1, 10, size=(batch_size, seq_length*n_external_feature)).astype('float32') # resized back to 3 dimensions in the model
    return input_ids, input_mask, segment_ids, external_features

os.environ["OMP_NUM_THREADS"] = str(24)
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import onnxruntime
print(f"ONNX Runtime Version: {onnxruntime.__version__}")

providers=['CPUExecutionProvider']
#providers=['CUDAExecutionProvider']
total_runs = 1000

sess_options = onnxruntime.SessionOptions()
onnx_session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers)


# warm-up
input_ids, input_mask, segment_ids, external_features = get_input(1, 32)
result_0 = onnx_session.run(None, {"x_1:0":input_ids, "x_2:0": input_mask, "x_3:0": segment_ids, "x:0": external_features})



# timing
for batch_size in [1, 32]:
    for seq_length in [32, 64, 128, 256]:
        input_ids, input_mask, segment_ids, external_features = get_input(batch_size, seq_length)

        start = time.time()
        for _ in range(total_runs):
            result = onnx_session.run(None, {"x_1:0":input_ids, "x_2:0": input_mask, "x_3:0": segment_ids, "x:0": external_features})
        end = time.time()

        print("ONNX Inference time for sequence length={} and batch size={} is {} ms".format(seq_length, batch_size, format((end - start) * 1000 / total_runs, '.2f')))
    print("*"*100)

optimized_model_path_fp32 = "./onnx_model/model_fp32.onnx"
import optimizer
optimized_model = optimizer.optimize_model(onnx_model_path, model_type='bert_tf')
optimized_model.save_model_to_file(optimized_model_path_fp32, False)
print(optimized_model.get_fused_operator_statistics())

onnx_session_1 = onnxruntime.InferenceSession(optimized_model_path_fp32, sess_options, providers)

# warm-up
result_1 = onnx_session_1.run(None, {"x_1:0":input_ids, "x_2:0": input_mask, "x_3:0": segment_ids, "x:0": external_features})

# timing
for batch_size in [1, 32]:
    for seq_length in [32, 64, 128, 256]:
        input_ids, input_mask, segment_ids, external_features = get_input(batch_size, seq_length)

        start = time.time()
        for _ in range(total_runs):
            result = onnx_session_1.run(None, {"x_1:0":input_ids, "x_2:0": input_mask, "x_3:0": segment_ids, "x:0": external_features})
        end = time.time()

        print("ONNX Inference time for sequence length={} and batch size={} is {} ms".format(seq_length, batch_size, format((end - start) * 1000 / total_runs, '.2f')))
    print("*"*100)

input_ids, input_mask, segment_ids, external_features = get_input(1, 32)
result_0 = onnx_session.run(None, {"x_1:0":input_ids, "x_2:0": input_mask, "x_3:0": segment_ids, "x:0": external_features})
result_1 = onnx_session_1.run(None, {"x_1:0":input_ids, "x_2:0": input_mask, "x_3:0": segment_ids, "x:0": external_features})
#for i in range(4):
    #print(result_1[i] - result_0[i])
