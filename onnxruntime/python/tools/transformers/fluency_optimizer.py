import numpy as np

def generate_input_data_greedy():
    import onnx
    from onnx import numpy_helper
    # Load a TensorProto
    tensor_0 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_greedy/test_data_set_0/input_0.pb', 'rb') as f:
        tensor_0.ParseFromString(f.read())
    tensor_1 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_greedy/test_data_set_0/input_1.pb', 'rb') as f:
        tensor_1.ParseFromString(f.read())
    tensor_2 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_greedy/test_data_set_0/input_2.pb', 'rb') as f:
        tensor_2.ParseFromString(f.read())
    tensor_3 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_greedy/test_data_set_0/input_3.pb', 'rb') as f:
        tensor_3.ParseFromString(f.read())
    # Convert the TensorProto to a Numpy array
    array_0 = numpy_helper.to_array(tensor_0)
    array_1 = numpy_helper.to_array(tensor_1)
    array_2 = numpy_helper.to_array(tensor_2)
    array_3 = numpy_helper.to_array(tensor_3)
    ort_inputs = {}
    ort_inputs['input_ids'] = array_0
    ort_inputs['attention_mask'] = array_1
    ort_inputs['max_length'] = array_2
    ort_inputs['decoder_start_token_id'] = array_3
    return ort_inputs

def generate_input_data_beam():
    import onnx
    from onnx import numpy_helper
    # Load a TensorProto
    tensor_0 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_beam/test_data_set_0/input_0.pb', 'rb') as f:
        tensor_0.ParseFromString(f.read())
    tensor_1 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_beam/test_data_set_0/input_1.pb', 'rb') as f:
        tensor_1.ParseFromString(f.read())
    tensor_2 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_beam/test_data_set_0/input_2.pb', 'rb') as f:
        tensor_2.ParseFromString(f.read())
    tensor_3 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_beam/test_data_set_0/input_3.pb', 'rb') as f:
        tensor_3.ParseFromString(f.read())
    tensor_4 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/new_model_0804/test_bart_beam/test_data_set_0/input_4.pb', 'rb') as f:
        tensor_4.ParseFromString(f.read())
    # Convert the TensorProto to a Numpy array
    array_0 = numpy_helper.to_array(tensor_0)
    array_1 = numpy_helper.to_array(tensor_1)
    array_2 = numpy_helper.to_array(tensor_2)
    array_3 = numpy_helper.to_array(tensor_3)
    array_4 = numpy_helper.to_array(tensor_4)
    ort_inputs = {}
    ort_inputs['input_ids'] = array_0
    ort_inputs['attention_mask'] = array_1
    ort_inputs['num_beams'] = array_2
    ort_inputs['max_length'] = array_3
    ort_inputs['decoder_start_token_id'] = array_4
    return ort_inputs

onnx_model_path_greedy = "/bert_ort/wy/Fluency/new_model_0804/test_bart_greedy/model_bart_greedy.onnx"
onnx_model_path_greedy_o = "/bert_ort/wy/Fluency/new_model_0804/test_bart_greedy/model_bart_greedy_o.onnx"

onnx_model_path_beam = "/bert_ort/wy/Fluency/new_model_0804/test_bart_beam/model_bart_beam.onnx"
onnx_model_path_beam_o = "/bert_ort/wy/Fluency/new_model_0804/test_bart_beam/model_bart_beam_o.onnx"

print("optimizing greedy search")
from optimizer import optimize_model
from onnx_model_bert import BertOptimizationOptions
optimization_options = BertOptimizationOptions('bert')
optimization_options.enable_embed_layer_norm = False
opt_model = optimize_model(onnx_model_path_greedy,
                           'bert',
                           num_heads=8,
                           hidden_size=512,
                           opt_level=2,
                           optimization_options=optimization_options,
                           use_gpu=False,
                           only_onnxruntime=False)

opt_model.save_model_to_file(onnx_model_path_greedy_o, use_external_data_format=False)

print("optimizing beam search")
opt_model_1 = optimize_model(onnx_model_path_beam,
                           'bert',
                           num_heads=8,
                           hidden_size=512,
                           opt_level=2,
                           optimization_options=optimization_options,
                           use_gpu=False,
                           only_onnxruntime=False)

opt_model_1.save_model_to_file(onnx_model_path_beam_o, use_external_data_format=False)

print("create session")
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
sess_options = SessionOptions()
sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
execution_providers = ['CPUExecutionProvider']

ort_session = InferenceSession(onnx_model_path_greedy, sess_options, providers=execution_providers)
ort_session_optimized = InferenceSession(onnx_model_path_greedy_o, sess_options, providers=execution_providers)
ort_input = generate_input_data_greedy()
ort_output = ort_session.run(None, ort_input)
ort_output_optimized = ort_session_optimized.run(None, ort_input)
parity_1 = np.allclose(ort_output[0], ort_output_optimized[0])

ort_session = InferenceSession(onnx_model_path_beam, sess_options, providers=execution_providers)
ort_session_optimized = InferenceSession(onnx_model_path_beam_o, sess_options, providers=execution_providers)
ort_input = generate_input_data_beam()
ort_output = ort_session.run(None, ort_input)
ort_output_optimized = ort_session_optimized.run(None, ort_input)
parity_2 = np.allclose(ort_output[0], ort_output_optimized[0])

print(opt_model.get_fused_operator_statistics())
print(opt_model_1.get_fused_operator_statistics())
print("check greedy search parity", parity_1)
print("check beam search parity", parity_2)