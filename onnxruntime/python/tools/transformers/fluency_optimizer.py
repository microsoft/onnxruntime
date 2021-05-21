def generate_input_data():
    import onnx
    from onnx import numpy_helper
    # Load a TensorProto
    tensor_0 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/test_bart_greedy/test_data_0/input_0.pb', 'rb') as f:
        tensor_0.ParseFromString(f.read())
    tensor_1 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/test_bart_greedy/test_data_0/input_1.pb', 'rb') as f:
        tensor_1.ParseFromString(f.read())
    tensor_2 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/test_bart_greedy/test_data_0/input_2.pb', 'rb') as f:
        tensor_2.ParseFromString(f.read())
    tensor_3 = onnx.TensorProto()
    with open('/bert_ort/wy/Fluency/test_bart_greedy/test_data_0/input_3.pb', 'rb') as f:
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

onnx_model_path = "/bert_ort/wy/Fluency/test_bart_greedy/model.onnx"
optimized_model_path = "/bert_ort/wy/Fluency/optimized_model/optimized.onnx"

print("optimizing")
from optimizer import optimize_model
from onnx_model_bert import BertOptimizationOptions
optimization_options = BertOptimizationOptions('bert')
optimization_options.enable_embed_layer_norm = False
opt_model = optimize_model(onnx_model_path,
                           'bert',
                           num_heads=8,
                           hidden_size=512,
                           opt_level=1,
                           optimization_options=optimization_options,
                           use_gpu=False,
                           only_onnxruntime=False)

opt_model.save_model_to_file(optimized_model_path, use_external_data_format=False)
print(opt_model.get_fused_operator_statistics())

print("create session")
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
sess_options = SessionOptions()
sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
execution_providers = ['CPUExecutionProvider']
ort_session = InferenceSession(onnx_model_path, sess_options, providers=execution_providers)
ort_session_optimized = InferenceSession(optimized_model_path, sess_options, providers=execution_providers)
print("create session done")
print("generate input data")
ort_input = generate_input_data()
print("run")
ort_output = ort_session.run(None, ort_input)
ort_output_optimized = ort_session_optimized.run(None, ort_input)
print("done")
for i in range(len(ort_output)):
    print(ort_output[i] - ort_output_optimized[i])



