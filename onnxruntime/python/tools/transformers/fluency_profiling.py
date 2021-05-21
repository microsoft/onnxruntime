onnx_model_path = "/bert_ort/wy/Fluency/test_bart_greedy/model.onnx"
optimized_model_path = "/bert_ort/wy/Fluency/optimized_model/optimized.onnx"

print("optimizing")
from optimizer import optimize_model
from onnx_model_bert import BertOptimizationOptions
optimization_options = BertOptimizationOptions('bert')
optimization_options.enable_embed_layer_norm = False
optimization_options.enable_skip_layer_norm = False
#optimization_options.use_raw_attention_mask(use_raw_attention_mask)
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

# print("create session")
# from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
# sess_options = SessionOptions()
# sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
# execution_providers = ['CPUExecutionProvider']
# ort_session_origin = InferenceSession(optimized_model_path, sess_options, providers=execution_providers)
# print("create session done")
