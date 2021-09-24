deep_conflation = r"C:\Users\wangye\Work\Turing-Sep-model\DeepConflation_20x512.onnx"
deep_conflation_opt = r"C:\Users\wangye\Work\Turing-Sep-model\DeepConflation_20x512_opt.onnx"
offensive_v4 = r"C:\Users\wangye\Work\Turing-Sep-model\OffensiveV4_10x128.onnx"
offensive_v4_opt = r"C:\Users\wangye\Work\Turing-Sep-model\OffensiveV4_10x128_opt.onnx"

def parity_check(path, opt_path, dict_size):
    import numpy
    for ep in ['CPUExecutionProvider', 'CUDAExecutionProvider']:
        for batch in [1, 4, 16, 32]:
            for seq_len in [128, 256, 512]:
                input_ids = numpy.random.randint(low=0, high=dict_size - 1, size=(batch, seq_len), dtype = numpy.int64)
                attention_mask = numpy.ones([batch, seq_len], dtype = numpy.int64)
                inputs = {'input_ids': input_ids, 'attention_mask' : attention_mask}
                from onnxruntime import SessionOptions, InferenceSession
                sess_options = SessionOptions()
                session = InferenceSession(path, sess_options, providers=[ep])
                session_opt = InferenceSession(opt_path, sess_options, providers=[ep])
                output = session.run(None, inputs)
                output_opt = session_opt.run(None, inputs)
                print("EP:", ep, "batch:", batch, "seq_len:", seq_len, "parity result:", numpy.allclose(output, output_opt, 1e-3, 1e-3))

from optimizer import optimize_model
from onnx_model_bert import BertOptimizationOptions
optimization_options = BertOptimizationOptions('bert')
optimization_options.enable_embed_layer_norm = True
opt_model = optimize_model(deep_conflation,
                           'bert',
                           num_heads=8,
                           hidden_size=256,
                           opt_level=1,
                           optimization_options=optimization_options,
                           use_gpu=False,
                           only_onnxruntime=False)
opt_model.save_model_to_file(deep_conflation_opt, use_external_data_format=False)
print(opt_model.get_fused_operator_statistics())

parity_check(deep_conflation, deep_conflation_opt, 64044)
        

# opt_model = optimize_model(offensive_v4,
#                            'gpt2',
#                            num_heads=16,
#                            hidden_size=1024,
#                            opt_level=99,
#                            optimization_options=optimization_options,
#                            use_gpu=False,
#                            only_onnxruntime=False)
# opt_model.save_model_to_file(offensive_v4_opt, use_external_data_format=False)
# print(opt_model.get_fused_operator_statistics())



# from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
# sess_options = SessionOptions()
# sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
# execution_providers = ['CPUExecutionProvider']

# ort_session = InferenceSession(onnx_model_path_greedy, sess_options, providers=execution_providers)

# ort_output = ort_session.run(None, ort_input)