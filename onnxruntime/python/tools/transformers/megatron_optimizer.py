onnx_model_path = "/bert_ort/wy/Megatron/scripts/scripts_after_change/fp16_merge.onnx"

from optimizer import optimize_model
from onnx_model_bert import BertOptimizationOptions
optimization_options = BertOptimizationOptions('gpt2')
#optimization_options.use_raw_attention_mask(use_raw_attention_mask)

opt_model = optimize_model(onnx_model_path,
                           'gpt2',
                           num_heads=16,
                           hidden_size=1024,
                           opt_level=0,
                           optimization_options=optimization_options,
                           use_gpu=True,
                           only_onnxruntime=False)

optimized_model_path = "/bert_ort/wy/Transformers/megatron/onnxruntime/python/tools/transformers/fp16_merge_optimized.onnx"
opt_model.save_model_to_file(optimized_model_path, use_external_data_format = False)
print(opt_model.get_fused_operator_statistics())
