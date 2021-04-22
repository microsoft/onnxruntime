import torch
import numpy
MERGE_PAST_KEY_VALUE = True

def create_dummy_inputs(batch_size, sequence_length, past_sequence_length,
                        hidden_size, num_attention_heads, num_layers, max_seq_length,
                        device, vocab_size, is_fp16=False):
    input_ids = torch.randint(low=0,
                              high=vocab_size - 1,
                              size=(batch_size, sequence_length),
                              dtype=torch.int64,
                              device=device)

    input_ids = torch.ones(batch_size, sequence_length, dtype=torch.int64, device=device)
    position_ids  = torch.ones(batch_size, sequence_length, dtype=torch.int64, device=device)

    attention_mask = torch.tril(torch.ones(batch_size, 1, max_seq_length, max_seq_length, dtype=torch.int64, device=device))

    if not MERGE_PAST_KEY_VALUE:
        past_shape = [batch_size, num_attention_heads, past_sequence_length, int(hidden_size / num_attention_heads)]
        """
        k = torch.rand(past_shape, dtype=torch.float32, device=device)
        v = torch.rand(past_shape, dtype=torch.float32, device=device)
        past_layer = (k, v)
        past_key_values = (past_layer,) * num_layers #past_key_values = [(k, v) for _ in range(num_layers)]
        """
        past_key_values = [torch.rand(past_shape, dtype=torch.float32 if not is_fp16 else torch.float16, device=device) for _ in range(2 * num_layers)]
    else:
        past_shape = [2, batch_size, num_attention_heads, past_sequence_length, int(hidden_size / num_attention_heads)]
        past_key_values = [torch.rand(past_shape, dtype=torch.float32 if not is_fp16 else torch.float16, device=device) for _ in range(num_layers)]

    return input_ids, position_ids, attention_mask, past_key_values



# optimize

onnx_model_path = "/bert_ort/wy/Megatron/scripts/scripts_after_change/fp16_merge.onnx"
#onnx_model_path = "/bert_ort/wy/Megatron/ChitChatONNX/megatron_onnx_partial/fp16_merge.onnx"
optimized_model_path = "/bert_ort/wy/Transformers/megatron/onnxruntime/python/tools/transformers/megatron_optimized/fp16_merge_optimized.onnx"

print("optimizing")
from optimizer import optimize_model
from onnx_model_bert import BertOptimizationOptions
optimization_options = BertOptimizationOptions('gpt2')
#optimization_options.use_raw_attention_mask(use_raw_attention_mask)
opt_model = optimize_model(onnx_model_path,
                           'gpt2',
                           num_heads=16,
                           hidden_size=1024,
                           opt_level=1,
                           optimization_options=optimization_options,
                           use_gpu=True,
                           only_onnxruntime=False)

opt_model.save_model_to_file(optimized_model_path, use_external_data_format = True)
print(opt_model.get_fused_operator_statistics())

#test parity
print("test parity")
num_layers = 6
input_ids, position_ids, attention_mask, past_key_values = create_dummy_inputs(1, 1, 0, 1024, 16, 6, 1024, "cuda", 50304, True)
past_key_values_fp32 = [v.clone().to(torch.float32) for v in past_key_values]

ort_inputs = {'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy())}
ort_inputs['attention_mask'] = numpy.ascontiguousarray(attention_mask.cpu().numpy())
ort_inputs['position_ids'] = numpy.ascontiguousarray(position_ids.cpu().numpy())
for i in range(num_layers):
    if not MERGE_PAST_KEY_VALUE:
        ort_inputs[f'past_key_{i}'] = numpy.ascontiguousarray(
            past_key_values[2 * i].cpu().numpy())
        ort_inputs[f'past_value_{i}'] = numpy.ascontiguousarray(
            past_key_values[2 * i + 1].cpu().numpy())
    else:
        ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(
            past_key_values[i].cpu().numpy())

from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel
sess_options = SessionOptions()
sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
execution_providers = ['CUDAExecutionProvider']
ort_session_origin = InferenceSession(onnx_model_path, sess_options, providers=execution_providers)
ort_session_optimized = InferenceSession(optimized_model_path, sess_options, providers=execution_providers)

ort_outputs_origin = ort_session_origin.run(None, ort_inputs)
ort_outputs_optimized = ort_session_optimized.run(None, ort_inputs)

total_outputs = 1 + 2 * num_layers if not MERGE_PAST_KEY_VALUE else 1 + num_layers
assert len(ort_outputs_origin) == total_outputs

for i in range(total_outputs):
    is_close = numpy.allclose(ort_outputs_origin[i], ort_outputs_optimized[i],
                              rtol=0.00001,
                              atol=0.00001)
    diff = numpy.abs(ort_outputs_optimized[i] - ort_outputs_origin[i])
    max_diff = numpy.amax(diff)
    print(f'Original onnx and Optimized onnx results {i}: max_abs_diff={max_diff:.8f}')

