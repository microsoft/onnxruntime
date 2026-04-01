// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/llm/attention.h"
#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"

/*
Key design decisions:
1. Input parsing: Handles both 3D (B, S, hidden) and 4D (B, N, S, H) input formats per the ONNX spec
2. MHA vs GQA: Detects whether q_num_heads == kv_num_heads (MHA) or q_num_heads > kv_num_heads (GQA) and configures WebgpuAttentionParameters accordingly
3. Flash attention: Used when available (no output_qk needed, subgroups feature present, no bias)
4. 3D→BNSH conversion: For 3D inputs, uses TransferBSDToBNSH to convert to the BNSH format expected by the attention kernels
5. 4D output: Computes in BSD layout (as the shader outputs), then transposes back to BNSH for 4D output format
6. Attention mask: Reshapes 2D/3D masks to 4D for the shader's broadcasting logic; boolean masks return NOT_SUPPORTED

Remaining failures fall into known limitation categories:
    Boolean masks (2) — not yet supported on WebGPU
    SoftCap (2) — not yet wired through to the shader
    GQA output (3) — output stride mismatch for GQA with different kv_num_heads
    QK matmul output (5) — the output_qk output needs additional work
    Present without past (2) — present key/value output without past input needs handling
    is_causal (1) — causal masking interaction

[  PASSED  ] 24 tests.
[  FAILED  ] 15 tests, listed below:
[  FAILED  ] AttentionTest.Attention4DAttnMaskBoolAllFalse
[  FAILED  ] AttentionTest.Attention4DAttnMaskBoolAllFalseDecodeWithPast
[  FAILED  ] AttentionTest.Attention4DSoftCap
[  FAILED  ] AttentionTest.Attention4DSoftCapFloat16
[  FAILED  ] AttentionTest.Attention4DAttnMaskBool
[  FAILED  ] AttentionTest.Attention4DAttnIsCausal
[  FAILED  ] AttentionTest.Attention3DGqaAttn
[  FAILED  ] AttentionTest.Attention3DGqaSelfAttnCausal
[  FAILED  ] AttentionTest.Attention4DGqaAttnMask
[  FAILED  ] AttentionTest.Attention4DWithPastAndPresentQkMatmul
[  FAILED  ] AttentionTest.Attention3DWithPastAndPresentQkMatmul
[  FAILED  ] AttentionTest.Attention4DWithMask3DPastAndPresentQkMatmul
[  FAILED  ] AttentionTest.Attention4DWithMask3DPastAndPresentQkMatmulCausal
[  FAILED  ] AttentionTest.TestAttention4DWithPastAndPresentQkMatmulBias4DMaskCausal
[  FAILED  ] AttentionTest.AttentionNoPastWithPresentOutput
*/

namespace onnxruntime {
namespace webgpu {

Status Attention::ComputeInternal(ComputeContext& context) const {
  const Tensor* Q = context.Input(0);
  const Tensor* K = context.Input(1);
  const Tensor* V = context.Input(2);
  const Tensor* attn_mask = context.Input(3);   // optional
  const Tensor* past_key = context.Input(4);    // optional
  const Tensor* past_value = context.Input(5);  // optional
  // Input 6 is nonpad_kv_seqlen (opset 24 only) - not yet supported

  ORT_RETURN_IF(Q == nullptr || K == nullptr || V == nullptr,
                "Q, K, and V inputs must not be null");
  ORT_RETURN_IF((past_key == nullptr) != (past_value == nullptr),
                "past_key and past_value must be both present or both absent");

  const auto& q_shape = Q->Shape();
  const auto& k_shape = K->Shape();
  const auto& v_shape = V->Shape();
  const int q_dims = static_cast<int>(q_shape.NumDimensions());

  ORT_RETURN_IF(q_dims != 3 && q_dims != 4, "Q must be a 3D or 4D tensor");
  ORT_RETURN_IF(q_dims != static_cast<int>(k_shape.NumDimensions()),
                "Q and K must have the same rank");
  ORT_RETURN_IF(q_dims != static_cast<int>(v_shape.NumDimensions()),
                "Q and V must have the same rank");

  const bool is_4d = (q_dims == 4);
  int batch_size, q_sequence_length, kv_sequence_length, head_size, v_head_size;
  int q_num_heads_val, kv_num_heads_val;

  if (is_4d) {
    batch_size = static_cast<int>(q_shape[0]);
    q_num_heads_val = static_cast<int>(q_shape[1]);
    q_sequence_length = static_cast<int>(q_shape[2]);
    head_size = static_cast<int>(q_shape[3]);
    kv_num_heads_val = static_cast<int>(k_shape[1]);
    kv_sequence_length = static_cast<int>(k_shape[2]);
    v_head_size = static_cast<int>(v_shape[3]);
  } else {
    // 3D: (batch_size, sequence_length, hidden_size)
    batch_size = static_cast<int>(q_shape[0]);
    q_sequence_length = static_cast<int>(q_shape[1]);
    q_num_heads_val = static_cast<int>(q_num_heads_);
    kv_num_heads_val = static_cast<int>(kv_num_heads_);
    ORT_RETURN_IF(q_num_heads_val <= 0 || kv_num_heads_val <= 0,
                  "q_num_heads and kv_num_heads attributes are required for 3D inputs");
    ORT_RETURN_IF(q_shape[2] % q_num_heads_val != 0,
                  "Q hidden size must be divisible by q_num_heads");
    ORT_RETURN_IF(v_shape[2] % kv_num_heads_val != 0,
                  "V hidden size must be divisible by kv_num_heads");
    head_size = static_cast<int>(q_shape[2]) / q_num_heads_val;
    kv_sequence_length = static_cast<int>(k_shape[1]);
    v_head_size = static_cast<int>(v_shape[2]) / kv_num_heads_val;
  }

  ORT_RETURN_IF(q_num_heads_val % kv_num_heads_val != 0,
                "q_num_heads must be a multiple of kv_num_heads");

  const int past_sequence_length = (past_key != nullptr)
                                       ? static_cast<int>(past_key->Shape()[2])
                                       : 0;
  const int total_sequence_length = past_sequence_length + kv_sequence_length;
  const float scale_val = (scale_ != 0.0f)
                              ? scale_
                              : (1.0f / std::sqrt(static_cast<float>(head_size)));
  const bool is_gqa = (q_num_heads_val != kv_num_heads_val);

  // Build contrib::AttentionParameters to construct WebgpuAttentionParameters
  contrib::AttentionParameters params = {};
  params.batch_size = batch_size;
  params.sequence_length = q_sequence_length;
  params.kv_sequence_length = kv_sequence_length;
  params.past_sequence_length = past_sequence_length;
  params.total_sequence_length = total_sequence_length;
  params.hidden_size = q_num_heads_val * head_size;
  params.head_size = head_size;
  params.v_hidden_size = q_num_heads_val * v_head_size;
  params.v_head_size = v_head_size;
  params.num_heads = q_num_heads_val;
  params.is_unidirectional = (is_causal_ == 1);
  params.scale = scale_val;
  params.mask_filter_value = -10000.0f;
  params.qkv_format = contrib::Q_K_V_BNSH;
  params.mask_type = contrib::MASK_NONE;

  contrib::webgpu::WebgpuAttentionParameters parameters(params);

  // For GQA (q_num_heads > kv_num_heads), set additional fields
  if (is_gqa) {
    parameters.is_gqa_ = true;
    parameters.kv_num_heads_ = kv_num_heads_val;
    parameters.kv_hidden_size_ = kv_num_heads_val * head_size;
    parameters.v_hidden_size_ = kv_num_heads_val * v_head_size;
    parameters.v_head_size_ = v_head_size;
    parameters.n_reps = q_num_heads_val / kv_num_heads_val;
  }

  if (softcap_ != 0.0f) {
    parameters.softcap_ = softcap_;
  }

  // Handle attention mask - reshape to 4D if needed for the shader.
  // The shader expects 4D (batch, heads, q_seq, total_seq) and broadcasts
  // via attn_bias_dim0/dim1 clamping.
  const Tensor* attention_bias = nullptr;
  Tensor reshaped_mask;
  if (attn_mask != nullptr) {
    ORT_RETURN_IF(attn_mask->IsDataType<bool>(),
                  "Boolean attention mask is not yet supported for WebGPU Attention");
    const auto mask_ndims = static_cast<int>(attn_mask->Shape().NumDimensions());
    if (mask_ndims == 4) {
      attention_bias = attn_mask;
    } else if (mask_ndims == 3) {
      // (A, q_seq, total_seq) → (1, A, q_seq, total_seq) per numpy broadcasting
      TensorShape new_shape({1, attn_mask->Shape()[0],
                             attn_mask->Shape()[1], attn_mask->Shape()[2]});
      reshaped_mask = Tensor(attn_mask->DataType(), new_shape,
                             const_cast<void*>(attn_mask->DataRaw()),
                             attn_mask->Location());
      attention_bias = &reshaped_mask;
    } else if (mask_ndims == 2) {
      // (q_seq, total_seq) → (1, 1, q_seq, total_seq)
      TensorShape new_shape({1, 1, attn_mask->Shape()[0], attn_mask->Shape()[1]});
      reshaped_mask = Tensor(attn_mask->DataType(), new_shape,
                             const_cast<void*>(attn_mask->DataRaw()),
                             attn_mask->Location());
      attention_bias = &reshaped_mask;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "attn_mask must be 2D, 3D, or 4D tensor");
    }
  }

  // Allocate output tensors.
  // ApplyAttention and FlashAttention output in BSNH (BSD) memory layout.
  // For 4D output, compute into a temporary BSD tensor then transpose to BNSH.
  const int output_hidden = q_num_heads_val * v_head_size;
  Tensor output_bsd_temp;
  Tensor* output;
  Tensor* compute_output;

  if (is_4d) {
    TensorShapeVector y_shape({batch_size, q_num_heads_val,
                               q_sequence_length, v_head_size});
    output = context.Output(0, TensorShape(y_shape));
    // Temporary BSD tensor for the computation
    TensorShapeVector bsd_shape({batch_size, q_sequence_length, output_hidden});
    output_bsd_temp = context.CreateGPUTensor(Q->DataType(), TensorShape(bsd_shape));
    compute_output = &output_bsd_temp;
  } else {
    TensorShapeVector y_shape({batch_size, q_sequence_length, output_hidden});
    output = context.Output(0, TensorShape(y_shape));
    compute_output = output;
  }

  // Present key/value outputs (optional)
  std::vector<int64_t> present_key_dims{
      batch_size, kv_num_heads_val, total_sequence_length, head_size};
  std::vector<int64_t> present_value_dims{
      batch_size, kv_num_heads_val, total_sequence_length, v_head_size};
  Tensor* present_key_output = context.Output(1, TensorShape(present_key_dims));
  Tensor* present_value_output = context.Output(2, TensorShape(present_value_dims));

  // QK matmul output (optional, output index 3)
  Tensor* output_qk = nullptr;
  if (context.OutputCount() > 3) {
    std::vector<int64_t> qk_dims{
        batch_size, q_num_heads_val, q_sequence_length, total_sequence_length};
    output_qk = context.Output(3, TensorShape(qk_dims));
  }

  // Prepare Q, K, V in BNSH format.
  // 4D inputs are already BNSH; 3D inputs need BSD→BNSH conversion.
  const Tensor* Q_bnsh = Q;
  const Tensor* K_bnsh = K;
  const Tensor* V_bnsh = V;
  Tensor Q_converted, K_converted, V_converted;

  if (!is_4d) {
    TensorShapeVector q_bnsh_dims({batch_size, q_num_heads_val,
                                   q_sequence_length, head_size});
    Q_converted = context.CreateGPUTensor(Q->DataType(), TensorShape(q_bnsh_dims));
    ORT_RETURN_IF_ERROR(contrib::webgpu::TransferBSDToBNSH(
        context, q_num_heads_val, q_sequence_length,
        head_size, Q, nullptr, 0, &Q_converted));
    Q_bnsh = &Q_converted;

    TensorShapeVector k_bnsh_dims({batch_size, kv_num_heads_val,
                                   kv_sequence_length, head_size});
    K_converted = context.CreateGPUTensor(K->DataType(), TensorShape(k_bnsh_dims));
    ORT_RETURN_IF_ERROR(contrib::webgpu::TransferBSDToBNSH(
        context, kv_num_heads_val, kv_sequence_length,
        head_size, K, nullptr, 0, &K_converted));
    K_bnsh = &K_converted;

    TensorShapeVector v_bnsh_dims({batch_size, kv_num_heads_val,
                                   kv_sequence_length, v_head_size});
    V_converted = context.CreateGPUTensor(V->DataType(), TensorShape(v_bnsh_dims));
    ORT_RETURN_IF_ERROR(contrib::webgpu::TransferBSDToBNSH(
        context, kv_num_heads_val, kv_sequence_length,
        v_head_size, V, nullptr, 0, &V_converted));
    V_bnsh = &V_converted;
  }

  // Try flash attention first (not available when output_qk is needed)
  if (output_qk == nullptr &&
      contrib::webgpu::CanApplyFlashAttention(nullptr, parameters, context)) {
    ORT_RETURN_IF_ERROR(contrib::webgpu::ApplyFlashAttention(
        Q_bnsh, K_bnsh, V_bnsh, attention_bias,
        compute_output, past_key, present_key_output,
        past_value, present_value_output, parameters, context));
  } else {
    // Fall back to tiled attention
    ORT_RETURN_IF_ERROR(contrib::webgpu::ApplyAttention(
        Q_bnsh, K_bnsh, V_bnsh, attention_bias,
        past_key, past_value,
        compute_output, present_key_output, present_value_output,
        output_qk, parameters, context));
  }

  // For 4D output, transpose from BSNH (BSD) to BNSH
  if (is_4d) {
    ORT_RETURN_IF_ERROR(contrib::webgpu::TransferBSDToBNSH(
        context, q_num_heads_val, q_sequence_length,
        v_head_size, compute_output, nullptr, 0, output));
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Attention, kOnnxDomain, 23, 23, kWebGpuExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T1", WebGpuSupportedFloatTypes())
                                      .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
                                  Attention);

ONNX_OPERATOR_KERNEL_EX(Attention, kOnnxDomain, 24, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T1", WebGpuSupportedFloatTypes())
                            .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
                        Attention);

}  // namespace webgpu
}  // namespace onnxruntime
