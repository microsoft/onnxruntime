// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/flash_attention.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FlashAttention,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      FlashAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
FlashAttention<T>::FlashAttention(const OpKernelInfo& info) : RocmKernel(info) {
  int64_t num_heads;
  info.GetAttrOrDefault("num_heads", &num_heads, static_cast<int64_t>(0));
  this->num_heads_ = num_heads;

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status FlashAttention<T>::CheckMask(const Tensor *att_mask
                                    AttentionMaskType &mask_type,
                                    int64_t batch,
                                    int64_t kv_seqlen) {
  auto &mask_shape = att_mask->Shape();
  auto m_dims = mask_shape.GetDims();
  if (m_dims.size() == 
}

template <typename T>
Status FlashAttention<T>::CheckInputs(const TensorShape &query_shape,
                                 const TensorShape &key_shape,
                                 const TensorShape &value_shape,
                                 const Tensor* att_mask,
                                 const Tensor* att_bias,
				 AttentionParameters *attn) const {
  auto &q_dims = query_shape.GetDims();
  auto &k_dims = key_shape.GetDims();
  auto &v_dims = value_shape.GetDims();

  if (q_dims.size() != k_dims.size() || k_dims.size() != v_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input QKV should have same dims, got ", dims.size());
  }

  atten->past_sequence_length = 0;
  atten->original_past_sequence_length = 0;
  atten->is_unidirectional = true;
  atten->past_present_share_buffer = false;
  atten->do_rotary = false;
  atten->pass_past_in_kv = false;
  atten->mask_filter_value = mask_filter_value_;
  atten->scale = scale_;

  AttentionMaskType mask_type = AttentionMaskType::MASK_NONE;
  if (att_mask != nullptr) {
    auto &m_dims = att_mask->Shape().GetDims();
    if (m_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "mask input shape is invalid");
    }
    mask_type = AttentionMaskType::MASK_4D_MEGATRON;
  }
  atten->mask_type = mask_type;
  atten->broadcast_res_pos_bias = false;
 
  if (q_dims.size() == 3) {
    // q,k, v is (b*n, s, h)
    auto batch_head = q_dims[0];
    auto seqlen = q_dims[1];
    auto hidden = q_dims[2];
    auto kv_seqlen = k_dims[1];
    auto v_hidden = v_dims[2];
    if (q_dims[0] != k_dims[0] || q_dims[1] != k_dims[1] || k_dims[1] != v_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input QKV(3D) dim is not valid");
    }
    atten->batch_size = static_cast<int>(batch_head);
    atten->sequence_length = static_cast<int>(seqlen);
    atten->kv_sequence_length = kv_seqlen;
    atten->total_sequence_length = kv_seqlen;
    atten->max_sequence_length = kv_seqlen;
    atten->input_hidden_size = hidden;
    atten->v_hidden_size = v_hidden;
    atten->head_size = batch_head / num_heads_;
    atten->v_head_size = batch_head / num_heads_;
    atten->num_heads = num_heads_;
  } else if (q_dims.size() == 4) {
    // q,k, v is (b, n, s, h)
    auto batch = q_dims[0];
    auto head = q_dims[1];
    auto seqlen = q_dims[2];
    auto hidden = q_dims[3];
    auto kv_seqlen = k_dims[2];
    auto v_hidden = v_dims[3];
    if (q_dims[0] != k_dims[0] || head != num_heads_ || q_dims[1] != k_dims[1] || k_dims[2] != v_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input QKV(4D) dim is not valid");
    }
    atten->batch_size = static_cast<int>(batch);
    atten->sequence_length = static_cast<int>(seqlen);
    atten->kv_sequence_length = static_cast<int>(kv_seqlen);
    atten->total_sequence_length = static_cast<int>(kv_seqlen);
    atten->max_sequence_length = static_cast<int>(kv_seqlen);
    atten->input_hidden_size = static_cast<int>(hidden);
    atten->v_hidden_size = static_cast<int>(v_hidden);
    atten->head_size = batch_head / num_heads_;
    atten->v_head_size = batch_head / num_heads_;
    atten->num_heads = num_heads_;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input QKV should be 3D or 4D");
  }

  return Status::OK();
}

template <typename T>
Status FlashAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* att_mask = context->Input<Tensor>(3);  // optional
  const Tensor* att_bias = context->Input<Tensor>(4);  // optional

  AttentionParameters attn;
  ORT_RETURN_IF_ERROR(CheckInputs(query->Shape(),
                                  key->Shape(),
                                  value->Shape(),
                                  att_mask,
                                  att_bias,
                                  &attn));

  auto query_shape = query->Shape();
  auto value_shape = value->Shape();
  TensorShape output_shape(query_shape);
  output_shape[query_shape.NumDimensions() - 1] = value_shape[value_shape.NumDimensions() - 1];

  Tensor* output = context->Output(0, output_shape);
  ORT_UNUSED_PARAMETER(output);

  // LOGS_DEFAULT(WARNING) << "need implementation of FlashAttention";

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionGeneric = GemmSoftmaxGemmPermuteGenericPipeline<HipT>;
  using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;

  size_t shared_workspace_bytes = AttentionGeneric::GetWorkspaceNumBytes(&attn);
  if (GetTuningContext()->IsTunableOpEnabled()) {
    shared_workspace_bytes = std::max(shared_workspace_bytes, AttentionTunableOp::GetWorkspaceNumBytes(&attn));
  }

  auto workspace = GetScratchBuffer<void>(shared_workspace_bytes, context->GetComputeStream());

  // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax
  const TransformerOptions* options = TransformerOptions::GetInstance();
  bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();

  auto& device_prop = GetDeviceProp();
  GemmSoftmaxGemmPermuteParams<HipT> gemm_softmax_gemm_permute_params;
  {
    auto& params = gemm_softmax_gemm_permute_params;
    params.tuning_ctx = GetTuningContext();
    params.stream = Stream(context);
    params.handle = GetRocblasHandle(context);
    params.attention = &attn;
    params.device_prop = &device_prop;
    // FIXME: the params.scale seems to be different from AttentionParameters::scale;
    params.scale = 1.0f / sqrt(static_cast<float>(attn.head_size));
    params.q_buffer = reinterpret_cast<const HipT*>(query->DataRaw());
    params.k_buffer = reinterpret_cast<const HipT*>(key->DataRaw());
    params.v_buffer = reinterpret_cast<const HipT*>(value->DataRaw());
    params.out_buffer = reinterpret_cast<HipT*>(output->MutableDataRaw());

    if (att_bias != nullptr) {
      params.bias_buffer = reinterpret_cast<const HipT*>(att_bias->DataRaw());
      params.bias_dims = att_bias->Shape().GetDims();
    }

    if (att_mask != nullptr) {
      // params.mask_index_buffer = mask_index->Data<int>();
      // params.mask_index_dims = mask_index->Shape().GetDims();
      params.mask_buffer = reinterpret_cast<const HipT*>(att_mask->DataRaw());
      params.mask_index_dims = att_mask->Shape().GetDims();
    }

    params.workspace_buffer = reinterpret_cast<HipT*>(workspace.get());
  }

  if (GetTuningContext()->IsTunableOpEnabled() &&
      !use_persistent_softmax) {
    return AttentionTunableOp{}(&gemm_softmax_gemm_permute_params);
  } else {
    return AttentionGeneric::Run(&gemm_softmax_gemm_permute_params, use_persistent_softmax);
  }
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
