// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/multihead_attention.h"

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"
#include "core/platform/env_var_utils.h"
#include "core/providers/rocm/rocm_common.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_MHA_KERNEL_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MultiHeadAttention,                                         \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MultiHeadAttention<T>)

REGISTER_MHA_KERNEL_TYPED(float);
REGISTER_MHA_KERNEL_TYPED(MLFloat16);

static constexpr int kPastSequenceLengthInputIndex = 7;
static constexpr int kBeamWidthInputIndex = 8;
static constexpr int kPastInputIndex = 5;
static constexpr int kPresentOutputIndex = 1;

#define REGISTER_DMMHA_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      DecoderMaskedMultiHeadAttention,                                        \
      kMSDomain,                                                              \
      1,                                                                      \
      T,                                                                      \
      kRocmExecutionProvider,                                                 \
      (*KernelDefBuilder::Create())                                           \
          .MayInplace(kPastInputIndex, kPresentOutputIndex)                   \
          .MayInplace(kPastInputIndex + 1, kPresentOutputIndex + 1)           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .InputMemoryType(OrtMemTypeCPUInput, kPastSequenceLengthInputIndex) \
          .InputMemoryType(OrtMemTypeCPUInput, kBeamWidthInputIndex),         \
      MultiHeadAttention<T>)

REGISTER_DMMHA_KERNEL_TYPED(float);
REGISTER_DMMHA_KERNEL_TYPED(MLFloat16);

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(const OpKernelInfo& info)
    : RocmKernel(info),
      attn_type_(info.node().OpType() == "DecoderMaskedMultiHeadAttention" ? kDecoderMaskedMultiHeadAttention
                                                                           : kMultiHeadAttention) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  past_present_share_buffer_ = info.GetAttrOrDefault<int64_t>("past_present_share_buffer", 0LL) != 0LL;

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;
  tunable_op_ = std::make_shared<AttentionTunableOp>();
}

template <typename T>
Status MultiHeadAttention<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(
      GetTuningContext()->IsTunableOpEnabled(),
      "MultiHeadAttention of ROCm EP is only supported if tunable op is used and tuning is enabled.");

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);

  const Tensor* bias{};
  const Tensor* key_padding_mask{};
  const Tensor* relative_position_bias{};
  const Tensor* past_key{};
  const Tensor* past_value{};
  const Tensor* past_seq_len{};

  if (attn_type_ == kMultiHeadAttention) {
    bias = context->Input<Tensor>(3);
    key_padding_mask = context->Input<Tensor>(4);
    relative_position_bias = context->Input<Tensor>(5);
    past_key = context->Input<Tensor>(6);
    past_value = context->Input<Tensor>(7);
  } else if (attn_type_ == kDecoderMaskedMultiHeadAttention) {
    key_padding_mask = context->Input<Tensor>(3);
    relative_position_bias = context->Input<Tensor>(4);
    past_key = context->Input<Tensor>(5);
    past_value = context->Input<Tensor>(6);
    past_seq_len = context->Input<Tensor>(kPastSequenceLengthInputIndex);
    // const Tensor* beam_width = context->Input<Tensor>(8);         // NOTE: not used
    // const Tensor* cache_indirection = context->Input<Tensor>(9);  // TODO: should not present for ROCm EP
    bias = context->Input<Tensor>(10);
  }

  if (nullptr != bias) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "qkv_bias is not supported on ROCm EP. "
                           "User should fuse the qkv bias to qkv projection instead.");
  }

  auto& device_prop = GetDeviceProp();
  RocmAttentionParameters attn;
  ORT_RETURN_IF_ERROR(
      multihead_attention_helper::CheckInputs<Tensor>(
          query, key, value, bias,
          key_padding_mask, relative_position_bias,
          past_key, past_value, past_seq_len,
          &attn,
          num_heads_, mask_filter_value_, scale_,
          past_present_share_buffer_, false, device_prop.maxThreadsPerBlock));

  if (attn_type_ == kDecoderMaskedMultiHeadAttention && attn.sequence_length != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input sequence length should be 1 to use DecoderMaskedMultiHeadAttention");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(attn.batch_size);
  output_shape[1] = static_cast<int64_t>(attn.sequence_length);
  output_shape[2] = static_cast<int64_t>(attn.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      attn.batch_size,
      attn.num_heads,
      past_present_share_buffer_ ? attn.max_sequence_length : attn.total_sequence_length,
      attn.head_size,
  };
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  ORT_RETURN_IF_ERROR(ClassifyAttentionMode(
      attn_type_, &attn,
      /*qkv=*/{query, key, value},
      /*past=*/{past_key, past_value},
      /*present=*/{present_key, present_value}));

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;
  auto workspace_bytes = AttentionTunableOp::GetWorkspaceNumBytes(&attn);
  auto workspace = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  hipStream_t stream = Stream(context);
  if (nullptr != present_key) {  // process past present concat
    Strides dst_strides;

    int4 past_shape;
    Strides past_src_strides;
    const HipT* past_key_src;
    const HipT* past_value_src;
    HipT* past_key_dst{};
    HipT* past_value_dst{};

    int4 add_shape;
    Strides add_src_strides;
    const HipT* add_key_src = reinterpret_cast<const HipT*>(key->DataRaw());
    const HipT* add_value_src = reinterpret_cast<const HipT*>(value->DataRaw());
    HipT* add_key_dst;
    HipT* add_value_dst;

    if (attn.mode == BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH ||
        attn.mode == BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH) {
      dst_strides = Strides::BNSHMemory(attn.batch_size, attn.num_heads, attn.total_sequence_length, attn.head_size);

      past_shape = {attn.batch_size, attn.num_heads, attn.past_sequence_length, attn.head_size};
      past_src_strides = Strides::BNSHMemory(attn.batch_size, attn.num_heads, attn.past_sequence_length, attn.head_size);
      past_key_src = reinterpret_cast<const HipT*>(past_key->DataRaw());
      past_value_src = reinterpret_cast<const HipT*>(past_value->DataRaw());
      past_key_dst = reinterpret_cast<HipT*>(present_key->MutableDataRaw());
      past_value_dst = reinterpret_cast<HipT*>(present_value->MutableDataRaw());

      if (attn.mode == BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH) {
        add_src_strides = Strides::BSNHMemory(attn.batch_size, attn.kv_sequence_length, attn.num_heads, attn.head_size);
      } else if (attn.mode == BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH) {
        add_src_strides = Strides::BNSHMemory(attn.batch_size, attn.num_heads, attn.kv_sequence_length, attn.head_size);
      }
    } else if (attn.mode == BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH ||
               attn.mode == BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH) {
      dst_strides = Strides::BNSHMemory(attn.batch_size, attn.num_heads, attn.total_sequence_length, attn.head_size);

      if (attn.mode == BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH) {
        add_src_strides = Strides::BSNHMemory(attn.batch_size, attn.kv_sequence_length, attn.num_heads, attn.head_size);
      } else if (attn.mode == BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH) {
        add_src_strides = Strides::BNSHMemory(attn.batch_size, attn.num_heads, attn.kv_sequence_length, attn.head_size);
      }
    } else if (
        attn.mode == BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH ||
        attn.mode == BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH ||
        attn.mode == BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH ||
        attn.mode == BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH) {
      dst_strides = Strides::BNSHMemory(attn.batch_size, attn.num_heads, attn.max_sequence_length, attn.head_size);

      if (attn.mode == BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH || attn.mode == BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH) {
        add_src_strides = Strides::BSNHMemory(attn.batch_size, attn.kv_sequence_length, attn.num_heads, attn.head_size);
      } else if (attn.mode == BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH || attn.mode == BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH) {
        add_src_strides = Strides::BNSHMemory(attn.batch_size, attn.num_heads, attn.kv_sequence_length, attn.head_size);
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "past present concatenation  is not implemented for attention mode ", attn.mode);
    }
    add_shape = {attn.batch_size, attn.num_heads, attn.kv_sequence_length, attn.head_size};  // kernel in coord (b,n,s,h)
    add_key_dst = reinterpret_cast<HipT*>(present_key->MutableDataRaw()) + dst_strides.OffsetAt(0, 0, attn.past_sequence_length, 0);
    add_value_dst = reinterpret_cast<HipT*>(present_value->MutableDataRaw()) + dst_strides.OffsetAt(0, 0, attn.past_sequence_length, 0);

    if (past_key_dst) {
      ORT_RETURN_IF_ERROR(LaunchStridedCopy(
          stream, past_key_src, past_shape, past_src_strides.ForBNSHCoord(),
          past_key_dst, dst_strides.ForBNSHCoord(), device_prop.maxThreadsPerBlock));
    }
    if (past_value_dst) {
      ORT_RETURN_IF_ERROR(LaunchStridedCopy(
          stream, past_value_src, past_shape, past_src_strides.ForBNSHCoord(),
          past_value_dst, dst_strides.ForBNSHCoord(), device_prop.maxThreadsPerBlock));
    }

    ORT_RETURN_IF_ERROR(LaunchStridedCopy(
        stream, add_key_src, add_shape, add_src_strides.ForBNSHCoord(),
        add_key_dst, dst_strides.ForBNSHCoord(), device_prop.maxThreadsPerBlock));
    ORT_RETURN_IF_ERROR(LaunchStridedCopy(
        stream, add_value_src, add_shape, add_src_strides.ForBNSHCoord(),
        add_value_dst, dst_strides.ForBNSHCoord(), device_prop.maxThreadsPerBlock));
  }

  GemmSoftmaxGemmPermuteParams<HipT> params;
  params.tuning_ctx = GetTuningContext();
  params.stream = context->GetComputeStream();
  params.handle = GetRocblasHandle(context);
  params.attention = &attn;
  params.device_prop = &device_prop;
  params.scale = scale_ == 0 ? 1.0f / sqrt(attn.head_size) : scale_;
  std::tie(params.q_buffer, params.k_buffer, params.v_buffer) = ConvertToOffsetedBufferViews<HipT>(
      &attn,
      nullptr == query ? nullptr : reinterpret_cast<const HipT*>(query->DataRaw()),
      nullptr == key ? nullptr : reinterpret_cast<const HipT*>(key->DataRaw()),
      nullptr == value ? nullptr : reinterpret_cast<const HipT*>(value->DataRaw()),
      nullptr == present_key ? nullptr : reinterpret_cast<const HipT*>(present_key->DataRaw()),
      nullptr == present_value ? nullptr : reinterpret_cast<const HipT*>(present_value->DataRaw()));
  params.out_buffer = reinterpret_cast<HipT*>(output->MutableDataRaw());

  if (key_padding_mask != nullptr) {
    params.mask_index_buffer = key_padding_mask->Data<int>();
    params.mask_index_dims = key_padding_mask->Shape().AsShapeVector();
  }

  if (relative_position_bias != nullptr) {
    params.bias_buffer = reinterpret_cast<const HipT*>(relative_position_bias->DataRaw());
  }

  params.workspace_buffer = reinterpret_cast<HipT*>(workspace.get());
  return (*std::static_pointer_cast<AttentionTunableOp>(tunable_op_))(&params);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
