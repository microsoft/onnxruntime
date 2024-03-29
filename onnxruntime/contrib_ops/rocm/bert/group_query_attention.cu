// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/rocm/bert/group_query_attention.h"
#include "contrib_ops/rocm/bert/group_query_attention_helper.h"
#include "contrib_ops/rocm/bert/group_query_attention_impl.cuh"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

// for input 3 and 4 may be use same tensor with output 1 and 2
// that is they are using a large tensor to store past key and value
#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      GroupQueryAttention,                                               \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                                 \
      kRocmExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())   \
          .MayInplace(3, 1)                                              \
          .MayInplace(4, 2)                                              \
          .InputMemoryType(OrtMemTypeCPUInput, 6),                       \
      GroupQueryAttention<T>);

// REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
// REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : RocmKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_past_bsnh_ = false;  // past kv cache is BNSH
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  attn_type_ = kGroupQueryAttention;

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionTunableOp = GroupedQueryAttentionTunableOp<HipT>;
  tunable_op_ = std::make_shared<AttentionTunableOp>();
}

template <typename T>
Status GroupQueryAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* seqlens_k = context->Input<Tensor>(5);
  const Tensor* total_seqlen = context->Input<Tensor>(6);
  // TODO: support rotary embedding
  const Tensor* cos_cache = nullptr;  // context->Input<Tensor>(7);
  const Tensor* sin_cache = nullptr;  // context->Input<Tensor>(8);

  auto& device_prop = GetDeviceProp();
  RocmAttentionParameters attn;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &attn,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                seqlens_k,
                                                                total_seqlen,
                                                                is_past_bsnh_,
                                                                scale_,
                                                                device_prop.maxThreadsPerBlock));

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(attn.batch_size);
  output_shape[1] = static_cast<int64_t>(attn.sequence_length);
  output_shape[2] = static_cast<int64_t>(attn.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims{
      attn.batch_size,
      attn.num_heads,
      attn.total_sequence_length,
      attn.head_size,
  };
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  ORT_RETURN_IF_ERROR(ClassifyAttentionMode(
      kGroupQueryAttention, &attn,
      /*qkv=*/{query, key, value},
      /*past=*/{past_key, past_value},
      /*present=*/{present_key, present_value}));

  using HipT = typename ToHipType<T>::MappedType;
  using AttentionTunableOp = GroupedQueryAttentionTunableOp<HipT>;

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

  GroupedQueryAttentionParams<HipT> params;
  params.tuning_ctx = GetTuningContext();
  params.stream = context->GetComputeStream();
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

  return (*std::static_pointer_cast<AttentionTunableOp>(tunable_op_))(&params);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
