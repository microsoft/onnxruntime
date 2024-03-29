// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor_shape.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_ck_impl/impl.cuh"
#ifdef USE_COMPOSABLE_KERNEL
#include "core/providers/rocm/composable_kernel_common.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#endif  // USE_COMPOSABLE_KERNEL

#include <array>
#include <vector>

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct GroupedQueryAttentionParams : onnxruntime::rocm::tunable::OpParams {
  std::string Signature() const override {
    return MakeString(
        "B", attention->batch_size,
        "_S", attention->sequence_length,
        "_T", attention->total_sequence_length,
        "_N", attention->num_heads,
        "_Nk", attention->kv_num_heads,
        "_H", attention->head_size,
        "_Hv", attention->v_head_size,
        "_QKV", attention->qkv_format,
        "_MODE", attention->mode);
  }

  const RocmAttentionParameters* attention;
  const hipDeviceProp_t* device_prop;

  float scale;
  const T* q_buffer;
  const T* k_buffer;
  const T* v_buffer;
  T* out_buffer;

  // optional, internal
  void* workspace_buffer{nullptr};
};

template <typename T>
class GroupedQueryAttentionTunableOp : public tunable::TunableOp<GroupedQueryAttentionParams<T>> {
 public:
  GroupedQueryAttentionTunableOp();

  inline static bool IsSupportedMode(const RocmAttentionParameters* attn) {
    switch (attn->mode) {
      case QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE:
      case QFMT_KFMT_VFMT_2BNPH_NONE_2BNTH_NONE:
        // depends on qkv format
        if (attn->qkv_format == Q_K_V_BNSH || attn->qkv_format == Q_K_V_BSNH) {
          return true;
        } else {
          return false;
        }
      case BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH:
      case BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH:
      case BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH:
      case BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH:
      case BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH:
      case BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH:
      case BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH:
      case BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH:
        return true;
      default:
        return false;
    }
  }
};

#ifdef USE_COMPOSABLE_KERNEL

template <typename T, int NUM_KV_HEADS>
using InstanceFactory = internal::device_grouped_query_attention_instances<
          2, 1, 1, 1, 1,
          T, internal::F32, NUM_KV_HEADS>;

template <typename T, int NUM_KV_HEADS>
auto GetCKGroupedQueryAttentionTypeStringAndOps() {
  using CKDataType = typename CKDataTypeAdaptor<T>::type;

  std::vector<std::pair<std::string, Op<GroupedQueryAttentionParams<T>>>> ret;

  ck::static_for<0, std::tuple_size_v<InstanceFactory<CKDataType, NUM_KV_HEADS>>, 1>{}([&](auto i) {
    auto inst = std::get<i>(InstanceFactory<CKDataType, NUM_KV_HEADS>{});
    using NewOpInstance = ck::remove_cvref_t<decltype(inst)>;
    auto impl = std::make_unique<NewOpInstance>(NewOpInstance());
    auto type_string = impl->GetTypeString();

    auto op = [impl = std::move(impl)](const GroupedQueryAttentionParams<T>* params) -> Status {
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          params->attention->kv_num_heads != NUM_KV_HEADS,
          "attention is not for kv_num_heads: ", params->attention->kv_num_heads);
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          !GroupedQueryAttentionTunableOp<T>::IsSupportedMode(params->attention),
          "attention mode is not supported, got ", params->attention->mode);

      auto attn = params->attention;
      const int& G0 = attn->batch_size;
      const int& G1 = attn->num_heads;
      const int& M = attn->sequence_length;
      const int& N = attn->total_sequence_length;
      const int& K = attn->head_size;
      const int& O = attn->v_head_size;

      // in permute means A shape[B, S, N, H] else [B, N, S, H]
      bool in_permute = attn->qkv_format == Q_K_V_BSNH;

      // out permute means C shape[B, S, N, H] else [B, N, S, H]
      // output should be [B, S, N, H]
      bool out_permute = true;
      auto arg = impl->MakeArgument(
          reinterpret_cast<const CKDataType*>(params->q_buffer),
          reinterpret_cast<const CKDataType*>(params->k_buffer),
          reinterpret_cast<const CKDataType*>(params->v_buffer),
          reinterpret_cast<CKDataType*>(params->out_buffer),
          M,
          N,
          K,
          O,
          G0,
          G1,
          params->scale,
          in_permute,
          out_permute);

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!impl->IsSupportedArgument(arg),
                                                impl->GetTypeString(), " does not support the params");
      auto invoker = impl->MakeInvoker();
      invoker.Run(arg, StreamConfig{params->StreamHandle()});
      return Status::OK();
    };
    ret.emplace_back(std::make_pair(std::move(type_string), std::move(op)));
  });

  return ret;
}
#endif  // USE_COMPOSABLE_KERNEL

template <typename T>
GroupedQueryAttentionTunableOp<T>::GroupedQueryAttentionTunableOp() {
#ifdef USE_COMPOSABLE_KERNEL
  for (auto&& [_, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 1 /*kv_heads*/>()) {
    this->RegisterOp(std::move(op));
  }
  for (auto&& [_, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 2 /*kv_heads*/>()) {
    this->RegisterOp(std::move(op));
  }
  for (auto&& [_, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 4 /*kv_heads*/>()) {
    this->RegisterOp(std::move(op));
  }
  for (auto&& [_, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 8 /*kv_heads*/>()) {
    this->RegisterOp(std::move(op));
  }
  for (auto&& [_, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 16 /*kv_heads*/>()) {
    this->RegisterOp(std::move(op));
  }
  for (auto&& [_, op] : GetCKGroupedQueryAttentionTypeStringAndOps<T, 32 /*kv_heads*/>()) {
    this->RegisterOp(std::move(op));
  }
#else
  ORT_THROW("Composable kernel is required for GroupedQueryAttentionTunableOp");
#endif
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
