// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/pool_base.h"

namespace onnxruntime {
namespace js {

#define POOL_ATTRIBUTES_JS_OBJ_MAPPING ({ \
  "format" : $15 ? "NHWC" : "NCHW",       \
  "auto_pad" : $1,                        \
  "ceil_mode" : $2,                       \
  "count_include_pad" : $3,               \
  "storage_order" : $4,                   \
  "dilations" : [ $5, $6 ],               \
  "kernel_shape" : [ $7, $8 ],            \
  "pads" : [ $9, $10, $11, $12 ],         \
  "strides" : [ $13, $14 ]                \
})

#define POOL_ATTRIBUTES_PARAM_LIST                                                                 \
  static_cast<int32_t>(pool_attrs_.auto_pad),                                                      \
      static_cast<int32_t>(pool_attrs_.ceil_mode),                                                 \
      static_cast<int32_t>(pool_attrs_.count_include_pad),                                         \
      static_cast<int32_t>(pool_attrs_.storage_order),                                             \
      static_cast<int32_t>(pool_attrs_.dilations.size() > 0 ? pool_attrs_.dilations[0] : 0),       \
      static_cast<int32_t>(pool_attrs_.dilations.size() > 1 ? pool_attrs_.dilations[1] : 0),       \
      static_cast<int32_t>(pool_attrs_.kernel_shape.size() > 0 ? pool_attrs_.kernel_shape[0] : 0), \
      static_cast<int32_t>(pool_attrs_.kernel_shape.size() > 1 ? pool_attrs_.kernel_shape[1] : 0), \
      static_cast<int32_t>(pool_attrs_.pads.size() > 0 ? pool_attrs_.pads[0] : 0),                 \
      static_cast<int32_t>(pool_attrs_.pads.size() > 1 ? pool_attrs_.pads[1] : 0),                 \
      static_cast<int32_t>(pool_attrs_.pads.size() > 2 ? pool_attrs_.pads[2] : 0),                 \
      static_cast<int32_t>(pool_attrs_.pads.size() > 3 ? pool_attrs_.pads[3] : 0),                 \
      static_cast<int32_t>(pool_attrs_.strides.size() > 0 ? pool_attrs_.strides[0] : 0),           \
      static_cast<int32_t>(pool_attrs_.strides.size() > 1 ? pool_attrs_.strides[1] : 0),           \
      static_cast<int32_t>(is_channels_last)

#define GLOBAL_POOL_ATTRIBUTES_JS_OBJ_MAPPING ({"format" : $1 ? "NHWC" : "NCHW"})
#define GLOBAL_POOL_ATTRIBUTES_PARAM_LIST static_cast<int32_t>(is_channels_last)

template <typename T, typename PoolType, bool is_channels_last>
class Pool : public JsKernel, public PoolBase {
 public:
  Pool(const OpKernelInfo& info) : JsKernel(info), PoolBase(info) {
    if (pool_attrs_.global_pooling) {
      if constexpr (PoolType::type == onnxruntime::PoolType::kAveragePool) {
        JSEP_INIT_KERNEL_ATTRIBUTE(GlobalAveragePool, GLOBAL_POOL_ATTRIBUTES_JS_OBJ_MAPPING, GLOBAL_POOL_ATTRIBUTES_PARAM_LIST);
      } else if constexpr (PoolType::type == onnxruntime::PoolType::kMaxPool) {
        JSEP_INIT_KERNEL_ATTRIBUTE(GlobalMaxPool, GLOBAL_POOL_ATTRIBUTES_JS_OBJ_MAPPING, GLOBAL_POOL_ATTRIBUTES_PARAM_LIST);
      } else {
        // TODO: GlobalLpPool
      }
    } else {
      if constexpr (PoolType::type == onnxruntime::PoolType::kAveragePool) {
        JSEP_INIT_KERNEL_ATTRIBUTE(AveragePool, POOL_ATTRIBUTES_JS_OBJ_MAPPING, POOL_ATTRIBUTES_PARAM_LIST);
      } else if constexpr (PoolType::type == onnxruntime::PoolType::kMaxPool) {
        JSEP_INIT_KERNEL_ATTRIBUTE(MaxPool, POOL_ATTRIBUTES_JS_OBJ_MAPPING, POOL_ATTRIBUTES_PARAM_LIST);
      } else {
        // TODO: LpPool
      }
    }
  }
};

template <typename T, bool is_channels_last>
class Pool<T, MaxPool<8>, is_channels_last> final : public Pool<T, MaxPool<1>, is_channels_last> {
 public:
  Pool(const OpKernelInfo& info) : Pool<T, MaxPool<1>, is_channels_last>(info) {}
};

}  // namespace js
}  // namespace onnxruntime
