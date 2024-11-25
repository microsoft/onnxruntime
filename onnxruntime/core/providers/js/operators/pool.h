// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/pool_base.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

#define POOL_ATTRIBUTES_JS_OBJ_MAPPING ({                                         \
  "format" : $13 ? "NHWC" : "NCHW",                                               \
  "auto_pad" : $1,                                                                \
  "ceil_mode" : $2,                                                               \
  "count_include_pad" : $3,                                                       \
  "storage_order" : $4,                                                           \
  "dilations" : $5 ? Array.from(HEAP32.subarray(Number($5), Number($6))) : [],    \
  "kernel_shape" : $7 ? Array.from(HEAP32.subarray(Number($7), Number($8))) : [], \
  "pads" : $9 ? Array.from(HEAP32.subarray(Number($9), Number($10))) : [],        \
  "strides" : $11 ? Array.from(HEAP32.subarray(Number($11), Number($12))) : []    \
})

#define POOL_ATTRIBUTES_PARAM_LIST                         \
  static_cast<int32_t>(pool_attrs_.auto_pad),              \
      static_cast<int32_t>(pool_attrs_.ceil_mode),         \
      static_cast<int32_t>(pool_attrs_.count_include_pad), \
      static_cast<int32_t>(pool_attrs_.storage_order),     \
      JSEP_HEAP32_INDEX_START(dilations),                  \
      JSEP_HEAP32_INDEX_END(dilations),                    \
      JSEP_HEAP32_INDEX_START(kernel_shapes),              \
      JSEP_HEAP32_INDEX_END(kernel_shapes),                \
      JSEP_HEAP32_INDEX_START(pads),                       \
      JSEP_HEAP32_INDEX_END(pads),                         \
      JSEP_HEAP32_INDEX_START(strides),                    \
      JSEP_HEAP32_INDEX_END(strides),                      \
      static_cast<int32_t>(is_channels_last)

#define GLOBAL_POOL_ATTRIBUTES_JS_OBJ_MAPPING ({"format" : $1 ? "NHWC" : "NCHW"})
#define GLOBAL_POOL_ATTRIBUTES_PARAM_LIST static_cast<int32_t>(is_channels_last)

template <typename Type>
inline const std::vector<Type> CastTensorShapeVector(const TensorShapeVector& shape) {
  std::vector<Type> castedShapes(shape.size(), 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    castedShapes[i] = gsl::narrow_cast<Type>(shape[i]);
  }
  return castedShapes;
}

template <typename PoolType, bool is_channels_last>
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
      auto kernel_shapes{CastTensorShapeVector<int32_t>(pool_attrs_.kernel_shape)};
      auto strides{CastTensorShapeVector<int32_t>(pool_attrs_.strides)};
      auto dilations{CastTensorShapeVector<int32_t>(pool_attrs_.dilations)};
      auto pads{CastTensorShapeVector<int32_t>(pool_attrs_.pads)};
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

template <bool is_channels_last>
class Pool<MaxPool<8>, is_channels_last> final : public Pool<MaxPool<1>, is_channels_last> {
 public:
  Pool(const OpKernelInfo& info) : Pool<MaxPool<1>, is_channels_last>(info) {}
};

}  // namespace js
}  // namespace onnxruntime
