// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

inline Status ValidateStatefulProviderDescriptor(const OrtStatefulProviderDescriptor& descriptor) {
  constexpr uint64_t kCacheOperations = (uint64_t{1} << ORT_STATEFUL_SAMPLE_GREEDY) - 1;
  constexpr uint64_t kOperations = (uint64_t{1} << 7) - 1;
  constexpr uint64_t kCacheTypes = (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
                                   (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) |
                                   (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  constexpr uint64_t kSamplerTypes = kCacheTypes & ~(uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

  ORT_RETURN_IF(descriptor.struct_size <
                    offsetof(OrtStatefulProviderDescriptor, Release) + sizeof(descriptor.Release),
                "Undersized stateful provider descriptor");
  const auto& caps = descriptor.capabilities;
  ORT_RETURN_IF(caps.struct_size <
                    offsetof(OrtStatefulProviderCapabilities, device_id) + sizeof(caps.device_id),
                "Undersized stateful provider descriptor");
  ORT_RETURN_IF(caps.version != 1 || (caps.operations & ~kOperations) != 0 ||
                    (caps.cache_element_types & ~kCacheTypes) != 0 ||
                    (caps.sampler_element_types & ~kSamplerTypes) != 0 ||
                    (caps.memory_types & ~uint32_t{15}) != 0 || caps.device_id < -1,
                "Unsupported stateful descriptor version/capability");
  const bool cache = (caps.operations & kCacheOperations) != 0;
  const bool sampler = (caps.operations & (uint64_t{1} << ORT_STATEFUL_SAMPLE_GREEDY)) != 0;
  ORT_RETURN_IF(cache != (descriptor.CacheOperation != nullptr) ||
                    sampler != (descriptor.Greedy != nullptr) ||
                    (cache && caps.cache_element_types == 0) || (sampler && caps.sampler_element_types == 0) ||
                    (caps.operations != 0 && caps.memory_types == 0) ||
                    (descriptor.context && (!descriptor.Retain || !descriptor.Release)) ||
                    ((descriptor.Retain != nullptr) != (descriptor.Release != nullptr)),
                "Stateful capabilities and callbacks are inconsistent");
  return Status::OK();
}

}  // namespace onnxruntime
