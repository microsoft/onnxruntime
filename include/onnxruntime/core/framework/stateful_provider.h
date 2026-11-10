// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

inline OrtStatus* ORT_API_CALL StatefulCpuCacheReference(
    void*, const OrtStatefulCacheOperationRequest* request, OrtStatefulCacheOperationResult* out) noexcept {
  return request->cpu_reference(request->cpu_context, out);
}

inline OrtStatus* ORT_API_CALL StatefulCpuGreedyReference(
    void*, const OrtStatefulGreedyRequest* request, int64_t* out) noexcept {
  return request->cpu_reference(request->cpu_context, out);
}

// This header is shared with provider DLLs: only C ABI types cross the boundary.
inline const OrtStatefulProviderDescriptor* GetCpuStatefulProviderDescriptor() noexcept {
  static const OrtStatefulProviderDescriptor descriptor{
      sizeof(OrtStatefulProviderDescriptor),
      {sizeof(OrtStatefulProviderCapabilities), 1, (uint64_t{1} << 7) - 1,
       (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
           (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) |
           (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
       (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) |
           (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
       uint32_t{1} << OrtMemoryInfoDeviceType_CPU, 0},
      nullptr,
      &StatefulCpuCacheReference,
      &StatefulCpuGreedyReference,
      nullptr,
      nullptr};
  return &descriptor;
}

}  // namespace onnxruntime
