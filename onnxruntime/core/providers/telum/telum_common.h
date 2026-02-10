// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

// zDNN headers
extern "C" {
#include "zdnn/zdnn.h"
}

namespace onnxruntime {
namespace telum {

// Telum EP constants
constexpr const char* TELUM = "TelumExecutionProvider";
constexpr size_t ZDNN_ALIGNMENT = 4096;  // 4K alignment required by zDNN

// Configuration options for Telum EP
struct TelumExecutionProviderInfo {
  // Enable strict mode: reject unsupported ops instead of silent fallback
  bool strict_mode = true;

  // Enable operator fusion optimizations
  bool enable_fusion = true;

  // Log fallback decisions for debugging
  bool log_fallbacks = true;

  // Maximum batch size for validation
  size_t max_batch_size = 32;

  // Maximum sequence length for transformer models
  size_t max_sequence_length = 512;

  // Create arena allocator
  bool create_arena = true;

  explicit TelumExecutionProviderInfo(bool use_arena = true)
      : create_arena(use_arena) {}
};

// Helper function to convert zDNN status to ORT Status
inline Status CheckZDNNStatus(zdnn_status status, const char* operation) {
  if (status == ZDNN_OK) {
    return Status::OK();
  }

  const char* error_msg = zdnn_get_status_message(status);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                        "zDNN operation '", operation, "' failed with status ",
                        status, ": ", error_msg);
}

// Helper to validate static shapes
inline Status ValidateStaticShape(const TensorShape& shape) {
  for (auto dim : shape.GetDims()) {
    if (dim < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Telum EP requires static shapes. Dynamic dimension found: ", dim);
    }
  }
  return Status::OK();
}

// Helper to check if zDNN is available
inline bool IsZDNNAvailable() {
  return zdnn_is_nnpa_installed();
}

// Map ONNX data type to zDNN data type
inline zdnn_data_types MapONNXTypeToZDNN(int32_t onnx_type) {
  switch (onnx_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return FP32;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return FP16;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return BFLOAT;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return INT8;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return INT32;
    default:
      return FP32;  // Default fallback
  }
}

// Get size of zDNN data type in bytes
inline size_t GetZDNNTypeSize(zdnn_data_types type) {
  switch (type) {
    case FP32:
    case INT32:
      return 4;
    case FP16:
    case BFLOAT:
      return 2;
    case INT8:
      return 1;
    case ZDNN_DLFLOAT16:
      return 2;
    default:
      return 4;
  }
}

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
