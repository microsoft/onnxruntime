// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>
#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"

namespace ONNX_NAMESPACE {
class TensorProto;
}
namespace onnxruntime {
namespace utils {
//How much memory it will need for putting the content of this tensor into a plain array
//string/complex64/complex128 tensors are not supported.
//The output value could be zero or -1.
template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);
class TensorUtils {
 public:
  template <typename T>
  static Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor,
                             /*out*/ T* p_data,
                             int64_t expected_size);

};  // namespace Utils
}  // namespace utils
}  // namespace onnxruntime
