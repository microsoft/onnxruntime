// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/ml_value.h"
#include "core/framework/mem_buffer.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
class Tensor;
namespace utils {
std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto);
/**
 * deserialize a TensorProto into a preallocated memory buffer.
 */
common::Status TensorProtoToMLValue(const Env& env, const ORTCHAR_T* tensor_proto_path,
                                    const ONNX_NAMESPACE::TensorProto& input, const MemBuffer& m, MLValue& value,
                                    OrtDeleter& deleter);
// This function doesn't support string tensors
ONNX_NAMESPACE::TensorProto::DataType GetTensorProtoType(const Tensor& tensor);

ONNXTensorElementDataType GetTensorElementType(const ONNX_NAMESPACE::TensorProto& tensor_proto);

// How much memory it will need for putting the content of this tensor into a plain array
// complex64/complex128 tensors are not supported.
// The output value could be zero or -1.
template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);

template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ T* p_data, int64_t expected_size);

}  // namespace utils
}  // namespace onnxruntime
