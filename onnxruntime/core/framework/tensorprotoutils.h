// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/ml_value.h"
#include "core/graph/onnx_protobuf.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
class Tensor;
namespace utils {
common::Status GetTensorFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, std::unique_ptr<Tensor>* p_tensor, AllocatorPtr allocator, void* preallocated = nullptr, size_t preallocated_size = 0);
std::vector<int64_t> GetTensorShapeFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto);
std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto);
common::Status TensorProtoToMLValue(const ONNX_NAMESPACE::TensorProto& input, AllocatorPtr allocator, void* preallocated,
                                    size_t preallocated_size, MLValue& value);
ONNX_NAMESPACE::TensorProto::DataType GetTensorProtoType(const Tensor& tensor);
}  // namespace utils
}  // namespace onnxruntime
