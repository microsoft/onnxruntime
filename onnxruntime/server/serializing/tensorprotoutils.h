// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/ml_value.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "mem_buffer.h"

#include "onnx-ml.pb.h"
#include "predict.pb.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace server {
// How much memory it will need for putting the content of this tensor into a plain array
// complex64/complex128 tensors are not supported.
// The output value could be zero or -1.
template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);
/**
 * deserialize a TensorProto into a preallocated memory buffer.
 *  Impl must correspond to onnxruntime/core/framework/tensorprotoutils.cc
 * This implementation does not support external data so as to reduce dependency surface.
 */
common::Status TensorProtoToMLValue(const ONNX_NAMESPACE::TensorProto& input, const server::MemBuffer& m, /* out */ Ort::Value& value);

template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ T* p_data, int64_t expected_size);

ONNXTensorElementDataType CApiElementTypeFromProtoType(int type);
ONNXTensorElementDataType GetTensorElementType(const ONNX_NAMESPACE::TensorProto& tensor_proto);
}
}