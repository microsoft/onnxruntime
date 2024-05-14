// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnx {
class TensorProto;
}

namespace Ort {
struct Value;
}

namespace onnxruntime {

namespace test {

struct OrtCallback;
class MemBuffer;

// How much memory it will need for putting the content of this tensor into a plain array
// complex64/complex128 tensors are not supported.
// The output value could be zero or -1.
template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const onnx::TensorProto& tensor_proto, size_t* out);
/**
 * deserialize a TensorProto into a preallocated memory buffer.
 *  Impl must correspond to onnxruntime/core/framework/tensorprotoutils.cc
 * This implementation does not support external data so as to reduce dependency surface.
 */
common::Status TensorProtoToMLValue(const onnx::TensorProto& input, const MemBuffer& m, /* out */ Ort::Value& value,
                                    OrtCallback& deleter);

template <typename T>
void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                  /*out*/ T* p_data, size_t expected_size);

ONNXTensorElementDataType CApiElementTypeFromProtoType(int type);
ONNXTensorElementDataType GetTensorElementType(const onnx::TensorProto& tensor_proto);
}  // namespace test
}  // namespace onnxruntime
