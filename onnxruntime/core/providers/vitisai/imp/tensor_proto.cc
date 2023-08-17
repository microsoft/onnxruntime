// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./tensor_proto.h"
#include "./vai_assert.h"
#include "core/framework/tensorprotoutils.h"

#include <cstdint>
#include <limits>

namespace vaip {

gsl::span<const char> tensor_proto_as_raw(
    const ONNX_NAMESPACE::TensorProto& tensor) {
  auto& mut_tensor = const_cast<ONNX_NAMESPACE::TensorProto&>(tensor);
  if (!tensor.has_raw_data()) {
    std::vector<uint8_t> unpacked_tensor;
    auto s = onnxruntime::utils::UnpackInitializerData(tensor, onnxruntime::Path(), unpacked_tensor);
    mut_tensor.mutable_raw_data()->resize(unpacked_tensor.size());
    memcpy(mut_tensor.mutable_raw_data()->data(), unpacked_tensor.data(), unpacked_tensor.size());
  }
  return gsl::span<const char>(tensor.raw_data().data(), tensor.raw_data().size());
}

size_t tensor_proto_raw_data_size(const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.raw_data().size();
}

std::vector<int64_t> tensor_proto_get_shape(
    const onnx::TensorProto& tensor_proto) {
  auto ret = std::vector<int64_t>();
  int rank = tensor_proto.dims_size();
  if (rank > 0) {
    ret.reserve((size_t)rank);
    for (auto i = 0; i < rank; ++i) {
      ret.push_back(tensor_proto.dims(i));
    }
  }
  return ret;
}

const std::string& tensor_proto_get_name(
    const ONNX_NAMESPACE::TensorProto& tensor) {
  return tensor.name();
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_i32(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::INT32);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int32_t));
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_i64(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int64_t>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::INT64);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int64_t));
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_i8(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<int8_t>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::INT8);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int8_t));
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto tensor_proto_new_floats(
    const std::string& name, const std::vector<int64_t>& shape,
    const std::vector<float>& data) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto();
  tensor_proto.set_name(name);
  tensor_proto.mutable_dims()->Clear();
  tensor_proto.mutable_dims()->Add(shape.begin(), shape.end());
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
  tensor_proto.mutable_raw_data()->assign(
      reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(float));
  return tensor_proto;
}

}  // namespace vaip
