// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./tensor_proto.h"

#include <cstdint>
#include <limits>

#include "./vai_assert.h"
#include "core/providers/shared_library/provider_api.h"
namespace vaip {
gsl::span<const char> tensor_proto_as_raw(const ONNX_NAMESPACE::TensorProto& tensor) {
  auto& mut_tensor = const_cast<ONNX_NAMESPACE::TensorProto&>(tensor);
  if (!tensor.has_raw_data()) {
    std::vector<uint8_t> unpacked_tensor;
    auto path = onnxruntime::Path::Create();
    auto s = onnxruntime::utils::UnpackInitializerData(tensor, *path, unpacked_tensor);
    mut_tensor.mutable_raw_data()->resize(unpacked_tensor.size());
    mut_tensor.clear_float_data();
    mut_tensor.clear_int32_data();
    mut_tensor.clear_string_data();
    mut_tensor.clear_int64_data();
    mut_tensor.clear_double_data();
    mut_tensor.clear_uint64_data();
    memcpy(mut_tensor.mutable_raw_data()->data(), unpacked_tensor.data(), unpacked_tensor.size());
  }
  return gsl::span<const char>(tensor.raw_data().data(), tensor.raw_data().size());
}

vaip_core::DllSafe<std::vector<int64_t>> tensor_proto_get_shape(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  auto ret = std::vector<int64_t>();
  int rank = tensor_proto.dims_size();
  if (rank > 0) {
    auto& dims = tensor_proto.dims();
    for (auto i = 0; i < dims.size(); ++i) {
      ret.push_back(dims[i]);
    }
  }
  return vaip_core::DllSafe(ret);
}
static ONNX_NAMESPACE::TensorProto* tensor_proto_new(const std::string& name, const std::vector<int64_t>& shape,
                                                     int data_type, const char* data, size_t data_size) {
  auto tensor_proto = ONNX_NAMESPACE::TensorProto::Create();
  tensor_proto->set_name(name);
  for (auto s : shape) {
    tensor_proto->add_dims(s);
  }
  tensor_proto->set_data_type(data_type);
  tensor_proto->mutable_raw_data()->assign(data, data_size);
  return tensor_proto.release();
}

ONNX_NAMESPACE::TensorProto* tensor_proto_new_i32(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int32_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_INT32,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int32_t));
}

ONNX_NAMESPACE::TensorProto* tensor_proto_new_i64(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int64_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_INT64,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int64_t));
}

ONNX_NAMESPACE::TensorProto* tensor_proto_new_i8(const std::string& name, const std::vector<int64_t>& shape,
                                                 const std::vector<int8_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_INT8,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(int8_t));
}

ONNX_NAMESPACE::TensorProto* tensor_proto_new_floats(const std::string& name, const std::vector<int64_t>& shape,
                                                     const std::vector<float>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(float));
}

}  // namespace vaip
