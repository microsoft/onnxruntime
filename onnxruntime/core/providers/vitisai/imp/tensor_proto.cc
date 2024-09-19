// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./tensor_proto.h"

#include <cstdint>
#include <limits>

#include "./vai_assert.h"
#include "core/providers/shared_library/provider_api.h"
namespace vaip {
using namespace onnxruntime;

static gsl::span<const char> process_ext_address(const ONNX_NAMESPACE::TensorProto& tensor) {
  auto tensor_proto = const_cast<ONNX_NAMESPACE::TensorProto*>(&tensor);
  auto file = std::string();
  uintptr_t offset = 0;
  size_t size = 0;
  if (tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
    auto external_data = tensor_proto->mutable_external_data();
    auto external_data_size = external_data->size();
    for (auto i = 0; i < external_data_size; ++i) {
      auto& data = external_data->at(i);
      char* end = nullptr;
      if (*data.mutable_key() == "location") {
        file = *data.mutable_value();
      } else if (*data.mutable_key() == "offset") {
        offset = (uintptr_t)std::strtoull(data.mutable_value()->data(), &end, 10);
      } else if (*data.mutable_key() == "length") {
        size = (size_t)std::strtoull(data.mutable_value()->data(), &end, 10);
      } else if (*data.mutable_key() == "checksum") {
        // checksum = (size_t)std::strtoull(data.mutable_value()->data(), &end, 10);
      }
    }
    if (file == "*/_ORT_MEM_ADDR_/*") {
      auto addr = reinterpret_cast<const char*>(offset);
      return {addr, size};
    }
  }
  return {};
}

gsl::span<const char> tensor_proto_as_raw(const onnxruntime::Graph& graph, const ONNX_NAMESPACE::TensorProto& tensor) {
  auto& mut_tensor = const_cast<ONNX_NAMESPACE::TensorProto&>(tensor);
  if (!tensor.has_raw_data()) {
    auto maybe_external_memory_address = process_ext_address(tensor);
    if (!maybe_external_memory_address.empty()) {
      return maybe_external_memory_address;
    }

    std::vector<uint8_t> unpacked_tensor;
    auto path = graph.ModelPath();
    auto s = onnxruntime::utils::UnpackInitializerData(tensor, path, unpacked_tensor);
    vai_assert(s.IsOK(), s.ErrorMessage());
    mut_tensor.mutable_raw_data()->resize(unpacked_tensor.size());
    mut_tensor.clear_float_data();
    mut_tensor.clear_int32_data();
    mut_tensor.clear_string_data();
    mut_tensor.clear_int64_data();
    mut_tensor.clear_double_data();
    mut_tensor.clear_uint64_data();
    memcpy(mut_tensor.mutable_raw_data()->data(), unpacked_tensor.data(), unpacked_tensor.size());
    mut_tensor.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_DEFAULT);
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

ONNX_NAMESPACE::TensorProto* tensor_proto_new_i8(const std::string& name, const std::vector<int64_t>& shape,
                                                 const std::vector<int8_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_INT8,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}

ONNX_NAMESPACE::TensorProto* tensor_proto_new_i16(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int16_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_INT16,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_i32(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int32_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_INT32,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_i64(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int64_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_INT64,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u8(const std::string& name, const std::vector<int64_t>& shape,
                                                 const std::vector<uint8_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u16(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<uint16_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_UINT16,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u32(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<uint32_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_UINT32,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u64(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<uint64_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_UINT64,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}

ONNX_NAMESPACE::TensorProto* tensor_proto_new_floats(const std::string& name, const std::vector<int64_t>& shape,
                                                     const std::vector<float>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_doubles(const std::string& name, const std::vector<int64_t>& shape,
                                                      const std::vector<double>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}

ONNX_NAMESPACE::TensorProto* tensor_proto_new_bf16(const std::string& name, const std::vector<int64_t>& shape,
                                                   const std::vector<int16_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
ONNX_NAMESPACE::TensorProto* tensor_proto_new_fp16(const std::string& name, const std::vector<int64_t>& shape,
                                                   const std::vector<int16_t>& data) {
  return tensor_proto_new(name, shape, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                          reinterpret_cast<const char*>(&data[0]), data.size() * sizeof(data[0]));
}
}  // namespace vaip
