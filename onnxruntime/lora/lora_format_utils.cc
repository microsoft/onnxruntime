// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora_format_utils.h"
#include "lora_format_version.h"

#include "core/common/common.h"
#include "core/common/span_utils.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"

#include <fstream>

namespace onnxruntime {
namespace lora {
namespace utils {

bool IsLoraFormatModelBytes(const void* bytes, size_t num_bytes) {
  return num_bytes > 8 &&  // check buffer is large enough to contain identifier so we don't read random memory
         AdapterBufferHasIdentifier(bytes);
}

flatbuffers::Offset<flatbuffers::String> SaveStringToLoraFormat(flatbuffers::FlatBufferBuilder& builder,
                                                                bool has_string, const std::string& src) {
  if (has_string) return builder.CreateString(src);

  // If the string does not exist, return 0 (the string does not exist in flatbuffer)
  return 0;
}

void LoadStringFromLoraFormat(std::string& dst, const flatbuffers::String* fbs_string) {
  if (fbs_string) {
    dst = fbs_string->str();
  }
}

std::vector<uint8_t> LoadLoraAdapterBytes(const std::filesystem::path& file_path) {
  Env& env = Env::Default();

  size_t file_size = 0;
  ORT_THROW_IF_ERROR(env.GetFileLength(file_path.c_str(), file_size));

  std::vector<uint8_t> result;
  result.resize(file_size);

  // The API accepts char span, so we need to reinterpret the uint8_t span as char span
  auto dest_span = ReinterpretAsSpan<char>(AsSpan(result));
  ORT_THROW_IF_ERROR(env.ReadFileIntoBuffer(file_path.c_str(), 0, file_size, dest_span));

  return result;
}

std::pair<Env::MappedMemoryPtr, size_t> MemoryMapAdapterFile(const std::filesystem::path& file_path) {
  Env& env = Env::Default();

  size_t file_size = 0;
  ORT_THROW_IF_ERROR(env.GetFileLength(file_path.c_str(), file_size));

  Env::MappedMemoryPtr result;
  ORT_THROW_IF_ERROR(env.MapFileIntoMemory(file_path.c_str(), 0, file_size, result));

  return {std::move(result), file_size};
}

const Adapter* ValidateAndGetAdapterFromBytes(gsl::span<const uint8_t> bytes) {
  if (!IsLoraFormatModelBytes(bytes.data(), bytes.size())) {
    ORT_THROW("The buffer does not appear to be a valid lora parameter format");
  }

  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!VerifyAdapterBuffer(verifier)) {
    ORT_THROW("The buffer fails lora adapter format verification");
  }

  auto* adapter = GetAdapter(bytes.data());
  if (!IsLoraFormatVersionSupported(adapter->format_version())) {
    ORT_THROW("Unsupported lora format version");
  }

  return adapter;
}

void SaveLoraParameter(flatbuffers::FlatBufferBuilder& flat_builder, std::string_view name,
                       TensorDataType data_type, gsl::span<const int64_t> shape,
                       gsl::span<const uint8_t> data,
                       flatbuffers::Offset<Parameter>& fbs_tensor) {
  auto name_str = (name.empty()) ? 0 : flat_builder.CreateString(name.data(), name.size());
  auto shape_vec = flat_builder.CreateVector(shape.data(), shape.size());
  auto data_vec = flat_builder.CreateVector(data.data(), data.size());

  fbs_tensor = CreateParameter(flat_builder, name_str, shape_vec, data_type, data_vec);
}

std::pair<std::string, OrtValue> CreateOrtValueOverLoraParameter(const Parameter& param) {
  OrtValue result;

  std::string name;
  LoadStringFromLoraFormat(name, param.name());

  const auto data_type = param.data_type();
  gsl::span<const int64_t> shape_span(param.dims()->data(), param.dims()->size());

  static const OrtMemoryInfo cpu_meminfo(CPU, OrtAllocatorType::OrtDeviceAllocator);

  auto elem_type = DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int32_t>(data_type))->GetElementType();
  // const_cast is necessery due to Tensor class API
  Tensor::InitOrtValue(elem_type,
                       TensorShape(shape_span),
                       const_cast<uint8_t*>(param.raw_data()->data()),
                       cpu_meminfo,
                       result);

  return std::make_pair(std::move(name), std::move(result));
}

OrtValue CreateOrtValueOnDevice(const OrtValue& ort_value_mapped, const AllocatorPtr& device_allocator) {
  OrtValue result;

  const auto& tensor = ort_value_mapped.Get<Tensor>();
  Tensor on_device(tensor.DataType(), tensor.Shape(), device_allocator);

  return result;
}

}  // namespace utils
}  // namespace lora
}  // namespace onnxruntime
