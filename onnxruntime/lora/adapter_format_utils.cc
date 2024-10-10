// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "adapter_format_utils.h"
#include "adapter_format_version.h"

#include "core/framework/allocator.h"
#include "core/common/common.h"
#include "core/framework/endian.h"
#include "core/framework/endian_utils.h"
#include "core/common/span_utils.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"

#include <fstream>

namespace onnxruntime {
namespace adapters {
namespace utils {

bool IsAdapterFormatModelBytes(const void* bytes, size_t num_bytes) {
  return num_bytes > 8 &&  // check buffer is large enough to contain identifier so we don't read random memory
         AdapterBufferHasIdentifier(bytes);
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
  if (!IsAdapterFormatModelBytes(bytes.data(), bytes.size())) {
    ORT_THROW("The buffer does not appear to be a valid lora parameter format");
  }

  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!VerifyAdapterBuffer(verifier)) {
    ORT_THROW("The buffer fails lora adapter format verification");
  }

  auto* adapter = GetAdapter(bytes.data());
  if (!IsAdapterFormatVersionSupported(adapter->format_version())) {
    ORT_THROW("Unsupported lora format version");
  }

  return adapter;
}

template <class T>
struct WriteDataForLittleEndian {
  Status operator()(gsl::span<const uint8_t> src, gsl::span<uint8_t> dest) const {
    auto src_span = ReinterpretAsSpan<const T>(src);
    auto dest_span = ReinterpretAsSpan<unsigned char>(dest);
    return onnxruntime::utils::WriteLittleEndian<T>(src_span, dest_span);
  }
};

void SaveLoraParameter(flatbuffers::FlatBufferBuilder& flat_builder, std::string_view name,
                       TensorDataType data_type, gsl::span<const int64_t> shape,
                       gsl::span<const uint8_t> data,
                       flatbuffers::Offset<Parameter>& fbs_tensor) {
  auto name_str = (name.empty()) ? 0 : flat_builder.CreateString(name.data(), name.size());
  auto shape_vec = flat_builder.CreateVector(shape.data(), shape.size());

  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_vec;
  if constexpr (endian::native == endian::big) {
    onnxruntime::utils::MLTypeCallDispatcher<float, double, int8_t, uint8_t,
                                             int16_t, uint16_t, int32_t, uint32_t,
                                             int64_t, uint64_t,
                                             BFloat16, MLFloat16>
        disp(static_cast<int32_t>(data_type));

    InlinedVector<uint8_t> be_data(data.size());
    auto status = disp.InvokeRet<Status, WriteDataForLittleEndian>(data, be_data);
    ORT_THROW_IF_ERROR(status);
    data_vec = flat_builder.CreateVector<uint8_t>(be_data.data(), be_data.size());
  } else {
    data_vec = flat_builder.CreateVector(data.data(), data.size());
  }
  fbs_tensor = CreateParameter(flat_builder, name_str, shape_vec, data_type, data_vec);
}

template <class T>
struct ReadDataForBigEndian {
  Status operator()(gsl::span<const uint8_t> src, Tensor& dst) const {
    auto src_span = ReinterpretAsSpan<const unsigned char>(src);
    auto dst_span = dst.MutableDataAsSpan<T>();
    return onnxruntime::utils::ReadLittleEndian<T>(src_span, dst_span);
  }
};

std::pair<std::string, OrtValue> CreateOrtValueOverLoraParameter(const Parameter& param) {
  OrtValue result;

  std::string name;
  LoadStringFromLoraFormat(name, param.name());

  const auto data_type = param.data_type();
  // Copying shape takes care of endianess using flatbuffers accessors
  TensorShapeVector shape(param.dims()->begin(), param.dims()->end());
  static const AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  const auto elem_type = DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int32_t>(data_type))->GetElementType();

  // If BE, we a allocate memory within the tensor and copy there swapping bytes
  if constexpr (endian::native == endian::big) {
    gsl::span<const uint8_t> src_span(param.raw_data()->data(), param.raw_data()->size());
    Tensor tensor(elem_type, shape, cpu_allocator);
    onnxruntime::utils::MLTypeCallDispatcher<float, double, int8_t, uint8_t,
                                             int16_t, uint16_t, int32_t, uint32_t,
                                             int64_t, uint64_t,
                                             BFloat16, MLFloat16>
        disp(static_cast<int32_t>(data_type));

    auto status = disp.InvokeRet<Status, ReadDataForBigEndian>(src_span, tensor);
    ORT_THROW_IF_ERROR(status);
    Tensor::InitOrtValue(std::move(tensor), result);
  } else {
    // const_cast is necessary due to Tensor class API
    Tensor::InitOrtValue(elem_type,
                         TensorShape(shape),
                         const_cast<uint8_t*>(param.raw_data()->data()),
                         cpu_allocator->Info(),
                         result);
  }

  return std::make_pair(std::move(name), std::move(result));
}

void AdapterFormatBuilder::AddParameter(const std::string& name, TensorDataType data_type,
                                        gsl::span<const int64_t> shape, gsl::span<const uint8_t> data) {
  flatbuffers::Offset<Parameter> fbs_param;
  SaveLoraParameter(builder_, name, data_type, shape, data, fbs_param);
  params_.push_back(fbs_param);
}

std::vector<uint8_t> AdapterFormatBuilder::Finish(int adapter_version, int model_version) {
  FinishImpl(adapter_version, model_version);

  std::vector<uint8_t> result;
  result.reserve(builder_.GetSize());
  gsl::span<uint8_t> buffer(builder_.GetBufferPointer(), builder_.GetSize());
  std::copy(buffer.begin(), buffer.end(), std::back_inserter(result));
  return result;
}

gsl::span<uint8_t> AdapterFormatBuilder::FinishWithSpan(int adapter_version, int model_version) {
  FinishImpl(adapter_version, model_version);
  return gsl::make_span(builder_.GetBufferPointer(), builder_.GetSize());
}

void AdapterFormatBuilder::FinishImpl(int adapter_version, int model_version) {
  auto fbs_params = builder_.CreateVector(params_);
  auto fbs_adapter = CreateAdapter(builder_, kAdapterFormatVersion, adapter_version,
                                   model_version, fbs_params);
  builder_.Finish(fbs_adapter, AdapterIdentifier());
}

}  // namespace utils
}  // namespace adapters
}  // namespace onnxruntime
