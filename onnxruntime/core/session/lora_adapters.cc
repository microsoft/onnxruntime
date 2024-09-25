// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora_adapters.h"
#include "lora/adapter_format_utils.h"

#include "core/framework/data_transfer.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_adapters.h"
#include "core/session/ort_apis.h"

#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_provider_options.h"

namespace onnxruntime {

ProviderInfo_CUDA* TryGetProviderInfo_CUDA();

namespace lora {

LoraAdapter::Param::Param(OrtValue ort_value_mapped) noexcept
    : ort_value_mapped_(std::move(ort_value_mapped)) {}

LoraAdapter::Param::Param(OrtValue ort_value_mapped, OrtValue ort_value_device) noexcept
    : ort_value_mapped_(std::move(ort_value_mapped)), ort_value_device_(std::move(ort_value_device)) {
}

void LoraAdapter::Load(const std::filesystem::path& file_path) {
  auto buffer = adapters::utils::LoadLoraAdapterBytes(file_path);
  Load(std::move(buffer));
}

void LoraAdapter::Load(std::vector<uint8_t> buffer) {
  adapter_ = adapters::utils::ValidateAndGetAdapterFromBytes(buffer);
  buffer_.emplace<BufferHolder>(std::move(buffer));
  InitializeParamsValues();
}

void LoraAdapter::MemoryMap(const std::filesystem::path& file_path) {
  auto [mapped_memory, file_size] = adapters::utils::MemoryMapAdapterFile(file_path);
  auto u8_span = ReinterpretAsSpan<const uint8_t>(gsl::make_span(mapped_memory.get(), file_size));
  adapter_ = adapters::utils::ValidateAndGetAdapterFromBytes(u8_span);
  buffer_.emplace<MemMapHolder>(std::move(mapped_memory), file_size);

  InitializeParamsValues();
}

static Status GetDataTransfer(const OrtMemoryInfo& mem_info,
                              std::unique_ptr<IDataTransfer>& data_transfer) {
  ORT_RETURN_IF(mem_info.device.Type() == OrtDevice::CPU, "Destination must not be on CPU");
  if (strcmp(mem_info.name, onnxruntime::CUDA) == 0) {
    auto* cuda_provider_info = TryGetProviderInfo_CUDA();
    if (cuda_provider_info != nullptr) {
      data_transfer = cuda_provider_info->CreateGPUDataTransfer();
    }
  }

  if (data_transfer == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Destination provider: ",
                           mem_info.name, " not available, copy failed");
  }

  return Status::OK();
}

static Status CreateOrtValueOnDevice(const OrtValue& ort_value_mapped,
                                     const AllocatorPtr& device_allocator,
                                     const IDataTransfer& data_transfer,
                                     OrtValue& out) {
  OrtValue result;
  const auto& src = ort_value_mapped.Get<Tensor>();
  Tensor on_device(src.DataType(), src.Shape(), device_allocator);
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(src, on_device));
  Tensor::InitOrtValue(std::move(on_device), result);
  out = std::move(result);
  return Status::OK();
}

void LoraAdapter::InitializeParamsValues() {
  if (adapter_ == nullptr) {
    ORT_THROW("Adapter is not loaded yet.");
  }

  std::unique_ptr<IDataTransfer> data_transfer;
  if (device_allocator_) {
    ORT_THROW_IF_ERROR(GetDataTransfer(device_allocator_->Info(), data_transfer));
  }

  const auto* params = adapter_->parameters();
  std::unordered_map<std::string, Param> params_values;
  params_values.reserve(params->size());
  for (const auto* param : *params) {
    auto [name, ort_value] = adapters::utils::CreateOrtValueOverLoraParameter(*param);
    if (device_allocator_) {
      OrtValue ort_value_ondevice;
      ORT_THROW_IF_ERROR(CreateOrtValueOnDevice(ort_value, device_allocator_,
                                                *data_transfer, ort_value_ondevice));
      Param lora_param(std::move(ort_value), std::move(ort_value_ondevice));
      params_values.emplace(std::move(name), std::move(lora_param));
    } else {
      Param lora_param(std::move(ort_value));
      params_values.emplace(std::move(name), std::move(lora_param));
    }
  }
  params_values_.swap(params_values);
}

size_t LoraAdapter::GetBufferSize() const {
  if (std::holds_alternative<MemMapHolder>(buffer_)) {
    return std::get<1>(buffer_).file_size_;
  } else if (std::holds_alternative<BufferHolder>(buffer_)) {
    return std::get<2>(buffer_).buffer_.size();
  }
  ORT_THROW("Non-exhaustive visitor for BinaryFormatHolder::GetSize()");
}

}  // namespace lora
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtApis::CreateLoraAdapter, _In_ const ORTCHAR_T* adapter_file_path, _In_ OrtAllocator* allocator,
                    _Outptr_ OrtLoraAdapter** adapter) {
  API_IMPL_BEGIN

  std::unique_ptr<onnxruntime::lora::LoraAdapter> lora_adapter;
  if (allocator != nullptr) {
    auto alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
    lora_adapter = std::make_unique<onnxruntime::lora::LoraAdapter>(std::move(alloc_ptr));
  } else {
    lora_adapter = std::make_unique<onnxruntime::lora::LoraAdapter>();
  }
  // For platforms that do not support Memmap, we can #ifdef it to ->Load(adapter_file_path)
  lora_adapter->MemoryMap(adapter_file_path);
  *adapter = reinterpret_cast<OrtLoraAdapter*>(lora_adapter.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseLoraAdapter, _Frees_ptr_opt_ OrtLoraAdapter* adapter) {
  delete reinterpret_cast<onnxruntime::lora::LoraAdapter*>(adapter);
}
