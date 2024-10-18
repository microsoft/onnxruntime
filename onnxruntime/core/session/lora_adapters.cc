// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/lora_adapters.h"
#include "lora/adapter_format_utils.h"

#include "core/framework/data_transfer.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_adapters.h"
#include "core/session/ort_apis.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif

#ifdef USE_DML
#include "core/session/abi_session_options_impl.h"
#include "core/providers/dml/dml_provider_factory_creator.h"
#include "core/providers/dml/dml_provider_factory.h"
#endif

#include <functional>
#include <unordered_map>

namespace onnxruntime {

#ifdef USE_CUDA
ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
#endif

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

namespace {
struct DataTransfer {
  std::unique_ptr<IExecutionProvider> ep;
  std::unique_ptr<IDataTransfer> data_transfer;
  Status CopyTensor(const Tensor& src, Tensor& dst) const {
    return data_transfer->CopyTensor(src, dst);
  }
  Status Sync() const {
#if USE_DML
    return ep->Sync();
#else
    return Status::OK();
#endif
  }
};
}  // namespace

static Status GetDataTransfer(const OrtMemoryInfo& mem_info, [[maybe_unused]] DataTransfer& dt) {
  ORT_RETURN_IF(strcmp(mem_info.name, onnxruntime::CPU) == 0, "Expecting on device allocator for LoraAdapter");

  Status status;
  if (strcmp(mem_info.name, onnxruntime::CUDA) == 0) {
#ifdef USE_CUDA
    auto* cuda_provider_info = TryGetProviderInfo_CUDA();
    if (cuda_provider_info != nullptr) {
      dt.data_transfer = cuda_provider_info->CreateGPUDataTransfer();
    } else {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA provider could not be loaded");
    }
#else
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA provider is not enabled in this build");
#endif
  } else if (strcmp(mem_info.name, onnxruntime::DML) == 0) {
#ifdef USE_DML
    auto ep_factory = onnxruntime::DMLProviderFactoryCreator::Create(ConfigOptions{}, 0, false, false, false);
    dt.ep = ep_factory->CreateProvider();
    dt.data_transfer = dt.ep->GetDataTransfer();
#else
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DML provider is not enabled in this build");
#endif
  } else {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported device allocator");
  }

  return status;
}

static Status CreateOrtValueOnDevice(const OrtValue& ort_value_mapped,
                                     const AllocatorPtr& device_allocator,
                                     const DataTransfer& data_transfer,
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

  DataTransfer data_transfer;
  if (device_allocator_) {
    ORT_THROW_IF_ERROR(GetDataTransfer(device_allocator_->Info(), data_transfer));
  }

  const auto* params = adapter_->parameters();
  ORT_ENFORCE(params != nullptr, "Params absent");
  std::unordered_map<std::string, Param> params_values;
  params_values.reserve(params->size());
  // Re-work in two separate loops due to compiler issues
  if (device_allocator_) {
    for (const auto* param : *params) {
      auto [name, ort_value] = adapters::utils::CreateOrtValueOverLoraParameter(*param);
      OrtValue ort_value_ondevice;
      ORT_THROW_IF_ERROR(CreateOrtValueOnDevice(ort_value, device_allocator_,
                                                data_transfer, ort_value_ondevice));
      Param lora_param(std::move(ort_value), std::move(ort_value_ondevice));
      params_values.emplace(std::move(name), std::move(lora_param));
    }
  } else {
    for (const auto* param : *params) {
      auto [name, ort_value] = adapters::utils::CreateOrtValueOverLoraParameter(*param);
      Param lora_param(std::move(ort_value));
      params_values.emplace(std::move(name), std::move(lora_param));
    }
  }

  if (device_allocator_) {
    ORT_THROW_IF_ERROR(data_transfer.Sync());
  }

  params_values_.swap(params_values);
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

ORT_API_STATUS_IMPL(OrtApis::CreateLoraAdapterFromArray, _In_ const void* bytes, size_t num_bytes,
                    _In_ OrtAllocator* allocator, _Outptr_ OrtLoraAdapter** adapter) {
  API_IMPL_BEGIN

  std::unique_ptr<onnxruntime::lora::LoraAdapter> lora_adapter;
  if (allocator != nullptr) {
    auto alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
    lora_adapter = std::make_unique<onnxruntime::lora::LoraAdapter>(std::move(alloc_ptr));
  } else {
    lora_adapter = std::make_unique<onnxruntime::lora::LoraAdapter>();
  }

  std::vector<uint8_t> buffer(num_bytes);
  memcpy(buffer.data(), bytes, num_bytes);
  lora_adapter->Load(std::move(buffer));
  *adapter = reinterpret_cast<OrtLoraAdapter*>(lora_adapter.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseLoraAdapter, _Frees_ptr_opt_ OrtLoraAdapter* adapter) {
  delete reinterpret_cast<onnxruntime::lora::LoraAdapter*>(adapter);
}
