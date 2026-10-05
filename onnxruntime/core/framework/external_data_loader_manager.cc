// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader_manager.h"

#include "core/framework/allocator.h"

namespace onnxruntime {
using namespace common;

Status ExternalDataLoaderManager::RegisterExternalDataLoader(std::unique_ptr<IExternalDataLoader> external_data_loader) {
  if (nullptr == external_data_loader) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "external_data_loader registered is nullptr.");
  }
  external_data_loaders_.push_back(std::move(external_data_loader));
  return Status::OK();
}

const IExternalDataLoader* ExternalDataLoaderManager::GetExternalDataLoader(const OrtMemoryInfo& target_memory_info) const {
  for (auto& external_data_loader : external_data_loaders_) {
    if (!external_data_loader->CanLoad(target_memory_info)) {
      continue;
    }

    return external_data_loader.get();
  }
  return nullptr;
}

const IExternalDataLoader* ExternalDataLoaderManager::GetExternalDataLoader(
    const OrtMemoryInfo& target_memory_info, int32_t tensor_data_type) const {
  for (auto& external_data_loader : external_data_loaders_) {
    if (external_data_loader->CanLoad(target_memory_info) &&
        external_data_loader->SupportsDataType(tensor_data_type)) {
      return external_data_loader.get();
    }
  }
  return nullptr;
}

#if defined(ENABLE_D3D12_FILE_LOADING)
Status ExternalDataLoaderManager::GetTensorCreator(
    const OrtDevice& target_device,
    int32_t tensor_data_type,
    const std::shared_ptr<IAllocator>& allocator,
    const IExternalDataLoader*& tensor_creator) const {
  tensor_creator = nullptr;
  for (const auto& external_data_loader : external_data_loaders_) {
    if (!external_data_loader->SupportsDataType(tensor_data_type) ||
        !external_data_loader->CreatesTensorForDevice(target_device)) {
      continue;
    }

    bool can_create_tensor = false;
    ORT_RETURN_IF_ERROR(
        external_data_loader->CanCreateTensorWithAllocator(allocator, can_create_tensor));
    if (can_create_tensor) {
      tensor_creator = external_data_loader.get();
      break;
    }
  }

  return Status::OK();
}

Status ExternalDataLoaderManager::GetExternalDataLoader(
    const OrtMemoryInfo& target_memory_info,
    int32_t tensor_data_type,
    const std::shared_ptr<IAllocator>& allocator,
    const IExternalDataLoader*& selected_loader) const {
  selected_loader = nullptr;
  for (const auto& external_data_loader : external_data_loaders_) {
    if (!external_data_loader->SupportsDataType(tensor_data_type)) {
      continue;
    }

    if (external_data_loader->CreatesTensorForDevice(target_memory_info.device)) {
      bool can_create_tensor = false;
      ORT_RETURN_IF_ERROR(
          external_data_loader->CanCreateTensorWithAllocator(allocator, can_create_tensor));
      if (!can_create_tensor) {
        continue;
      }
    } else if (!external_data_loader->CanLoad(target_memory_info)) {
      continue;
    }

    selected_loader = external_data_loader.get();
    break;
  }

  return Status::OK();
}

Status ExternalDataLoaderManager::BeginLoad() const {
  for (const auto& external_data_loader : external_data_loaders_) {
    auto status = external_data_loader->BeginLoad();
    if (!status.IsOK()) {
      AbortLoad();
      return status;
    }
  }

  return Status::OK();
}

Status ExternalDataLoaderManager::FinalizeLoad(const std::function<bool()>& is_canceled) const {
  for (const auto& external_data_loader : external_data_loaders_) {
    auto status = external_data_loader->FinalizeLoad(is_canceled);
    if (!status.IsOK()) {
      AbortLoad();
      return status;
    }
  }

  return Status::OK();
}

void ExternalDataLoaderManager::AbortLoad() const noexcept {
  for (const auto& external_data_loader : external_data_loaders_) {
    external_data_loader->AbortLoad();
  }
}
#endif

}  // namespace onnxruntime
