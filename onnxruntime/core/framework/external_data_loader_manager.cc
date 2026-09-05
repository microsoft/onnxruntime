// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader_manager.h"

#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
#include <algorithm>

#include "core/framework/tensor.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#endif

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

#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
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

const IExternalDataLoader* ExternalDataLoaderManager::GetTensorCreator(
    const OrtDevice& target_device, int32_t tensor_data_type) const {
  for (const auto& external_data_loader : external_data_loaders_) {
    if (external_data_loader->SupportsDataType(tensor_data_type) &&
        external_data_loader->CreatesTensorForDevice(target_device)) {
      return external_data_loader.get();
    }
  }

  return nullptr;
}

bool ExternalDataLoaderManager::HasPreloader() const {
  return std::any_of(
      external_data_loaders_.begin(), external_data_loaders_.end(),
      [](const auto& loader) { return loader->SupportsPreload(); });
}

Status ExternalDataLoaderManager::PreloadExternalData(
    const Env& env,
    const std::filesystem::path& model_path,
    const Graph& graph,
    const std::unordered_set<std::string>& excluded_initializer_names,
    const std::unordered_set<PathString>& excluded_external_data_files,
    const std::function<bool()>& is_cancelled) const {
  bool has_preloader = false;
  for (const auto& loader : external_data_loaders_) {
    if (loader->SupportsPreload()) {
      has_preloader = true;
      auto status = loader->BeginPreload();
      if (!status.IsOK()) {
        AbortLoad();
        return status;
      }
    }
  }
  if (!has_preloader) {
    return Status::OK();
  }

  bool preload_finalized = false;
  auto abort_preload = gsl::finally([&]() {
    if (!preload_finalized) {
      AbortLoad();
    }
  });

  std::unordered_set<std::filesystem::path> validated_external_files;
  for (const auto& [name, tensor_proto] : graph.GetAllInitializedTensors()) {
    if (excluded_initializer_names.contains(name) ||
        !utils::HasExternalData(*tensor_proto) ||
        utils::HasExternalDataInMemory(*tensor_proto)) {
      continue;
    }
    if (!excluded_external_data_files.empty()) {
      std::unique_ptr<ExternalDataInfo> external_data_info;
      ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto->external_data(), external_data_info));
      if (excluded_external_data_files.contains(external_data_info->GetRelPath())) {
        continue;
      }
    }
    if (is_cancelled && is_cancelled()) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, MODEL_LOAD_CANCELED,
          "Preloading external weights was canceled due to user request.");
    }
    for (const auto& loader : external_data_loaders_) {
      if (loader->SupportsPreload() &&
          loader->SupportsDataType(tensor_proto->data_type())) {
        ORT_RETURN_IF_ERROR(utils::PrepareExtDataForTensorFromTensorProto(
            env, model_path, *tensor_proto, *loader, true,
            &validated_external_files));
      }
    }
  }

  for (const auto& loader : external_data_loaders_) {
    if (loader->SupportsPreload()) {
      ORT_RETURN_IF_ERROR(loader->FinalizePreload(is_cancelled));
    }
  }
  preload_finalized = true;
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

Status ExternalDataLoaderManager::FinalizeLoad(const std::function<bool()>& is_cancelled) const {
  for (const auto& external_data_loader : external_data_loaders_) {
    auto status = external_data_loader->FinalizeLoad(is_cancelled);
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
