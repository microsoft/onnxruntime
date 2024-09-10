// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader_manager.h"
#include "core/framework/tensor.h"

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

}  // namespace onnxruntime
