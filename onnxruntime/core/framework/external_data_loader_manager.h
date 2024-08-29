// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/common/common.h"
#include "core/framework/external_data_loader.h"

namespace onnxruntime {

// The external data loader manager manages all registered external data loaders to allow custom
// external data loading implemented by execution providers.
class ExternalDataLoaderManager {
 public:
  ExternalDataLoaderManager() = default;

  common::Status RegisterExternalDataLoader(std::unique_ptr<IExternalDataLoader> external_data_loader);

  const IExternalDataLoader* GetExternalDataLoader(const OrtMemoryInfo& target_memory_info) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExternalDataLoaderManager);

  // It's assumed that external data loaders in this array have no overlap in terms of copying functionality.
  std::vector<std::unique_ptr<IExternalDataLoader>> external_data_loaders_;
};
}  // namespace onnxruntime
