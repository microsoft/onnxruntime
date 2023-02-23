// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "core/providers/providers.h"

namespace onnxruntime {
struct NnapiProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
      uint32_t nnapi_flags, const std::optional<std::string>& partitioning_stop_ops_list);
};
}  // namespace onnxruntime
