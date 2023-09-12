// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include "core/providers/providers.h"

namespace onnxruntime {

struct ShlProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const std::unordered_map<std::string, std::string>& config);
};

}  // namespace onnxruntime
