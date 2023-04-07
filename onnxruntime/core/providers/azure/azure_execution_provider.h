// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {

class AzureExecutionProvider : public IExecutionProvider {
 public:
  explicit AzureExecutionProvider(const std::unordered_map<std::string, std::string>& config);
  ~AzureExecutionProvider() = default;
  const std::unordered_map<std::string, std::string>& GetConfig() const { return config_; }

 private:
  std::unordered_map<std::string, std::string> config_;
};

}  // namespace onnxruntime