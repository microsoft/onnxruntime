// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {

class CloudExecutionProvider : public IExecutionProvider {
 public:
  explicit CloudExecutionProvider(const std::unordered_map<std::string, std::string>& config);
  ~CloudExecutionProvider();
  const std::unordered_map<std::string, std::string>& GetConfig() const { return config_; }

 private:
  std::unordered_map<std::string, std::string> config_;
};

}  // namespace onnxruntime