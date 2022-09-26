// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "core/framework/execution_provider.h"

namespace onnxruntime {

class OpWrapperExecutionProvider : public IExecutionProvider {
 public:
  OpWrapperExecutionProvider(const ProviderOptions& provider_options);
  virtual ~OpWrapperExecutionProvider();

  ProviderOptions GetProviderOptions() const override { return provider_options_; }

 private:
  ProviderOptions provider_options_;
};
}  // namespace onnxruntime