// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {

struct CloudExecutionProviderInfo {
  std::string url;
  std::string access_token;
};

class CloudExecutionProvider : public IExecutionProvider {
 public:
  explicit CloudExecutionProvider(const CloudExecutionProviderInfo& info);
  virtual ~CloudExecutionProvider();
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  CloudExecutionProviderInfo info_;
};

}  // namespace onnxruntime