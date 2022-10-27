// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {

//namespace cloud {
//class EndPointInvoker;
//}

class CloudExecutionProvider : public IExecutionProvider {
 public:
  explicit CloudExecutionProvider(const std::unordered_map<std::string, std::string>& config);
  ~CloudExecutionProvider();
  const std::unordered_map<std::string, std::string>& GetConfig() const { return config_; }
  //cloud::EndPointInvoker* GetInvoker() const { return endpoint_invoker_.get(); }

 private:
  std::unordered_map<std::string, std::string> config_;
  //std::unique_ptr<cloud::EndPointInvoker> endpoint_invoker_;
};

}  // namespace onnxruntime