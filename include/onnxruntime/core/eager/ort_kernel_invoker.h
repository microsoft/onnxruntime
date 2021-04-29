// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/session/environment.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {
#ifdef __GNUC__
#pragma GCC diagnostic push
#endif

class ORTInvoker {
 public:
  ORTInvoker(std::unique_ptr<IExecutionProvider> execution_provider, const logging::Logger& logger) : 
      execution_provider_(std::move(execution_provider)), logger_(logger) {
    if (!execution_provider_) {
    ORT_THROW("Execution provider is nullptr");
    }
  }

  IExecutionProvider& GetCurrentExecutionProvider() {
    return *execution_provider_;
  }

  common::Status Invoke(const std::string& op_name,
                        //optional inputs / outputs?
                        const std::vector<OrtValue>& inputs,
                        std::vector<OrtValue>& outputs,
                        const NodeAttributes* attributes,
                        const std::string domain = kOnnxDomain,
                        const int version = -1);

 private:
  std::unique_ptr<IExecutionProvider> execution_provider_;
  const logging::Logger& logger_;
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime