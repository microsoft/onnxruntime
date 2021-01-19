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

namespace onnxruntime {
#ifdef __GNUC__
#pragma GCC diagnostic push
#endif

class ORTInvoker {
 public:
  ORTInvoker(std::unique_ptr<IExecutionProvider> execution_provider);

  IExecutionProvider& GetCurrentExecutionProvider();

  common::Status Invoke(const std::string& op_name,
                        //optional inputs / outputs?
                        const std::vector<OrtValue>& inputs,
                        std::vector<OrtValue>& outputs,
                        const NodeAttributes* attributes,
                        const std::string domain = kOnnxDomain,
                        const int version = -1);

 private:

  std::unique_ptr<IExecutionProvider> execution_provider_;
  std::unique_ptr<logging::LoggingManager> logging_manager_;
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime
