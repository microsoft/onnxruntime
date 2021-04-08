// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/sinks/clog_sink.h"
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
  ORTInvoker(std::unique_ptr<IExecutionProvider> execution_provider);

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

  struct EnvironmentManager {
    EnvironmentManager() {
      std::string logger_id{"ORTInvoker"};
      auto logging_manager = onnxruntime::make_unique<logging::LoggingManager>(
        std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
        logging::Severity::kVERBOSE, false,
        logging::LoggingManager::InstanceType::Default,
        &logger_id);
      Environment::Create(std::move(logging_manager), ort_env_);
    }

    std::unique_ptr<Environment> ort_env_;
    friend class ORTInvoker;
  };

  static EnvironmentManager env_manager_;
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime