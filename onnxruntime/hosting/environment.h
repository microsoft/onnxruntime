// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/environment.h"
#include "core/common/logging/logging.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
namespace hosting {

namespace logging = logging;

class HostingEnvironment {
 public:
  explicit HostingEnvironment(logging::Severity severity);
  ~HostingEnvironment() = default;
  HostingEnvironment(const HostingEnvironment&) = delete;

  const logging::Logger& GetAppLogger();
  std::unique_ptr<logging::Logger> GetLogger(const std::string& id);
  const logging::Severity GetLogSeverity() const;

  common::Status
  InitializeModel(std::string model_path);
  const std::vector<std::string> GetModelOutputNames() const;

  std::unique_ptr<onnxruntime::InferenceSession> session;

 private:
  const logging::Severity severity_;
  const std::string logger_id_;
  logging::LoggingManager default_logging_manager_;

  std::unique_ptr<onnxruntime::Environment> runtime_environment_;
  onnxruntime::SessionOptions options_;
  std::vector<std::string> model_output_names_;
};

}  // namespace hosting
}  // namespace onnxruntime
