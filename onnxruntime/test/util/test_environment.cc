// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/test_environment.h"

#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"

using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

static std::unique_ptr<::onnxruntime::logging::LoggingManager> s_default_logging_manager;

::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
  ORT_ENFORCE(s_default_logging_manager != nullptr,
              "Need a TestEnvironment instance to provide the default logging manager.");

  return *s_default_logging_manager;
}

TestEnvironment::TestEnvironment(int argc, char** argv) {
  std::clog << "Initializing unit testing." << std::endl;
  testing::InitGoogleTest(&argc, argv);

  std::string default_logger_id{"Default"};
  auto logging_manager = onnxruntime::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                                  Severity::kWARNING,  // TODO cmd-line configurable?
                                                                  false);
  runtime_environment_ = onnxruntime::make_unique<Environment>(std::move(logging_manager));
  Status status = runtime_environment_->Initialize(default_logger_id);
  ORT_ENFORCE(status == Status::OK(), "Failed creating runtime environment. ", status.ErrorMessage());
}

TestEnvironment::~TestEnvironment() {
}

}  // namespace test
}  // namespace onnxruntime
