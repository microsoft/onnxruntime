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
static std::unique_ptr<::onnxruntime::logging::Logger> s_default_logger;

::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
  ORT_ENFORCE(s_default_logging_manager != nullptr,
              "Need a TestEnvironment instance to provide the default logging manager.");

  return *s_default_logging_manager;
}

TestEnvironment::TestEnvironment(int argc, char** argv, bool create_default_logging_manager) {
  ORT_ENFORCE(s_default_logging_manager == nullptr,
              "Only expected one instance of TestEnvironment to be created.");

  std::clog << "Initializing unit testing." << std::endl;
  testing::InitGoogleTest(&argc, argv);

  if (create_default_logging_manager) {
    static std::string default_logger_id{"Default"};
    s_default_logging_manager = std::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                            Severity::kWARNING,  // TODO make this configurable through
                                                                                 // cmd line arguments or some other way
                                                            false);
    // make sure default logging manager exists and is working
    s_default_logger = ::onnxruntime::test::DefaultLoggingManager().CreateLogger(default_logger_id);
    LOGS(*s_default_logger, VERBOSE) << "Logging manager initialized.";
  }

#ifdef HAVE_FRAMEWORK_LIB
  runtime_environment_ = std::make_unique<Environment>(std::move(logging_manager));
  Status status = runtime_environment_->Initialize(name);
  ORT_ENFORCE(status == Status::OK(), "Failed creating runtime environment. ", status.ErrorMessage());
#endif
}

TestEnvironment::~TestEnvironment() {
  s_default_logger = nullptr;
  s_default_logging_manager = nullptr;

 #ifdef HAVE_FRAMEWORK_LIB
  // release environment followed by logging manager so any log output from runtime environment shutdown
  // using the default logger will succeed.
  runtime_environment_ = nullptr;
#else
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
}

}  // namespace test
}  // namespace onnxruntime
