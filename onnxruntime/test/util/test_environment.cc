// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/test_environment.h"

#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "test/random_seed.h"

using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

static std::unique_ptr<::onnxruntime::logging::LoggingManager> s_default_logging_manager;

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
<<<<<<< HEAD
    s_default_logging_manager = std::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                                 Severity::kWARNING,  // TODO make this configurable through
                                                                                      // cmd line arguments or some other way
                                                                 false,
                                                                 LoggingManager::InstanceType::Default,
                                                                 &default_logger_id);
=======
    s_default_logging_manager = onnxruntime::make_unique<LoggingManager>(std::unique_ptr<ISink>{new CLogSink{}},
                                                        Severity::kWARNING,  // TODO make this configurable through
                                                                             // cmd line arguments or some other way
                                                        false,
                                                        LoggingManager::InstanceType::Default,
                                                        &default_logger_id);
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

    // make sure default logging manager exists and is working
    auto logger = DefaultLoggingManager().DefaultLogger();
    LOGS(logger, VERBOSE) << "Logging manager initialized.";
  }

  uint32_t seed = 0;
  {
    // Set a specific random seed value here, for reproducing test failure issues.
    // TODO Make this configurable from cmd line argument.
    // seed = 2826461700;
  }
  if (seed != 0) {
    SetStaticRandomSeed(seed); // set the random seed value for this test run.
  }
  std::clog << "ORT test random seed value: " << GetStaticRandomSeed() << std::endl;

#ifdef HAVE_FRAMEWORK_LIB
  auto status = Environment::Create(runtime_environment_);
  ORT_ENFORCE(status == Status::OK(), "Failed creating runtime environment. ", status.ErrorMessage());
#endif
}

TestEnvironment::~TestEnvironment() {
#ifdef HAVE_FRAMEWORK_LIB
  // release environment followed by logging manager so any log output from runtime environment shutdown
  // using the default logger will succeed.
  runtime_environment_ = nullptr;
#else
  ::google::protobuf::ShutdownProtobufLibrary();
#endif

  // dispose logging manager manually to make sure it's destructed before the default logging mutex
  s_default_logging_manager = nullptr;
}

}  // namespace test
}  // namespace onnxruntime
