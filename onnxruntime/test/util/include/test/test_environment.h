// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"

#ifdef HAVE_FRAMEWORK_LIB
#include "core/framework/environment.h"
#endif

namespace onnxruntime {
namespace test {

/**
Static logging manager with a CLog based sink so logging macros that use the default logger will work
Instance is created and owned by TestEnvironment. The sharing via this static is for convenience.
*/
::onnxruntime::logging::LoggingManager& DefaultLoggingManager();

/**
Perform default initialization of a unit test executable.
This includes setting up google test, the default logger, and the framework runtime environment.
Keep the instance of this class until tests have completed.
*/
class TestEnvironment {
 public:
  TestEnvironment(int argc, char** argv, bool create_default_logging_manager = true);
  ~TestEnvironment();

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestEnvironment);

#ifdef HAVE_FRAMEWORK_LIB
  std::unique_ptr<Environment> runtime_environment_;
#endif
};

}  // namespace test
}  // namespace onnxruntime
