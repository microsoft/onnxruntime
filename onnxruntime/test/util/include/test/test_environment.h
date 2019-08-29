// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"
#include "core/session/environment.h"

namespace onnxruntime {
namespace test {

/**
Static logging manager with a CLog based sink so logging macros that use the default 
logger will work. Instance is created and owned by TestEnvironment. The sharing via 
this static is for convenience.
*/
::onnxruntime::logging::LoggingManager& DefaultLoggingManager();

/**
Perform default initialization of a unit test executable.
This includes setting up google test, the default logger, and the framework runtime environment.
Keep the instance of this class until tests have completed.
*/
class TestEnvironment {
 public:
  TestEnvironment(int argc, char** argv);
  ~TestEnvironment();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestEnvironment);

  std::unique_ptr<Environment> runtime_environment_;
};

}  // namespace test
}  // namespace onnxruntime
