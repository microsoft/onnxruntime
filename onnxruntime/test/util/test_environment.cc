// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/test_environment.h"

#include <iostream>
#include <memory>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"
#include "core/session/ort_env.h"
#include "google/protobuf/stubs/common.h"
#include "gtest/gtest.h"
#include "onnxruntime_cxx_api.h"

using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

static std::unique_ptr<::onnxruntime::logging::LoggingManager> s_default_logging_manager;

const ::onnxruntime::Environment& GetEnvironment() {
  return ((OrtEnv*)*ort_env.get())->GetEnvironment();
}

::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
  return *((OrtEnv*)*ort_env.get())->GetEnvironment().GetLoggingManager();
}

}  // namespace test
}  // namespace onnxruntime
