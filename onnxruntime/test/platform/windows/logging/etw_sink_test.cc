// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/windows/logging/etw_sink.h"

#ifdef ETW_TRACE_LOGGING_SUPPORTED

#include "core/common/logging/capture.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/composite_sink.h"

#include "test/common/logging/helpers.h"
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26400)
#endif
namespace onnxruntime {
namespace test {

using namespace ::onnxruntime::logging;

/// <summary>
/// Test usage of the ETW sinks does not fail.
/// </summary>
TEST(LoggingTests, TestEtwSink) {
  const std::string logid{"ETW"};
  const std::string message{"Test message"};

  // create scoped manager so sink gets destroyed once done and we check disposal
  // within the scope of this test
  {
    LoggingManager manager{std::unique_ptr<ISink>{new EtwSink{}}, Severity::kWARNING, false,
                           LoggingManager::InstanceType::Temporal};

    auto logger = manager.CreateLogger(logid);

    LOGS(*logger, WARNING) << message;

    // can't test much else without creating an interface for ETW, using that in EtwSink
    // and mocking that interface here. too much work given how trivial the logic in EtwSink is.
  }
}

/// <summary>
/// Test that attempting to create two ETW sinks is fine.
/// We register the ETW handler for the duration of the program so it can be shared
/// across multiple sinks.
/// </summary>
TEST(LoggingTests, TestEtwSinkCtor) {
  CompositeSink* sinks = new CompositeSink();
  sinks->AddSink(std::unique_ptr<ISink>(new EtwSink()))
      .AddSink(std::unique_ptr<ISink>(new EtwSink()));

  LoggingManager manager{std::unique_ptr<ISink>{sinks},
                         Severity::kWARNING,
                         false,
                         LoggingManager::InstanceType::Temporal};

  auto logger = manager.CreateLogger("logid");

  LOGS(*logger, WARNING) << "Two sinks aren't better than one";
}

}  // namespace test
}  // namespace onnxruntime
#endif  // ETW_TRACE_LOGGING_SUPPORTED
