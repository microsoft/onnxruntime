// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/logging/make_platform_default_log_sink.h"

#if defined(__ANDROID__)
#include "core/platform/android/logging/android_log_sink.h"
#elif defined(__APPLE__)
#include "core/platform/apple/logging/apple_log_sink.h"
#else
#include "core/common/logging/sinks/clog_sink.h"
#endif

namespace onnxruntime {
namespace logging {

std::unique_ptr<ISink> MakePlatformDefaultLogSink() {
#if defined(__ANDROID__)
  return std::make_unique<AndroidLogSink>();
#elif defined(__APPLE__)
  return std::make_unique<AppleLogSink>();
#else
  return std::make_unique<CLogSink>();
#endif
}

}  // namespace logging
}  // namespace onnxruntime
