// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __ANDROID__
namespace onnxruntime {
namespace logging {
/// <summary>
/// A __android_log_print based ISink
/// </summary>
/// <seealso cref="ISink" />
class AndroidLogSink : public ISink {
 public:
  AndroidLogSink() = default;
  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override;
};
}  // namespace logging
}  // namespace onnxruntime
#endif
