// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __APPLE__
namespace onnxruntime {
namespace logging {
/// <summary>
/// A NSLog based ISink
/// </summary>
/// <seealso cref="ISink" />
class NSLogSink : public ISink {
 public:
  NSLogSink() = default;
  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override;
};
}  // namespace logging
}  // namespace onnxruntime
#endif
