// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/sinks/ostream_sink.h"
#include "date/date.h"

namespace onnxruntime {
namespace logging {

void OStreamSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  // operator for formatting of timestamp in ISO8601 format including microseconds
  using date::operator<<;

  // Two options as there may be multiple calls attempting to write to the same sink at once:
  // 1) Use mutex to synchronize access to the stream.
  // 2) Create the message in an ostringstream and output in one call.
  //
  // Going with #2 as it should scale better at the cost of creating the message in memory first
  // before sending to the stream.

  std::ostringstream msg;

  msg << timestamp << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
      << message.Location().ToString() << "] " << message.Message() << "\n";

  (*stream_) << msg.str();

  if (flush_) {
    stream_->flush();
  }
}
#ifdef _WIN32
void WOStreamSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  // operator for formatting of timestamp in ISO8601 format including microseconds
  using date::operator<<;

  // Two options as there may be multiple calls attempting to write to the same sink at once:
  // 1) Use mutex to synchronize access to the stream.
  // 2) Create the message in an ostringstream and output in one call.
  //
  // Going with #2 as it should scale better at the cost of creating the message in memory first
  // before sending to the stream.

  std::wostringstream msg;

  msg << timestamp << L" [" << message.SeverityPrefix() << L":" << message.Category() << L":" << ToWideString(logger_id) << L", "
      << ToWideString(message.Location().ToString()) << L"] " << ToWideString(message.Message()) << L"\n";

  (*stream_) << msg.str();

  if (flush_) {
    stream_->flush();
  }
}
#endif
}  // namespace logging
}  // namespace onnxruntime
