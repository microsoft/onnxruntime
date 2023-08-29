// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/sinks/ostream_sink.h"
#include "date/date.h"

namespace onnxruntime {
namespace logging {

#ifndef ORT_MINIMAL_BUILD
struct Color {
  constexpr static const char* kWarn = "\033[0;93m";      // yellow
  constexpr static const char* kError = "\033[1;31m";     // bold red
  constexpr static const char* kFatal = "\033[1;37;41m";  // bold white on red background
  constexpr static const char* kEnd = "\033[m";
#ifdef _WIN32
  constexpr static const wchar_t* kLWarn = L"\033[0;93m";      // yellow
  constexpr static const wchar_t* kLError = L"\033[1;31m";     // bold red
  constexpr static const wchar_t* kLFatal = L"\033[1;37;41m";  // bold white on red background
  constexpr static const wchar_t* kLEnd = L"\033[m";
#endif
};
#endif

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

#ifndef ORT_MINIMAL_BUILD
  if (message.Severity() == Severity::kWARNING) {
    msg << Color::kWarn;
  } else if (message.Severity() == Severity::kERROR) {
    msg << Color::kError;
  } else if (message.Severity() == Severity::kFATAL) {
    msg << Color::kFatal;
  }
#endif

  msg << timestamp << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
      << message.Location().ToString() << "] " << message.Message();

#ifndef ORT_MINIMAL_BUILD
  if (message.Severity() == Severity::kWARNING ||
      message.Severity() == Severity::kERROR ||
      message.Severity() == Severity::kFATAL) {
    msg << Color::kEnd;
  }
#endif
  msg << "\n";

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

#ifndef ORT_MINIMAL_BUILD
  if (message.Severity() == Severity::kWARNING) {
    msg << Color::kLWarn;
  } else if (message.Severity() == Severity::kERROR) {
    msg << Color::kLError;
  } else if (message.Severity() == Severity::kFATAL) {
    msg << Color::kLFatal;
  }
#endif

  msg << timestamp << L" [" << message.SeverityPrefix() << L":" << message.Category() << L":" << ToWideString(logger_id) << L", "
      << ToWideString(message.Location().ToString()) << L"] " << ToWideString(message.Message());

#ifndef ORT_MINIMAL_BUILD
  if (message.Severity() == Severity::kWARNING ||
      message.Severity() == Severity::kERROR ||
      message.Severity() == Severity::kFATAL) {
    msg << Color::kLEnd;
  }
#endif
  msg << L"\n";

  (*stream_) << msg.str();

  if (flush_) {
    stream_->flush();
  }
}
#endif
}  // namespace logging
}  // namespace onnxruntime
