// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/sinks/ostream_sink.h"
#include <sstream>

#ifdef _WIN32
#include <Windows.h>
#include <io.h>
#include <stdio.h>
#endif

namespace onnxruntime {
namespace logging {

#ifndef ORT_MINIMAL_BUILD
struct Color {
  constexpr static const char* kWarn = "\033[0;93m";      // yellow
  constexpr static const char* kError = "\033[1;31m";     // bold red
  constexpr static const char* kFatal = "\033[1;37;41m";  // bold white on red background
  constexpr static const char* kEnd = "\033[m";
};
#endif

#ifdef _WIN32
// Writes a UTF-8 string to the Windows console using WriteConsoleW.
static void WriteToConsole(const std::string& utf8_string, FILE* stream) {
  // Convert the UTF-8 string to UTF-16
  int utf16_length = MultiByteToWideChar(CP_UTF8, 0, utf8_string.c_str(), -1, NULL, 0);
  if (utf16_length == 0) {
    // fallback to fprintf on failure
    fprintf(stream, "%s", utf8_string.c_str());
    return;
  }

  std::wstring utf16_string(utf16_length, L'\0');
  if (MultiByteToWideChar(CP_UTF8, 0, utf8_string.c_str(), -1, &utf16_string[0], utf16_length) == 0) {
    // fallback to fprintf on failure
    fprintf(stream, "%s", utf8_string.c_str());
    return;
  }

  // Get the handle to the appropriate console buffer (stdout or stderr)
  HANDLE hConsole = (stream == stderr) ? GetStdHandle(STD_ERROR_HANDLE) : GetStdHandle(STD_OUTPUT_HANDLE);

  // Write the UTF-16 string to the console
  // We subtract 1 from the length to exclude the null terminator.
  DWORD chars_written = 0;
  if (!WriteConsoleW(hConsole, utf16_string.c_str(), static_cast<DWORD>(utf16_string.length() - 1), &chars_written, NULL)) {
    // fallback to fprintf on failure
    fprintf(stream, "%s", utf8_string.c_str());
  }
}
#endif

void OStreamSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  std::ostringstream msg_stream;

  // Format the timestamp.
  timestamp_ns::operator<<(msg_stream, timestamp);

  msg_stream << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
             << message.Location().ToString() << "] ";

#ifndef ORT_MINIMAL_BUILD
  const char* color_begin = "";
  const char* color_end = "";
  if (message.Severity() == Severity::kWARNING) {
    color_begin = Color::kWarn;
    color_end = Color::kEnd;
  } else if (message.Severity() == Severity::kERROR) {
    color_begin = Color::kError;
    color_end = Color::kEnd;
  } else if (message.Severity() == Severity::kFATAL) {
    color_begin = Color::kFatal;
    color_end = Color::kEnd;
  }
  msg_stream << color_begin << message.Message() << color_end;
#else
  msg_stream << message.Message();
#endif

  msg_stream << "\n";

  std::string message_to_log = msg_stream.str();

#ifdef _WIN32
  // On Windows, if we are writing to a console, use WriteConsoleW for proper Unicode support.
  // _isatty returns non-zero if the file descriptor is a TTY.
  if ((stream_ == stdout || stream_ == stderr) && _isatty(_fileno(stream_))) {
    WriteToConsole(message_to_log, stream_);
  } else {
    // If not a console (e.g., redirected to a file), write the original UTF-8 string.
    fprintf(stream_, "%s", message_to_log.c_str());
  }
#else
  // On other platforms, print the UTF-8 string directly.
  fprintf(stream_, "%s", message_to_log.c_str());
#endif

  if (flush_) {
    fflush(stream_);
  }
}

}  // namespace logging
}  // namespace onnxruntime