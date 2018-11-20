// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vstest_logger.h"
#include <CppUnitTest.h>
#include <ostream>
#include <sstream>
#include <string>

#include "date/date.h"

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"

void VsTestSink::SendImpl(const ::onnxruntime::logging::Timestamp& timestamp, const std::string& logger_id_, const ::onnxruntime::logging::Capture& message) {
  // operator for formatting of timestamp in ISO8601 format including microseconds
  using date::operator<<;

  // Two options as there may be multiple calls attempting to write to the same sink at once:
  // 1) Use mutex to synchronize access to the stream.
  // 2) Create the message in an ostringstream and output in one call.
  //
  // Going with #2 as it should scale better at the cost of creating the message in memory first
  // before sending to the stream.

  std::ostringstream msg;

  msg << timestamp << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id_ << ", "
      << message.Location().ToString() << "] " << message.Message();
  std::string s = msg.str();
  Microsoft::VisualStudio::CppUnitTestFramework::Logger::WriteMessage(s.c_str());
}
