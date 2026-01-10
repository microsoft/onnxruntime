// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"
#include "core/common/logging/isink.h"

#include "date/date.h"

namespace onnxruntime {
namespace test {

using namespace ::onnxruntime::logging;

class CapturingSink : public logging::ISink {
 public:
  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override {
    // operator for formatting of timestamp in ISO8601 format including microseconds
    using date::operator<<;
    std::ostringstream msg;

    msg << timestamp << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
        << message.Location().ToString() << "] " << message.Message();

    messages_.push_back(msg.str());
  }

  const std::vector<std::string>& Messages() const {
    return messages_;
  }

 private:
  std::vector<std::string> messages_;
};
}  // namespace test
}  // namespace onnxruntime
