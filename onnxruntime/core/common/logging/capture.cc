// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/capture.h"
#include "core/common/logging/logging.h"
#include "gsl/span"
#include "gsl/gsl_util"

namespace onnxruntime {
namespace logging {

void Capture::CapturePrintf(msvc_printf_check const char* format, ...) {
  va_list arglist;
  va_start(arglist, format);

  ProcessPrintf(format, arglist);

  va_end(arglist);
}

// from https://github.com/KjellKod/g3log/blob/master/src/logcapture.cpp LogCapture::capturef
// License: https://github.com/KjellKod/g3log/blob/master/LICENSE
// Modifications Copyright (c) Microsoft.
void Capture::ProcessPrintf(msvc_printf_check const char* format, va_list args) {
  static constexpr auto kTruncatedWarningText = "[...truncated...]";
  static const int kMaxMessageSize = 2048;
  char message_buffer[kMaxMessageSize];
  const auto message = gsl::make_span(message_buffer);

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) && !defined(__GNUC__))
  const int nbrcharacters = vsnprintf_s(message.data(), message.size(), _TRUNCATE, format, args);
#else
  const int nbrcharacters = vsnprintf(message.data(), message.size(), format, args);
#endif

  if (nbrcharacters <= 0) {
    stream_ << "\n\tERROR LOG MSG NOTIFICATION: Failure to successfully parse the message";
    stream_ << '"' << format << '"' << std::endl;
  } else if (nbrcharacters > message.size()) {
    stream_ << message.data() << kTruncatedWarningText;
  } else {
    stream_ << message.data();
  }
}

Capture::~Capture() {
  if (logger_ != nullptr) {
    logger_->Log(*this);
  }
}
}  // namespace logging
}  // namespace onnxruntime
