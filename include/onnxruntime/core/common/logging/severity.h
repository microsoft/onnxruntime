// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>

namespace onnxruntime {
namespace logging {
// mild violation of naming convention. the 'k' lets us use token concatenation in the macro
// ::onnxruntime::Logging::Severity::k##severity. It's not legal to have ::onnxruntime::Logging::Severity::##severity
// the uppercase makes the LOG macro usage look as expected for passing an enum value as it will be LOGS(logger, ERROR)
enum class Severity {
  kVERBOSE = 0,
  kINFO = 1,
  kWARNING = 2,
  kERROR = 3,
  kFATAL = 4
};

constexpr const char* SEVERITY_PREFIX = "VIWEF";

/**
 * Parses a string into a Severity value.
 * Accepts: "VERBOSE" (or "0"), "INFO" (or "1"), "WARNING" (or "2"), "ERROR" (or "3"), "FATAL" (or "4").
 * @param str The string to parse (case-sensitive).
 * @param[out] severity The parsed severity value.
 * @return true if parsing succeeded, false if the string was not recognized.
 */
inline bool SeverityFromString(const char* str, Severity& severity) {
  if (str == nullptr) return false;

  if (std::strcmp(str, "VERBOSE") == 0 || std::strcmp(str, "0") == 0) {
    severity = Severity::kVERBOSE;
  } else if (std::strcmp(str, "INFO") == 0 || std::strcmp(str, "1") == 0) {
    severity = Severity::kINFO;
  } else if (std::strcmp(str, "WARNING") == 0 || std::strcmp(str, "2") == 0) {
    severity = Severity::kWARNING;
  } else if (std::strcmp(str, "ERROR") == 0 || std::strcmp(str, "3") == 0) {
    severity = Severity::kERROR;
  } else if (std::strcmp(str, "FATAL") == 0 || std::strcmp(str, "4") == 0) {
    severity = Severity::kFATAL;
  } else {
    return false;
  }
  return true;
}

}  // namespace logging
}  // namespace onnxruntime
