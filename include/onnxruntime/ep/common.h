// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include <stdexcept>

#include "api.h"

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

// see ORT_ENFORCE for implementations that also capture a stack trace and work in builds with exceptions disabled
// NOTE: In this simplistic implementation you must provide an argument, even it if's an empty string
#define EP_ENFORCE(condition, ...)                       \
  do {                                                   \
    if (!(condition)) {                                  \
      std::ostringstream oss;                            \
      oss << "EP_ENFORCE failed: " << #condition << " "; \
      oss << __VA_ARGS__;                                \
      throw std::runtime_error(oss.str());               \
    }                                                    \
  } while (false)

// Ignores an OrtStatus* while taking ownership of it so that it does not get leaked.
#define IGNORE_ORTSTATUS(status_expr)   \
  do {                                  \
    OrtStatus* _status = (status_expr); \
    Ort::Status _ignored{_status};      \
  } while (false)

// Helper macros to convert exceptions to OrtStatus* return values.
// Usage:
//   EXCEPTION_TO_RETURNED_STATUS_BEGIN
//     ... code that may throw ...
//   EXCEPTION_TO_RETURNED_STATUS_END
#define EXCEPTION_TO_RETURNED_STATUS_BEGIN try {
#define EXCEPTION_TO_RETURNED_STATUS_END                  \
  }                                                       \
  catch (const Ort::Exception& ex) {                      \
    Ort::Status status(ex);                               \
    return status.release();                              \
  }                                                       \
  catch (const std::exception& ex) {                      \
    Ort::Status status(ex.what(), ORT_EP_FAIL);           \
    return status.release();                              \
  }                                                       \
  catch (...) {                                           \
    Ort::Status status("Unknown exception", ORT_EP_FAIL); \
    return status.release();                              \
  }
