// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/common/exceptions.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// Convert onnxruntime::common::Status to OrtStatus*.
_Ret_notnull_ OrtStatus* ToOrtStatus(const onnxruntime::common::Status& st);

// Convert OrtStatus* to onnxruntime::common::Status.
Status ToStatus(const OrtStatus* ort_status, common::StatusCategory category = common::StatusCategory::ONNXRUNTIME);
};

#ifndef ORT_NO_EXCEPTIONS
#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                \
  }                                                                 \
  catch (const onnxruntime::NotImplementedException& ex) {          \
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, ex.what());   \
  }                                                                 \
  catch (const std::exception& ex) {                                \
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }                                                                 \
  catch (...) {                                                     \
    return OrtApis::CreateStatus(ORT_FAIL, "Unknown Exception");    \
  }

#else
#define API_IMPL_BEGIN {
#define API_IMPL_END }
#endif

// Return the OrtStatus if it indicates an error
#define ORT_API_RETURN_IF_ERROR(expr) \
  do {                                \
    auto _status = (expr);            \
    if (_status)                      \
      return _status;                 \
  } while (0)

// Convert internal onnxruntime::Status to OrtStatus and return if there's an error
#define ORT_API_RETURN_IF_STATUS_NOT_OK(expr)   \
  do {                                          \
    auto _status = (expr);                      \
    if (!_status.IsOK())                        \
      return onnxruntime::ToOrtStatus(_status); \
  } while (0)
