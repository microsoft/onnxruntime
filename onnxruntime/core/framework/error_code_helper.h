// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"

namespace onnxruntime {
OrtStatus* ToOrtStatus(const onnxruntime::common::Status& st);
};

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                          \
  }                                                           \
  catch (const onnxruntime::NotImplementedException& ex) {    \
    return OrtCreateStatus(ORT_NOT_IMPLEMENTED, ex.what());   \
  }                                                           \
  catch (const std::exception& ex) {                          \
    return OrtCreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }
