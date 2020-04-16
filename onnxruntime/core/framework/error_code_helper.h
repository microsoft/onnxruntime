// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/common/exceptions.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
_Ret_notnull_ OrtStatus* ToOrtStatus(const onnxruntime::common::Status& st);
};

#define API_IMPL_BEGIN try {
#define API_IMPL_END                                                \
  }                                                                 \
  catch (const onnxruntime::NotImplementedException& ex) {          \
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, ex.what());   \
  }                                                                 \
  catch (const std::exception& ex) {                                \
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what()); \
  }
