// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cann_inc.h"
#include "core/common/status.h"

namespace onnxruntime {

template <typename ERRTYPE, bool THRW>
bool CannCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");

#define CANN_CALL(expr) (CannCall<aclError, false>((expr), #expr, "CANN", ACL_SUCCESS))
#define CANN_CALL_THROW(expr) (CannCall<aclError, true>((expr), #expr, "CANN", ACL_SUCCESS))

}  // namespace onnxruntime
