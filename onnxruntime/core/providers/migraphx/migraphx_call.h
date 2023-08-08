// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "migraphx_inc.h"

#pragma once

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, Status> RocmCall(
    ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg, const char* file, const int line);

#define HIP_CALL(expr) (RocmCall<hipError_t, false>((expr), #expr, "HIP", hipSuccess, "", __FILE__, __LINE__))
#define HIP_CALL_THROW(expr) (RocmCall<hipError_t, true>((expr), #expr, "HIP", hipSuccess, "", __FILE__, __LINE__))

}  // namespace onnxruntime
