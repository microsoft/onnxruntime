// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/migraphx/migraphx_inc.h"
#include "core/common/common.h"

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
std::conditional_t<THRW, void, Status> RocmCall(
    ERRTYPE retCode, std::string_view exprString, std::string_view libName, ERRTYPE successCode, std::string_view msg, std::string_view file, int line);

#define HIP_CALL(expr) (RocmCall<hipError_t, false>((expr), #expr, "HIP", hipSuccess, "", __FILE__, __LINE__))
#define HIP_CALL_THROW(expr) (RocmCall<hipError_t, true>((expr), #expr, "HIP", hipSuccess, "", __FILE__, __LINE__))
#define HIP_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIP_CALL(expr))

}  // namespace onnxruntime
