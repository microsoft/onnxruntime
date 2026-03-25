// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/migraphx/migraphx_inc.h"
#include "core/common/common.h"

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERROR_TYPE, bool THROW>
std::conditional_t<THROW, void, Status> RocmCall(
    ERROR_TYPE retCode, std::string_view exprString, std::string_view libName, ERROR_TYPE successCode, std::string_view msg, std::string_view file, int line);

}  // namespace onnxruntime

#define HIP_CALL(expr) (::onnxruntime::RocmCall<hipError_t, false>((expr), #expr, "HIP", hipSuccess, "", __FILE__, __LINE__))
#define HIP_CALL_THROW(expr) (::onnxruntime::RocmCall<hipError_t, true>((expr), #expr, "HIP", hipSuccess, "", __FILE__, __LINE__))
#define HIP_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(HIP_CALL(expr))
