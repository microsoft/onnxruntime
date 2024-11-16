// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#if !defined(__CUDACC__) && !defined(USE_CUDA_MINIMAL)
#include <cudnn_frontend.h>
#endif
namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

#ifndef USE_CUDA_MINIMAL
#define CUDNN_FE_CALL(expr) (CudaCall<cudnn_frontend::error_t, false,                                                   \
                                      cudnn_frontend::error_code_t>((cudnn_frontend::error_t)(expr), #expr, "CUDNN_FE", \
                                                                    cudnn_frontend::error_code_t::OK, "", __FILE__, __LINE__))
#define CUDNN_FE_CALL_THROW(expr) (CudaCall<cudnn_frontend::error_t, true,                                                    \
                                            cudnn_frontend::error_code_t>((cudnn_frontend::error_t)(expr), #expr, "CUDNN_FE", \
                                                                          cudnn_frontend::error_code_t::OK, "", __FILE__, __LINE__))
#endif
}  // namespace onnxruntime
