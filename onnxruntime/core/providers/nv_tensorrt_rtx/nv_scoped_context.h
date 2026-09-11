// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include "nv_includes.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
struct ScopedContext {
  explicit ScopedContext(int device_id) : pushed_(true) {
    CUcontext cu_context = 0;
    CU_CALL_THROW(cuCtxGetCurrent(&cu_context));
    if (!cu_context) {
      // cuCtxGetCurrent succeeded but returned nullptr, which indicates that no CUDA context
      // is currently set for this thread. This implicates that there is not user created context.
      // We use runtime API to initialize a context for the specified device.
      CUDA_CALL_THROW(cudaSetDevice(device_id));
      CU_CALL_THROW(cuCtxGetCurrent(&cu_context));
    }
    CU_CALL_THROW(cuCtxPushCurrent(cu_context));
  }

  /** \brief Push an existing context (e.g. CIG context); pop on destruction. */
  explicit ScopedContext(CUcontext ctx) : pushed_(ctx != nullptr) {
    if (ctx != nullptr) {
      CU_CALL_THROW(cuCtxPushCurrent(ctx));
    }
  }

  ScopedContext(const ScopedContext&) = delete;

  ~ScopedContext() {
    if (pushed_) {
      cuCtxPopCurrent(nullptr);
    }
  }

 private:
  bool pushed_ = true;
};
}  // namespace onnxruntime
