// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "api.h"
#include "common.h"

// This header is only used when building WebGPU/CUDA EP as a shared library.
//
// This header file is used as a precompiled header so it is always included first.

#pragma push_macro("ORT_EP_API_ADAPTER_HEADER_INCLUDED")
#undef ORT_EP_API_ADAPTER_HEADER_INCLUDED
#define ORT_EP_API_ADAPTER_HEADER_INCLUDED

#include "adapter/allocator.h"
#include "adapter/logging.h"
#include "adapter/ep.h"
#include "adapter/kernel_registry.h"

#pragma pop_macro("ORT_EP_API_ADAPTER_HEADER_INCLUDED")

//
// EP specific using declarations
//

#define EP_SPECIFIC_USING_DECLARATIONS                                              \
  using FuncManager = onnxruntime::ep::adapter::FuncManager;                        \
  using KernelCreatePtrFn = onnxruntime::ep::adapter::KernelCreatePtrFn;            \
  using KernelDefBuilder = onnxruntime::ep::adapter::KernelDefBuilder;              \
  using KernelRegistry = onnxruntime::ep::adapter::KernelRegistry;                  \
  using KernelCreateInfo = onnxruntime::ep::adapter::KernelCreateInfo;              \
  using BuildKernelCreateInfoFn = onnxruntime::ep::adapter::KernelCreateInfo (*)(); \
  using OpKernelInfo = onnxruntime::ep::adapter::OpKernelInfo;                      \
  using OpKernelContext = onnxruntime::ep::adapter::OpKernelContext;                \
  using OpKernel = onnxruntime::ep::adapter::OpKernel;                              \
  using DataTransferManager = onnxruntime::ep::adapter::DataTransferManager;        \
  namespace logging {                                                               \
  using Logger = onnxruntime::ep::adapter::Logger;                                  \
  }

namespace onnxruntime {
namespace webgpu {
EP_SPECIFIC_USING_DECLARATIONS
}  // namespace webgpu
namespace cuda {
EP_SPECIFIC_USING_DECLARATIONS
}  // namespace cuda

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
namespace webgpu {
EP_SPECIFIC_USING_DECLARATIONS
}  // namespace webgpu
namespace cuda {
EP_SPECIFIC_USING_DECLARATIONS
}  // namespace cuda
}  // namespace contrib
#endif

}  // namespace onnxruntime

#undef EP_SPECIFIC_USING_DECLARATIONS
