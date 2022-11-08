// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/util.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using onnxruntime::rocm::CeilDiv;
using onnxruntime::rocm::aligned_vector;

namespace onnxruntime {

template <typename T, int BlockSize>
std::string GenerateTritonKernelName() {
  return "add_kernel";
}

std::string GenerateTritonCodeFile() {
  return "add_kernel.hsaco";
}

template <typename T, int BlockSize>
Status LaunchTritonVectorAdd(hipStream_t stream, const T* x, const T* y, T* z, int n) {
  hipInit(0);
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, GenerateTritonCodeFile()));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, GenerateTritonKernelName()));

  struct {
      void* _Ad;
      void* _Bd;
      void* _Cd;
      int   _nz;
  } args = {x,y,z,n};
  size_t size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(Function, CeilDiv(n, 1024), 1, 1,
                                  BlockSize, 1, 1, 0, 0, NULL, (void**)&config));

  auto status = hipGetLastError();
  ORT_RETURN_IF(status != hipSuccess, hipGetErrorName(status));
  return Status::OK();
}

}  // namespace onnxruntime
