// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once


//#include "hip/hip_runtime.h"
//#include "hip/hip_ext.h"
#include "hip/hip_runtime_api.h"

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/util.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include <iostream>
#include <unordered_map>

using onnxruntime::rocm::CeilDiv;
using onnxruntime::rocm::aligned_vector;

#define HIP_CHECK(status)                                       \
	if (status != hipSuccess) {                             \
		ORT_RETURN_IF(true, hipGetErrorName(status));  \
	}


namespace onnxruntime {

template <typename T, int BlockSize>
const char* GenerateTritonKernelName() {
  return "add_kernel";
}

const std::string GenerateTritonCodeFile() {
  std::string lib_path = "/workspace/onnxruntime/onnxruntime/python/tools/kernel_explorer/kernels/codegen/tmp";
  return lib_path + "/add_kernel.hsaco";
}

static std::unordered_map<std::string, hipFunction_t> tritonKernelMap;

hipFunction_t getTritonHipFunctionByName(std::string fName) {
	return tritonKernelMap[fName];
}

template<typename T, int BlockSize>
Status initTritonKernels() {
  hipInit(0);
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  hipModule_t Module;
  hipFunction_t Function;
  const std::string code_fullpath = GenerateTritonCodeFile();
  HIP_CHECK(hipModuleLoad(&Module, code_fullpath.c_str()));
  std::string fname =  GenerateTritonKernelName<T,BlockSize>();
  HIP_CHECK(hipModuleGetFunction(&Function, Module, fname.c_str()));
  tritonKernelMap[fname] = Function;
  return Status::OK();
}

template <typename T, int BlockSize>
Status LaunchTritonVectorAdd(hipStream_t stream, const T* x, const T* y, T* z, int n) {
/*
  hipInit(0);
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  hipModule_t Module;
  hipFunction_t Function;
  const std::string code_fullpath = GenerateTritonCodeFile();
  HIP_CHECK(hipModuleLoad(&Module, code_fullpath.c_str()));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, GenerateTritonKernelName<T,BlockSize>()));
*/
  hipFunction_t Function = getTritonHipFunctionByName(GenerateTritonKernelName<T,BlockSize>());
  struct {
      void* _Xd;
      void* _Yd;
      void* _Zd;
      int   _Nz;
  } args = {(void*)x, (void*)y, (void*)z,n};

  size_t size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(Function, CeilDiv(n, 1024), 1, 1,
                                  BlockSize, 1, 1, 0, 0, NULL, (void**)&config));

  return Status::OK();
}

}  // namespace onnxruntime
