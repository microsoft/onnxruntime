/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include "core/providers/cuda/cuda_common.h"

#define cuErrCheck(stat, wrap)                                                 \
  {                                                                            \
    onnxruntime::contrib::cuda::cuErrCheck_((stat), wrap, __FILE__, __LINE__); \
  }

namespace onnxruntime {
namespace contrib {
namespace cuda {

class CUDADriverWrapper {
 public:
  CUDADriverWrapper();

  ~CUDADriverWrapper();

  // Delete default copy constructor and copy assignment constructor
  CUDADriverWrapper(const CUDADriverWrapper&) = delete;
  CUDADriverWrapper& operator=(const CUDADriverWrapper&) = delete;

  CUresult cuGetErrorName(CUresult error, const char** pStr) const;

  CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const;

  CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const;

  CUresult cuModuleUnload(CUmodule hmod) const;

  CUresult cuLinkDestroy(CUlinkState state) const;

  CUresult cuModuleLoadData(CUmodule* module, const void* image) const;

  CUresult cuLinkCreate(uint32_t numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const;

  CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) const;

  CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, uint32_t numOptions,
                         CUjit_option* options, void** optionValues) const;

  CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name,
                         uint32_t numOptions, CUjit_option* options, void** optionValues) const;

  CUresult cuLaunchCooperativeKernel(
      CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
      uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
      void** kernelParams) const;

  CUresult cuLaunchKernel(
      CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ, uint32_t blockDimX,
      uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream, void** kernelParams,
      void** extra) const;

 private:
  void* handle;
  CUresult (*_cuGetErrorName)(CUresult, const char**);
  CUresult (*_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int);
  CUresult (*_cuLinkComplete)(CUlinkState, void**, size_t*);
  CUresult (*_cuModuleUnload)(CUmodule);
  CUresult (*_cuLinkDestroy)(CUlinkState);
  CUresult (*_cuLinkCreate)(unsigned int, CUjit_option*, void**, CUlinkState*);
  CUresult (*_cuModuleLoadData)(CUmodule*, const void*);
  CUresult (*_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
  CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, const char*, unsigned int, CUjit_option*, void**);
  CUresult (*_cuLinkAddData)(
      CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
  CUresult (*_cuLaunchCooperativeKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                         unsigned int, unsigned int, unsigned int, CUstream, void**);
  CUresult (*_cuLaunchKernel)(
      CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
      uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
      void** kernelParams, void** extra);
};

inline void cuErrCheck_(CUresult stat, const CUDADriverWrapper& wrap, const char* file, int line) {
  if (stat != CUDA_SUCCESS) {
    const char* msg = nullptr;
    wrap.cuGetErrorName(stat, &msg);
    fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
    ORT_THROW("CUDA Error", msg);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
