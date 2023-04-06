// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"
#include "hip/hip_runtime_api.h"

namespace onnxruntime {
namespace rocm {

struct TritonKernelMetaData {
  int num_warps;
  int shared_mem_size;
  int block_size;
  hipFunction_t func;
};

int NextPowerOf2(int size) {
  int pow = 0;
  while (size > 2) {
    size /= 2;
    pow++;
  }
  return pow + 1;
}

int GetTritonKernelBlockSize(std::string fname);

Status LaunchTritonKernel(hipStream_t stream, std::string fname, int grid0, int grid1, int grid2, void *args, size_t args_size);

Status LoadRocmTritonKernel();

}  // end of rocm
}  // end of onnxruntime
