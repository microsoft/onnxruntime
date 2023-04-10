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

namespace {

template <typename T>
struct DataTypeToName;

#define DTYPE_TO_STR(type, name) \
    template<> struct DataTypeToName<type>{ constexpr static const char* value = name; }; \

DTYPE_TO_STR(float, "float");
DTYPE_TO_STR(half, "half");
DTYPE_TO_STR(double, "double");
DTYPE_TO_STR(BFloat16, "BFloat16");

}  // end of namespace

template <typename T>
const std::string GetDataTypeName() {
  return DataTypeToName<T>::value;
}

int NextPowerOf2(int size);

Status LaunchTritonKernel(hipStream_t stream, std::string fname, int grid0, int grid1, int grid2, void *args, size_t args_size);

Status LoadRocmTritonKernel();

}  // end of rocm
}  // end of onnxruntime
