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
  hipFunction_t func;
  std::unordered_map<std::string, int> constants;
};

namespace {

template <typename T>
struct DataTypeToName;

#define DTYPE_TO_STR(type, name) \
    template<> struct DataTypeToName<type>{ constexpr static const char* value = name; }; \

DTYPE_TO_STR(float, "fp32");
DTYPE_TO_STR(half, "fp16");
DTYPE_TO_STR(double, "fp64");
DTYPE_TO_STR(BFloat16, "bf16");

}  // end of namespace

template <typename T>
const std::string GetDataTypeName() {
  return DataTypeToName<T>::value;
}

Status LaunchTritonKernel(hipStream_t stream, std::string fname, int grid0, int grid1, int grid2, void *args, size_t args_size);

void LoadRocmTritonKernel();

}  // end of rocm
}  // end of onnxruntime
