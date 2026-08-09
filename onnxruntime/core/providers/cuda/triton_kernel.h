// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include <cuda.h>

namespace onnxruntime {
namespace cuda {

struct TritonKernelMetaData {
  int num_warps;
  int shared_mem_size;
  CUfunction func;
  std::unordered_map<std::string, int> constants;
  std::string name;
};

namespace {

template <typename T>
struct DataTypeToName;

#define DTYPE_TO_STR(type, name)               \
  template <>                                  \
  struct DataTypeToName<type> {                \
    constexpr static const char* value = name; \
  };

DTYPE_TO_STR(float, "fp32");
DTYPE_TO_STR(half, "fp16");
DTYPE_TO_STR(double, "fp64");
DTYPE_TO_STR(BFloat16, "bf16");

}  // end of namespace

template <typename T>
const std::string GetDataTypeName() {
  return DataTypeToName<T>::value;
}

void LoadOrtTritonKernel();

Status LaunchTritonKernel(cudaStream_t stream, std::string fname, int grid0, int grid1, int grid2, void* args, size_t args_size);

const TritonKernelMetaData* GetOrtTritonKernelMetadata(size_t idx);

const std::vector<int>* GetOrtTritonKernelByGroup(std::string group_name);

Status LaunchTritonKernel(cudaStream_t stream, size_t idx, int grid0, int grid1, int grid2, void* args, size_t args_size);

}  // namespace cuda
}  // namespace onnxruntime
