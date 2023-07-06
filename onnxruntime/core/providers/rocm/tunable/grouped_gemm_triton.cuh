// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/providers/rocm/triton_kernel.h"

namespace onnxruntime {
namespace rocm {

#ifdef USE_TRITON_KERNEL

namespace {

template <typename T>
std::string GetGroupedGemmTritonGroupName() {
  std::string ret = "grouped_gemm_";
  ret += GetDataTypeName<T>();
  return ret;
}

}  // end of namespace

template <typename T, typename ALayout, typename BLayout>
auto GetGroupedGemmTritonOps() {
  std::vector<std::pair<std::string, tunable::Op<GroupedGemmParams<T>>>> ret;
  auto group_name = GetGroupedGemmTritonGroupName<T>();
  auto *kernel_list = GetOrtTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    // check params match
    auto *metadata = GetOrtTritonKernelMetadata(i);
    auto grid0 = -1;
    const std::string grid0_name = "GRID_SIZE0";
    if (metadata->constants.count(grid0_name) != 0) {
      grid0 = metadata->constants.at(grid0_name);
    }
    auto grid1 = -1;
    const std::string grid1_name = "GRID_SIZE1";
    if (metadata->constants.count(grid1_name) != 0) {
      grid1 = metadata->constants.at(grid1_name);
    }
    auto impl = [i, grid0, grid1](const GroupedGemmParams<T> *params) -> Status {
      // construct args for launch kernel
      struct {
	int num_matrix;
        const void* msizes;
	float alpha;
	const void* A;
	int lda;
	const void* B;
	int ldb;
	float beta;
        void *out;
	int ldc;
      } args = {params->num_matrix, (const void*)params->msizes, param->alpha, (const void*)params->a, params->lda, (const void*)params->b, params->ldb, params->beta, (void*)params->c, params->ldc};

      // grid dim is (batch_count, 1, 1)
      return LaunchTritonKernel(params->stream, i, grid0, grid1, 1, &args, sizeof(args));
    };
    ret.emplace_back(std::make_pair(metadata->name, std::move(impl)));
  }
  return ret;
}

#endif  // USE_TRITON_KERNEL

}  // namespace rocm
}  // namespace onnxruntime
