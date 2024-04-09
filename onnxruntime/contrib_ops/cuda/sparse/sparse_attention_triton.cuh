// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/providers/cuda/triton_kernel.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#ifdef USE_TRITON_KERNEL

namespace {

template <typename T>
std::string GetBlockSparseAttentionTritonGroupName() {
  std::string ret = "BlockSparseAttentionTriton_";
  ret += GetDataTypeName<T>();
  return ret;
}

}  // namespace

template <typename T>
auto GetTritonBlockSparseAttentionTypeStringAndOps() {
  std::vector<std::pair<std::string, tunable::Op<SparseAttentionTunableParams<T>>>> ret;

  auto group_name = GetBlockSparseAttentionTritonGroupName<T>();
  auto* kernel_list = GetOrtTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    auto* metadata = GetOrtTritonKernelMetadata(i);
    auto block_m = metadata->constants.at("BLOCK_M");
    auto block_d = metadata->constants.at("BLOCK_D");
    auto block_n = metadata->constants.at("BLOCK_N");
    auto even_m = metadata->constants.at("EVEN_M");
    auto even_n = metadata->constants.at("EVEN_N");
    auto num_blocks_d = metadata->constants.at("NUM_D_BLOCKS");

    auto impl = [i, block_m, block_n, even_m, even_n, block_d, num_blocks_d](
                    const SparseAttentionTunableParams<T>* params) -> Status {
      // Validate input parameters are compatible with the kernel.
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          even_m != static_cast<int>(params->sequence_length % block_m == 0),
          "sequence_length (", params->sequence_length, "), block_m (", block_m, "), even_m (", even_m, ") mismatch.");

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          even_n != static_cast<int>(params->total_sequence_length % block_n == 0),
          "k_seq_len (", params->total_sequence_length, "), block_n (", block_n, "), even_n (", even_n, ") mismatch.");

      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
          block_d * num_blocks_d != params->head_size,
          "head_size (", params->head_size, "), block_d (", block_d, "), num_blocks_d (", num_blocks_d, ") mismatch.");

      // Construct args for launch kernel
      struct {
        void* out;
        const void* q;
        const void* k;
        const void* v;
        const void* layout_crow_ptr;
        const void* layout_col_ptr;
        int layout_crow_stride_h;
        int layout_crow_stride_m;
        int layout_col_stride_h;
        int layout_col_stride_m;
        float softmax_scale;
        int stride_qz;
        int stride_qh;
        int stride_qm;
        int stride_qd;
        int stride_kz;
        int stride_kh;
        int stride_kn;
        int stride_kd;
        int stride_vz;
        int stride_vh;
        int stride_vn;
        int stride_vd;
        int stride_oz;
        int stride_oh;
        int stride_om;
        int stride_od;
        int num_heads;
        int total_sequence_length;
        int past_sequence_length;
      } args = {
          params->output,
          params->q,
          params->k,
          params->v,
          params->layout_crow,
          params->layout_col,
          params->layout_crow_stride_h,
          params->layout_crow_stride_m,
          params->layout_col_stride_h,
          params->layout_col_stride_m,
          params->softmax_scale,
          params->num_heads * params->sequence_length * params->head_size,
          params->sequence_length * params->head_size,
          params->head_size,
          1,
          params->num_heads * params->total_sequence_length * params->head_size,
          params->total_sequence_length * params->head_size,
          params->head_size,
          1,
          params->num_heads * params->total_sequence_length * params->head_size,
          params->total_sequence_length * params->head_size,
          params->head_size,
          1,
          params->num_heads * params->sequence_length * params->head_size,
          params->sequence_length * params->head_size,
          params->head_size,
          1,
          params->num_heads,
          params->total_sequence_length,
          params->past_sequence_length};

      int grid_0 = (params->sequence_length + block_m - 1) / block_m;
      int grid_1 = params->batch_size * params->num_heads;
      return LaunchTritonKernel(params->StreamHandle(), i, grid_0, grid_1, 1, &args, sizeof(args));
    };

    ret.emplace_back(std::make_pair(metadata->name, std::move(impl)));
  }
  return ret;
}

#endif  // USE_TRITON_KERNEL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
