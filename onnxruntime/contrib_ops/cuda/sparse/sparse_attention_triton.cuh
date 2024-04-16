// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/providers/cuda/triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#ifdef USE_TRITON_KERNEL

template <typename T>
struct SparseAttentionTunableParams;  // Defined in sparse_attention_tunable.h

template <typename T>
std::string GetBlockSparseAttentionTritonGroupName() {
  std::string ret = "BlockSparseAttentionTriton_";
  ret += ::onnxruntime::cuda::GetDataTypeName<T>();
  return ret;
}

template <typename T>
auto GetTritonBlockSparseAttentionTypeStringAndOps() {
  std::vector<std::pair<std::string, ::onnxruntime::cuda::tunable::Op<SparseAttentionTunableParams<T>>>> ret;

  auto group_name = GetBlockSparseAttentionTritonGroupName<T>();
  auto* kernel_list = ::onnxruntime::cuda::GetOrtTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    auto* metadata = ::onnxruntime::cuda::GetOrtTritonKernelMetadata(i);
    auto block_m = metadata->constants.at("BLOCK_M");
    auto block_d = metadata->constants.at("BLOCK_D");
    auto block_n = metadata->constants.at("BLOCK_N");
    auto even_m = metadata->constants.at("EVEN_M");
    auto even_n = metadata->constants.at("EVEN_N");
    auto num_blocks_d = metadata->constants.at("NUM_D_BLOCKS");

    auto impl = [i, block_m, block_n, even_m, even_n, block_d, num_blocks_d](
                    const SparseAttentionTunableParams<T>* params) -> Status {
      // Exclude kernels that are not compatible with the parameters.
      TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        even_m != static_cast<int>(params->sequence_length % block_m == 0) ||
          even_n != static_cast<int>(params->total_sequence_length % block_n == 0) ||
          block_d * num_blocks_d != params->head_size ||
          (block_m > 16 && params->sequence_length <= 16) ||
          block_n != params->kernel_block_size, "input parameters do mot match with SparseAttention kernel");

      int num_rows = (params->sequence_length + block_m - 1) / block_m;
      int num_cols = (params->total_sequence_length + block_n - 1) / block_n;

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
        int num_layout;
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
          1,
          params->layout_col_stride_h,
          1,
          params->num_layout,
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
      return onnxruntime::cuda::LaunchTritonKernel(params->StreamHandle(), i, grid_0, grid_1, 1, &args, sizeof(args));
    };

    ret.emplace_back(std::make_pair(metadata->name, std::move(impl)));
  }
  return ret;
}

#endif  // USE_TRITON_KERNEL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
