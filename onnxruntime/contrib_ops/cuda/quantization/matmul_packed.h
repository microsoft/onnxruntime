// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// Leave room for the largest (32-column) decode tile.
inline constexpr int64_t kMatMulPackedMaxDimension = std::numeric_limits<int>::max() - 31;
inline constexpr int kMatMulPackedPrefillNTile = 4096;
inline constexpr int kMatMulPackedPrefillMTile = 128;
inline constexpr int kMatMulPackedBf16TcMinN = 65536;

struct MatMulPackedParams {
  const uint8_t* b_data{nullptr};
  int k{0};
  int n{0};
  int64_t row_bytes{0};
  int ggml_type{0};
  bool bf16_tc_supported{false};
  bool fp32_prefill{false};
};

Status GetMatMulPackedBf16KernelSupport(int device_id, bool& supported);

Status GetMatMulPackedWorkspaceBytes(const MatMulPackedParams& params, int64_t m, size_t& workspace_bytes);

Status LaunchMatMulPackedCuda(
    cublasHandle_t cublas,
    cudaStream_t stream,
    bool is_bf16,
    const void* a,
    const MatMulPackedParams& params,
    void* scratch_w,
    void* y,
    int64_t m);

template <typename T>
class MatMulPacked final : public CudaKernel {
 public:
  explicit MatMulPacked(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t k_{0};
  int64_t n_{0};
  int64_t ggml_type_{-1};
  bool bf16_tc_supported_{false};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
