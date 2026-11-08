// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

inline constexpr int kMoEPackedMaxChunkRows = 2048;
// Covers 512/640-element K-loop increments and rounded 2048-element finalize tiles.
inline constexpr int64_t kMoEPackedMaxDimension = std::numeric_limits<int>::max() - 2047;

struct alignas(16) MoEPackedTile {
  int expert;
  int start_route;
  int count;
  int pad;
};

struct MoEPackedParams {
  int num_rows{0};
  int num_experts{0};
  int hidden_size{0};
  int inter_size{0};
  int k{0};
  bool normalize_routing_weights{true};
  bool bf16_tc_supported{false};
  int gate_type{0};
  int up_type{0};
  int down_type{0};
  int64_t gate_row_bytes{0};
  int64_t up_row_bytes{0};
  int64_t down_row_bytes{0};
};

Status GetMoEPackedWorkspaceBytes(const MoEPackedParams& params, size_t elem_bytes, size_t& workspace_bytes);

Status GetMoEPackedBf16KernelSupport(int device_id, bool& supported);

Status LaunchMoEPackedCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* input,
    const void* router_probs,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const MoEPackedParams& params,
    void* workspace,
    void* output);

template <typename T>
class MoEPacked final : public CudaKernel {
 public:
  explicit MoEPacked(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t k_{0};
  int64_t normalize_routing_weights_{1};
  std::string activation_type_;
  int64_t gate_type_{-1};
  int64_t up_type_{-1};
  int64_t down_type_{-1};
  bool bf16_tc_supported_{false};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
