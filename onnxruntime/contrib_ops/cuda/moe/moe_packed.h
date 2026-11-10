// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/framework/moe_offload_plan.h"
#include "core/providers/cuda/qmoe_session_controller.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

inline constexpr int kMoEPackedMaxChunkRows = 2048;
inline constexpr int kMoEPackedMaxOffloadChunkRows = 8192;
// Covers 512/640-element K-loop increments and rounded 2048-element finalize tiles.
inline constexpr int64_t kMoEPackedMaxDimension = std::numeric_limits<int>::max() - 2047;
constexpr int kMoEPackedMaxDecodeRows = 16;
constexpr int kMoEPackedMaxDecodeK = 16;
constexpr int kMoEPackedMaxDecodeRoutes = kMoEPackedMaxDecodeRows * kMoEPackedMaxDecodeK;
constexpr int kMoEPackedMaxHiddenSize = 4096;

struct alignas(16) MoEPackedTile {
  int expert;
  int start_route;
  int count;
  int pad;
};

struct alignas(64) MoEPackedDecodePinned {
  volatile int32_t doorbell_seq{0};
  volatile int32_t doorbell_done{0};
  int32_t num_rows{0};
  int32_t cold_count{0};
  int32_t cold_routes[kMoEPackedMaxDecodeRoutes];
  int32_t cold_experts[kMoEPackedMaxDecodeRoutes];
  float cold_scales[kMoEPackedMaxDecodeRoutes];
  uint16_t host_input[kMoEPackedMaxDecodeRows * kMoEPackedMaxHiddenSize];
  // FP32 sum of the scaled cold-route outputs per token. The finalize kernel adds it to the FP32 resident sum and
  // rounds to T once, as the GPU-only path does; rounding this partial sum to T first would round the output twice.
  float host_cold_output[kMoEPackedMaxDecodeRows * kMoEPackedMaxHiddenSize];
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

Status LaunchMoEPackedOffloadDecodeCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* input,
    const void* router_probs,
    const uint8_t* gpu_gate_bank,
    const uint8_t* gpu_up_bank,
    const uint8_t* gpu_down_bank,
    const int* d_expert_to_slot,
    int resident_slots,
    const MoEPackedParams& params,
    float tail_sub_threshold,
    int tail_sub_min_rank,
    float min_expert_weight,
    bool skip_tail_cold_only,
    void* d_decode_ws,
    MoEPackedDecodePinned* pinned_dev,
    void* output);

Status LaunchMoEPackedSoftmaxTopKCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* router_probs,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales);

Status LaunchMoEPackedGroupTilesCuda(
    cudaStream_t stream,
    bool is_bf16,
    bool use_prefill_tiles,
    const void* input,
    const uint8_t* gate_bank,
    const uint8_t* up_bank,
    const uint8_t* down_bank,
    const int* d_sorted_routes,
    const MoEPackedTile* d_tiles,
    const int* d_num_tiles,
    int max_tiles,
    const MoEPackedParams& params,
    void* d_fc1_out,
    void* d_fc2_out);

Status LaunchMoEPackedFinalizeOnlyCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* d_fc2_out,
    const float* d_topk_scales,
    int num_rows,
    int hidden_size,
    int k,
    void* output);

struct MoEPackedOffloadState;

template <typename T>
class MoEPacked final : public CudaKernel, public onnxruntime::moe_offload::IMoeOffloadNode {
 public:
  explicit MoEPacked(const OpKernelInfo& info);
  ~MoEPacked() override;
  Status ComputeInternal(OpKernelContext* context) const override;

  const onnxruntime::moe_offload::MoeExpertSourceDescriptor& GetSourceDescriptor() const noexcept override {
    return offload_source_desc_;
  }
  const onnxruntime::moe_offload::MoeBackendCapabilities& GetBackendCapabilities() const noexcept override {
    return offload_capabilities_;
  }
  Status InitializeAdmittedPlan(const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                                const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan) override;

 private:
  int64_t k_{0};
  int64_t normalize_routing_weights_{1};
  std::string activation_type_;
  int64_t gate_type_{-1};
  int64_t up_type_{-1};
  int64_t down_type_{-1};
  bool bf16_tc_supported_{false};

  onnxruntime::moe_offload::MoeOffloadSessionPolicy offload_policy_{};
  onnxruntime::moe_offload::MoeExpertSourceDescriptor offload_source_desc_{};
  onnxruntime::moe_offload::MoeBackendCapabilities offload_capabilities_{};
  onnxruntime::moe_offload::MoeNodeExecutionPlan offload_plan_{};
  int32_t offload_layer_index_{-1};
  bool offload_node_registered_{false};
  std::shared_ptr<MoEPackedOffloadState> offload_state_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
