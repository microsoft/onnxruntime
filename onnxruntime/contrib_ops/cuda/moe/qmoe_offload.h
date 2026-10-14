#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/qmoe_session_controller.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"
#include "contrib_ops/cuda/moe/qmoe_callback_timing.h"

namespace onnxruntime::contrib::cuda {

class QMoEOffload final : public onnxruntime::cuda::CudaKernel, public MoEBase {
 public:
  explicit QMoEOffload(const OpKernelInfo& info);
  ~QMoEOffload() override;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  enum Buffer : size_t {
    Fc1Weights,
    Fc2Weights,
    Fc1Scales,
    Fc2Scales,
    Fc1Bias,
    Fc2Bias,
    RouteBias,
    RouterScales,
    RouterIndices,
    UnitScales,
    IdentityPermutation,
    RouteIds,
    CompactExperts,
    WavePermutation,
    GatheredInput,
    WaveOutput,
    Contributions,
    RunnerWorkspace,
    BufferCount
  };

  using PinnedBuffer = decltype(std::declval<const onnxruntime::cuda::CudaKernel&>()
                                    .AllocateBufferOnCPUPinned<uint8_t>(0));
  using ResidentSlab = decltype(std::declval<const onnxruntime::cuda::CudaKernel&>()
                                    .GetScratchBuffer<uint8_t>(0, nullptr));

  void StageExperts(const int* experts, int count) const;
  void InitializeShared(const OpKernelInfo& info);
  void InitializeNvfp4Offload(const OpKernelInfo& info);
  Status ComputeShared(OpKernelContext* context) const;
  Status PromoteResident(cudaStream_t stream, const InlinedVector<InlinedVector<int>>& cold_routes,
                         int max_promotions = 0, const int* streamed_experts = nullptr,
                         int streamed_count = 0) const;
  void RecordResidentRoute(int phase, int expert, bool hit) const;
  void CommitInvocationObservation(const std::array<int, 5>& phase_rows,
                                   const std::array<std::vector<int>, 5>& phase_expert_rows) const;

  struct CpuWork;
  struct AsyncTile;
  struct SharedLayerCache;
  template <typename ActT, typename ParallelRunner>
  int RunColdExperts(CpuWork& work, int group_calls, const ActT* host_input, int tile_rows, ActT* host_result,
                     int* host_cpu_ids, bool compact, ParallelRunner&& run_parallel,
                     const char* skip_experts = nullptr) const;
  Status ComputeSharedAsync(OpKernelContext* context, const Tensor* input, const Tensor* router, const Tensor* bias,
                            const Tensor* phase, Tensor* output, int32_t* route_rows, int32_t* route_calls,
                            int32_t* counts, int tile_rows) const;
  void RunAsyncCpu(AsyncTile& tile) const;

  bool shared_mode_ = false;
  bool v3_compact_shared_ = false;
  bool nvfp4_mode_ = false;
  bool is_bf16_ = false;
  const Tensor* expert_blocks_ = nullptr;
  // NVFP4 multi-bank CPU tensors (when nvfp4_mode_ && !expert_blocks_)
  const Tensor* nvfp4_fc1_w_ = nullptr;
  const Tensor* nvfp4_fc1_s_ = nullptr;
  const Tensor* nvfp4_fc1_g_ = nullptr;
  const Tensor* nvfp4_fc1_b_ = nullptr;
  const Tensor* nvfp4_fc2_w_ = nullptr;
  const Tensor* nvfp4_fc2_s_ = nullptr;
  const Tensor* nvfp4_fc2_g_ = nullptr;
  const Tensor* nvfp4_fc2_b_ = nullptr;
  ResidentSlab fc2_bias_gpu_;
  size_t fc2_bias_gpu_bytes_ = 0;

  onnxruntime::contrib::qmoe_shared::ExpertBlockLayout shared_layout_;
  onnxruntime::contrib::qmoe_shared::Nvfp4ExpertBlockLayout nvfp4_layout_;
  onnxruntime::contrib::qmoe_shared::ExpertBlockLease expert_lease_;
  int64_t layer_index_ = 0;
  size_t shared_gpu_limit_ = 0;
  size_t shared_pinned_limit_ = 0;
  int group_calls_ = 1;

  // Decoupled dispatch controls (v3 Section 8.1 / F7).
  int max_sync_tile_tokens_ = 32;
  int async_max_tokens_ = 32;
  int stream_min_tokens_ = 33;
  int prefill_chunk_tokens_ = 0;

  bool async_ = false;
  bool experts_registered_ = false;
  std::vector<const void*> registered_host_ptrs_;
  bool async_aux_ = false;
  mutable cudaStream_t aux_stream_ = nullptr;
  mutable cudaEvent_t fork_event_ = nullptr;
  mutable cudaEvent_t join_event_ = nullptr;
  mutable bool slot_table_dirty_ = true;
  int cpu_threads_ = 0;
  int pool_task_batch_ = 1;
  bool unified_cpu_service_ = false;
  std::unique_ptr<CpuWork> async_work_;
  mutable std::atomic<bool> async_in_flight_{false};
  mutable std::atomic<bool> async_failed_{false};
  mutable std::string async_error_message_;
  std::unique_ptr<qmoe::CallbackTiming> callback_timing_;
  std::shared_ptr<onnxruntime::cuda::QMoESessionController> session_controller_;
  std::shared_ptr<SharedLayerCache> owned_layer_cache_;

  // Dynamic per-layer GPU expert cache with in-slab spare and dummy slots: P = C + S + D (v3 Section 7.4/7.5 / F3).
  int resident_slots_ = 0;       // C: logical resident capacity
  int spare_slots_ = 0;          // S: spare physical slots for promotions
  int dummy_slots_ = 0;          // D: 1 if CutlassMoeFCRunner async needs an immutable dummy slot, 0 for NVFP4 GEMV
  int physical_slots_ = 0;       // P = C + S + D
  int dummy_slot_index_ = -1;    // C + S when D == 1, -1 when D == 0
  int resident_promotions_ = 0;
  int resident_decay_routes_ = 0;
  int resident_margin_percent_ = 100;
  bool resident_phase_scores_ = false;
  bool continuous_decay_ = false;
  mutable bool resident_decode_active_ = false;
  std::array<size_t, 6> resident_bank_bytes_{};
  std::array<size_t, 6> resident_bank_offsets_{};
  size_t resident_workspace_bytes_ = 0;
  size_t stream_workspace_bytes_ = 0;
  ResidentSlab resident_slab_;
  ResidentSlab slot_table_dev_;
  mutable InlinedVector<int> slot_expert_;
  mutable InlinedVector<int> expert_slot_;
  mutable InlinedVector<float> expert_score_;
  mutable InlinedVector<float> decode_score_;
  mutable std::array<int, 2> phase_routes_since_decay_{};
  mutable std::array<uint64_t, 5> phase_hit_routes_{};
  mutable std::array<uint64_t, 5> phase_total_routes_{};
  mutable int routes_since_decay_ = 0;
  mutable uint64_t resident_hit_routes_ = 0;
  mutable uint64_t resident_total_routes_ = 0;
  mutable uint64_t resident_promoted_ = 0;
  mutable uint64_t placement_epoch_ = 0;
  bool resident_log_ = false;

  // Prefill streaming via session-owned landing ring (v3 Section 7.6 / F5).
  int stream_rows_ = 0;
  int stream_min_rows_ = 16;
  int stream_wave_ = 32;
  int stream_promotions_ = 16;
  bool stream_promotion_reuse_ = false;
  size_t stream_gpu_limit_ = 0;
  std::array<std::array<size_t, 6>, 2> stream_bank_offsets_{};
  size_t stream_landing_bytes_ = 0;
  std::shared_ptr<onnxruntime::cuda::QMoESessionLandingRing> stream_shared_;
  mutable uint64_t stream_tiles_ = 0;
  mutable uint64_t stream_experts_ = 0;
  mutable uint64_t stream_routes_ = 0;
  mutable uint64_t stream_waves_ = 0;
  mutable uint64_t stream_cpu_routes_ = 0;
  mutable uint64_t stream_reused_promotions_ = 0;

  bool cpu_timing_ = false;
  bool cpu_fp16_boundaries_ = false;
  int fc1_cols_ = 64;
  int fc2_cols_ = 128;
  mutable uint64_t timing_calls_ = 0;
  mutable uint64_t timing_routes_ = 0;
  mutable uint64_t timing_cold_routes_ = 0;
  mutable uint64_t timing_cold_experts_ = 0;
  mutable uint64_t timing_cpu_ns_ = 0;
  mutable uint64_t timing_classify_ns_ = 0;
  mutable uint64_t timing_prep_ns_ = 0;
  mutable uint64_t timing_fc1_ns_ = 0;
  mutable uint64_t timing_fc2_ns_ = 0;
  mutable uint64_t timing_write_ns_ = 0;

  std::array<const Tensor*, 6> banks_{};
  std::array<size_t, 6> expert_bytes_{};
  std::array<size_t, 6> staging_offsets_{};
  std::array<size_t, BufferCount> offsets_{};
  int num_experts_ = 0;
  int hidden_size_ = 0;
  int inter_size_ = 0;
  int token_tile_size_ = 0;
  int expert_wave_size_ = 0;
  int max_routes_ = 0;
  int max_wave_routes_ = 0;
  int staging_optimization_ = 0;
  size_t scratch_bytes_ = 0;
  size_t expert_slab_bytes_ = 0;
  size_t workspace_bytes_ = 0;
  size_t host_indices_offset_ = 0;
  size_t host_routes_offset_ = 0;
  size_t host_compact_offset_ = 0;
  PinnedBuffer pinned_ring_;
  mutable InlinedVector<int> expert_to_slot_;
  mutable InlinedVector<int> wave_experts_;
  std::unique_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface> runner_;
  mutable std::mutex mutex_;
  mutable bool stream_failed_ = false;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QMoEOffload);
};

}  // namespace onnxruntime::contrib::cuda
