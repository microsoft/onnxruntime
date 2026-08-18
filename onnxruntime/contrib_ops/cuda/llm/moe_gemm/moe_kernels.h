/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "cutlass/gemm/gemm.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/quantization.h"
#include "contrib_ops/cuda/llm/nv_infer_datatype.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
#include <cuda_runtime_api.h>
#include <limits>

#include <array>
#include <map>
#include <optional>
#include <random>
#include <utility>

namespace onnxruntime::llm::kernels {

namespace cutlass_kernels {
/**
 * \brief Describes what parallelism mode the MoE is using
 *
 * Tensor Parallelism refers to the mode where the weight matrices for each expert are sliced up between nodes.
 * Each node will handle part of each expert, the final result is achieved by summing the result.
 * The inter_size dimension should be divided by the number of nodes prior to passing it to the MoE plugin, only the
 * required slice of the weights should be provided to the plugin FC1 is a ColumnLinear and FC2 is a RowLinear, see
 * tensorrt_llm/mlp/mlp.py for an example of how this works for a single MLP
 *
 * NOTE: The bias for fc2 is only applied on rank 0. If we added it on all nodes the allreduce() would contain multiple
 * copies of the bias. The bias on other node will be ignored, and may be set to nullptr
 *
 * Expert Parallelism refers to the mode where experts are divided between the nodes. Each node will handle only the
 * tokens that are routed to the experts it is assigned to. Only the weights for the node's experts should be provided
 * to the plugin For example, with #experts = 8, expert parallelism = 2: Node 0 would handle experts 0-3, and node 1
 * would handle experts 4-7
 *
 * Regardless of parallelism mode:
 *  * The input routing values must be the complete routing for all tokens/experts (required for softmax)
 *  * An allreduce must be run on the result to combine the results from different nodes if parallelism > 1
 */
struct MOEParallelismConfig {
  int tp_size = 1;
  int tp_rank = 0;
  int ep_size = 1;
  int ep_rank = 0;
  int cluster_size = 1;
  int cluster_rank = 0;

  MOEParallelismConfig() = default;

  MOEParallelismConfig(int tp_size, int tp_rank, int ep_size, int ep_rank)
      : tp_size(tp_size), tp_rank(tp_rank), ep_size(ep_size), ep_rank(ep_rank), cluster_size(1), cluster_rank(0) {
    // Do some basic sanity checks
    ORT_ENFORCE(tp_rank < tp_size);
    ORT_ENFORCE(tp_rank >= 0);
    ORT_ENFORCE(tp_size >= 1);
    ORT_ENFORCE(ep_rank < ep_size);
    ORT_ENFORCE(ep_rank >= 0);
    ORT_ENFORCE(ep_size >= 1);
  }

  MOEParallelismConfig(int tp_size, int tp_rank, int ep_size, int ep_rank, int cluster_size, int cluster_rank)
      : tp_size(tp_size), tp_rank(tp_rank), ep_size(ep_size), ep_rank(ep_rank), cluster_size(cluster_size), cluster_rank(cluster_rank) {
    // Do some basic sanity checks
    ORT_ENFORCE(tp_rank < tp_size);
    ORT_ENFORCE(tp_rank >= 0);
    ORT_ENFORCE(tp_size >= 1);
    ORT_ENFORCE(ep_rank < ep_size);
    ORT_ENFORCE(ep_rank >= 0);
    ORT_ENFORCE(ep_size >= 1);
    ORT_ENFORCE(cluster_rank < cluster_size);
    ORT_ENFORCE(cluster_rank >= 0);
    ORT_ENFORCE(cluster_size >= 1);
    ORT_ENFORCE(ep_size == 1 || cluster_size == 1);
  }

  bool operator==(const MOEParallelismConfig& other) const {
    return tp_size == other.tp_size && tp_rank == other.tp_rank && ep_size == other.ep_size && ep_rank == other.ep_rank && cluster_size == other.cluster_size && cluster_rank == other.cluster_rank;
  }

  friend std::ostream& operator<<(std::ostream& os, const MOEParallelismConfig& config) {
    os << "tp_size: " << config.tp_size << ", tp_rank: " << config.tp_rank << ", ep_size: " << config.ep_size
       << ", ep_rank: " << config.ep_rank << ", cluster_size: " << config.cluster_size
       << ", cluster_rank: " << config.cluster_rank;
    return os;
  }
};

struct QuantParams {
  // Int weight only quantization params
  struct
  {
    const void* fc1_weight_scales = nullptr;
    const void* fc2_weight_scales = nullptr;
  } wo;

  // FP8 quantization params
  struct
  {
    bool fc2_use_per_expert_act_scale = false;
    const float* dequant_fc1 = nullptr;    // (num_experts_per_node, )
    const float* quant_fc2 = nullptr;      // (1, ) or (num_experts_per_node, ) based on fc2_use_per_expert_act_scale
    const float* dequant_fc2 = nullptr;    // (num_experts_per_node, )
    const float* quant_final = nullptr;    // (1, )
    const float* dequant_input = nullptr;  // (1, )
  } fp8;

  // FP8 MXFP4 quantization params
  // This mode uses regular global scale for FP8 activations and block scaling for MXFP4 weights
  struct FP8MXFP4Inputs {
    struct GemmInputs {
      bool use_per_expert_act_scale = false;
      const float* act_global_scale = nullptr;                                                 // (1, ) or (num_experts_per_node, ) based on use_per_expert_act_scale
      const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF* weight_block_scale = nullptr;  // (experts, n, k / 32)
      const float* global_scale = nullptr;                                                     // (num_experts_per_node, )
    };

    GemmInputs fc1;
    GemmInputs fc2;
  } fp8_mxfp4;

  // MXFP8 MXFP4 quantization params
  // This mode uses block scaled MXFP8 and MXFP4 weights
  struct MXFP8MXFP4Inputs {
    struct GemmInputs {
      const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF* weight_block_scale = nullptr;  // (experts, n, k / 32)
      const float* global_scale = nullptr;                                                     // (num_experts_per_node, )
    };

    GemmInputs fc1;
    GemmInputs fc2;
  } mxfp8_mxfp4;

  // FP4 quantization params
  struct FP4Inputs {
    struct GemmInputs {
      bool use_per_expert_act_scale = false;

      const float* act_global_scale = nullptr;                                                 // (1, ) or (num_experts_per_node, ) based on use_per_expert_act_scale
      const TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF* weight_block_scale = nullptr;  // (experts, n, k / 16)
      const float* global_scale = nullptr;                                                     // (num_experts_per_node, )
    };

    GemmInputs fc1;
    GemmInputs fc2;
  } fp4;

  // GPTQ/AWQ quantization params
  struct GroupwiseInputs {
    struct GroupwiseGemmInputs {
      const void* act_scales = nullptr;
      const void* weight_scales = nullptr;
      const void* weight_zeros = nullptr;
      const float* alpha = nullptr;
    };

    int group_size = -1;
    GroupwiseGemmInputs fc1;
    GroupwiseGemmInputs fc2;
  } groupwise;

  static QuantParams Int(const void* fc1_weight_scales, const void* fc2_weight_scales) {
    QuantParams qp;
    qp.wo = {fc1_weight_scales, fc2_weight_scales};
    return qp;
  }

  static QuantParams FP8(const float* dequant_fc1, const float* quant_fc2, const float* dequant_fc2,
                         const float* quant_final = nullptr, const float* dequant_input = nullptr,
                         bool fc2_use_per_expert_act_scale = false) {
    QuantParams qp;
    qp.fp8 = {fc2_use_per_expert_act_scale, dequant_fc1, quant_fc2, dequant_fc2, quant_final, dequant_input};
    return qp;
  }

  static QuantParams FP8MXFP4(const float* fc1_act_global_scale,
                              const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF* fc1_weight_block_scale,
                              const float* fc1_global_scale,  //
                              const float* fc2_act_global_scale,
                              const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF* fc2_weight_block_scale,
                              const float* fc2_global_scale,  //
                              bool fc1_use_per_expert_act_scale = false, bool fc2_use_per_expert_act_scale = false) {
    QuantParams qp;
    qp.fp8_mxfp4.fc1 = {fc1_use_per_expert_act_scale, fc1_act_global_scale, fc1_weight_block_scale, fc1_global_scale};
    qp.fp8_mxfp4.fc2 = {fc2_use_per_expert_act_scale, fc2_act_global_scale, fc2_weight_block_scale, fc2_global_scale};
    return qp;
  }

  static QuantParams MXFP8MXFP4(const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF* fc1_weight_block_scale,
                                const float* fc1_global_scale,  //
                                const TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF* fc2_weight_block_scale, const float* fc2_global_scale) {
    QuantParams qp;
    qp.mxfp8_mxfp4.fc1 = {fc1_weight_block_scale, fc1_global_scale};
    qp.mxfp8_mxfp4.fc2 = {fc2_weight_block_scale, fc2_global_scale};
    return qp;
  }

  static QuantParams FP4(const float* fc1_act_global_scale,
                         const TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF* fc1_weight_block_scale,
                         const float* fc1_global_scale,  //
                         const float* fc2_act_global_scale,
                         const TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF* fc2_weight_block_scale,
                         const float* fc2_global_scale,  //
                         bool fc1_use_per_expert_act_scale = false, bool fc2_use_per_expert_act_scale = false)

  {
    QuantParams qp;
    qp.fp4.fc1 = {fc1_use_per_expert_act_scale, fc1_act_global_scale, fc1_weight_block_scale, fc1_global_scale};
    qp.fp4.fc2 = {fc2_use_per_expert_act_scale, fc2_act_global_scale, fc2_weight_block_scale, fc2_global_scale};
    return qp;
  }

  static QuantParams GroupWise(int group_size, const void* fc1_weight_scales, const void* fc2_weight_scales,
                               const void* fc1_activation_scales = nullptr, const void* fc2_activation_scales = nullptr,
                               const void* fc1_weight_zeros = nullptr, const void* fc2_weight_zeros = nullptr,
                               const float* fc1_alpha = nullptr, const float* fc2_alpha = nullptr) {
    QuantParams qp;
    qp.groupwise.group_size = group_size;
    qp.groupwise.fc1 = {fc1_activation_scales, fc1_weight_scales, fc1_weight_zeros, fc1_alpha};
    qp.groupwise.fc2 = {fc2_activation_scales, fc2_weight_scales, fc2_weight_zeros, fc2_alpha};
    return qp;
  }
};

// Optional fused routing (optimization C). When router_logits is non-null, runMoe computes the
// softmax + top-k itself, fused into the kernel that builds the expert permutation maps, instead of
// the caller launching a separate softmax/top-k kernel first. At decode both are single-block
// kernels dominated by launch latency, so merging them removes close to a full kernel's overhead
// per layer.
//
// The caller must have checked isFusedMoeRoutingSupported() for the same shape parameters, and must
// pass the same buffers it passes as runMoe's token_selected_experts / token_final_scales arguments
// (runMoe fills them before reading them). router_logits is [num_rows, num_experts] and its element
// type must match the runner's InputType.
struct FusedRoutingParams {
  void const* router_logits{nullptr};
  int* token_selected_experts{nullptr};
  float* token_final_scales{nullptr};
  bool normalize_routing_weights{false};
};

// Host-side, deterministic predicate for the fused routing prologue. It must be the single source
// of truth: the caller uses it to decide whether to skip its own softmax/top-k launch, and runMoe
// re-checks it, so the two can never disagree about who computed the routing.
bool isFusedMoeRoutingSupported(int64_t const num_tokens, int const num_experts,
                                int const num_experts_per_node, int const experts_per_token,
                                int const ep_size);

class CutlassMoeFCRunnerInterface {
 public:
  virtual ~CutlassMoeFCRunnerInterface() = default;
  virtual size_t getWorkspaceSize(const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size,
                                  const int num_experts, const int experts_per_token, ActivationType activation_type,
                                  MOEParallelismConfig parallelism_config, bool use_awq) = 0;
  virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                         std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) = 0;
  virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

  // wfp4a16 only: route prefill through the SM80 fused-dequant grouped GEMM (vs the SM90 TMA WS
  // path). The QMoE op decides this at construction time (reading the relevant env vars then) and
  // pushes it in here, so inference-time config selection does not depend on the live environment.
  // Default no-op for runners that do not implement the SM80 FP4 path.
  virtual void setUseSm80Fp4(bool /*use_sm80_fp4*/) {}

  virtual void runMoe(const void* input_activations, const void* input_sf, const int* token_selected_experts,
                      const float* token_final_scales, const void* fc1_expert_weights, const void* fc1_expert_biases,
                      ActivationType fc1_activation_type, const void* fc2_expert_weights, const void* fc2_expert_biases,
                      QuantParams quant_params, const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size,
                      const int num_experts, const int experts_per_token, char* workspace_ptr, void* final_output,
                      int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config,
                      ActivationParameters activation_params, FusedRoutingParams fused_routing,
                      cudaStream_t stream) = 0;

  // Aliases for profiling the gemms
  virtual void gemm1(const void* const input, void* const output, void* const intermediate_result,
                     const int64_t* const expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
                     const void* const fc1_expert_weights, const void* const fc1_expert_biases,
                     const int64_t* const num_valid_tokens_ptr, const void* const fc1_int_scales, const float* const fc1_fp8_dequant,
                     const float* const fc2_fp8_quant, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_fp4_act_flat,
                     TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params,
                     const int64_t num_rows, const int64_t expanded_num_rows, const int64_t hidden_size, const int64_t inter_size,
                     const int num_experts_per_node, ActivationType fc1_activation_type, const float** alpha_scale_ptr_array,
                     bool bias_is_broadcast, cudaStream_t stream,
                     cutlass_extensions::CutlassGemmConfig config,
                     ActivationParameters activation_params) = 0;

  virtual void gemm2(const void* const input, void* const gemm_output, void* const final_output,
                     const int64_t* const expert_first_token_offset, const TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
                     const void* const fc2_expert_weights, const void* const fc2_expert_biases, const void* const fc2_int_scales,
                     const float* const fc2_fp8_dequant, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat,
                     QuantParams quant_params, const float* const token_topk_unpermuted_scales,
                     const float* const token_topk_permuted_scales, const int* const unpermuted_row_to_permuted_row,
                     const int* permuted_row_to_unpermuted_row, const int* const token_selected_experts,
                     const int64_t* const num_valid_tokens_ptr, const int64_t num_rows, const int64_t expanded_num_rows,
                     const int64_t hidden_size, const int64_t inter_size, const int num_experts_per_node,
                     const int64_t experts_per_token, const float** alpha_scale_ptr_array,
                     cudaStream_t stream, MOEParallelismConfig parallelism_config,
                     cutlass_extensions::CutlassGemmConfig config) = 0;

  virtual std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecializedDispatch(const int64_t* expert_first_token_offset,
                                           TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                           int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
                                           int64_t gemm2_k, const int num_experts_per_node, const void* gemm1_in, const void* gemm2_in,
                                           const void* weights1, const void* weights2, const float* alpha_scale_flat1, const float* alpha_scale_flat2,
                                           const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat1,
                                           const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat2, QuantParams quant_params, const void* bias1,
                                           const void* bias2, void* gemm1_output, void* gemm2_output, cudaStream_t stream) = 0;

  virtual size_t getGemmWorkspaceSize(int num_experts_per_node) const = 0;

  bool is_profiler = false;
  bool use_deterministic_hopper_reduce_ = false;
};

// Assumes inputs activations are row major. Weights need to be preprocessed by th_op/weight_quantize.cc .
// Nested in a class to avoid multiple calls to cudaGetDeviceProperties as this call can be expensive.
// Avoid making several duplicates of this class.
template <typename T,                         /*The type used for activations*/
          typename WeightType,                /* The type for the MoE weights */
          typename OutputType = T,            /* The type for the MoE final output */
          typename InputType = T,             /* The type for the MoE input */
          typename BackBoneType = OutputType, /* The unquantized backbone data type of the model */
          typename Enable = void>
class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface {
  using ScaleBiasType = BackBoneType;
  using Self = CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType>;
#if defined(ENABLE_FP8)
  static constexpr bool use_fp8 = (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) && !std::is_same_v<WeightType, cutlass::uint4b_t>;
  static constexpr bool use_w4afp8 = std::is_same_v<WeightType, cutlass::uint4b_t> && std::is_same_v<T, __nv_fp8_e4m3>;
  // W8A16-FP8: FP8 e4m3 weights with FP16/BF16 activations
#if defined(ENABLE_BF16)
  static constexpr bool use_wfp8a16 = std::is_same_v<WeightType, __nv_fp8_e4m3> && (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>);
#else
  static constexpr bool use_wfp8a16 = std::is_same_v<WeightType, __nv_fp8_e4m3> && std::is_same_v<T, half>;
#endif
  static_assert(!std::is_same_v<BackBoneType, __nv_fp8_e4m3>, "Current logic requires backbone type to be >=16-bits");
  static_assert(!std::is_same_v<OutputType, __nv_fp8_e4m3>, "Current logic requires output type to be >=16-bits");
#else
  static constexpr bool use_fp8 = false;
  static constexpr bool use_w4afp8 = false;
  static constexpr bool use_wfp8a16 = false;
#endif
#if defined(ENABLE_FP4)
  static constexpr bool act_fp4 = std::is_same_v<T, __nv_fp4_e2m1>;
  static constexpr bool weight_fp4 = std::is_same_v<WeightType, __nv_fp4_e2m1>;
  static constexpr bool use_wfp4afp8 = std::is_same_v<T, __nv_fp8_e4m3> && weight_fp4;
  static constexpr bool use_fp4 = act_fp4 && weight_fp4;
#if defined(ENABLE_BF16)
  static constexpr bool use_wfp4a16 = weight_fp4 && (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>);
#else
  static constexpr bool use_wfp4a16 = weight_fp4 && std::is_same_v<T, half>;
#endif
  static_assert(!std::is_same_v<BackBoneType, __nv_fp4_e2m1>, "Current logic requires backbone type to be >=16-bits");
  static_assert(!std::is_same_v<OutputType, __nv_fp4_e2m1>, "Current logic requires output type to be >=16-bits");
#else
  static constexpr bool act_fp4 = false;
  static constexpr bool weight_fp4 = false;
  static constexpr bool use_wfp4afp8 = false;
  static constexpr bool use_fp4 = false;
  static constexpr bool use_wfp4a16 = false;
#endif

  // Added by ORT

  ActivationType activation_type_;
  bool normalize_routing_weights_;
  bool use_sparse_mixer_;
  int sm_;

  static constexpr bool use_block_scaling = use_fp4 || use_wfp4afp8;

  // This should leave the variable unchanged in any currently supported configuration
  using UnfusedGemmOutputType = BackBoneType;

  // We introduce this as a separate parameter, so that if we ever remove the above condition we can decouple
  // BackBoneType and OutputType easily. For now these are required to be equivalent
  static_assert(std::is_same_v<OutputType, BackBoneType>, "Scale and bias types must match OutputType");

 public:
  CutlassMoeFCRunner(int sm_version, ActivationType activation_type, bool normalize_routing_weights, bool use_sparse_mixer);

  ~CutlassMoeFCRunner() override = default;

  static_assert(
      std::is_same_v<T, WeightType> || !std::is_same_v<T, float>, "Does not support float with quantized weights");

  size_t getWorkspaceSize(const int64_t num_rows, const int64_t hidden_size, const int64_t fc1_output_size,
                          const int num_experts, const int experts_per_token, ActivationType activation_type,
                          MOEParallelismConfig parallelism_config, bool use_awq) override;

  void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                 std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) override {
    // Only overwrite if a valid config is provided; preserve constructor defaults when profiling
    // cannot find a valid tactic (e.g. problem dimensions too small for available tile shapes).
    if (gemm1_config.has_value()) gemm1_config_ = std::move(gemm1_config);
    if (gemm2_config.has_value()) gemm2_config_ = std::move(gemm2_config);
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override {
    return moe_gemm_runner_.getConfigs();
  }

  // Push the QMoE op's SM80-FP4 decision into the inner runner and re-pick a valid default tactic
  // from the (now filtered) config list, so even if profiling later finds nothing, the preserved
  // default matches the selected GEMM family (SM80 Ampere vs SM90 TMA WS).
  void setUseSm80Fp4(bool use_sm80_fp4) override {
    moe_gemm_runner_.setUseSm80Fp4(use_sm80_fp4);
    auto tactics = moe_gemm_runner_.getConfigs();
    if (!tactics.empty()) {
      gemm1_config_ = tactics[0];
      gemm2_config_ = tactics[0];
    }
  }

  static std::vector<cutlass_extensions::CutlassGemmConfig> getTactics(int sm) {
    using RunnerType = decltype(moe_gemm_runner_);
    return RunnerType::getConfigs(sm);
  }

  void runMoe(const void* input_activations, const void* input_sf, const int* token_selected_experts,
              const float* token_final_scales, const void* fc1_expert_weights, const void* fc1_expert_biases,
              ActivationType fc1_activation_type, const void* fc2_expert_weights, const void* fc2_expert_biases,
              QuantParams quant_params, const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size,
              const int num_experts, const int experts_per_token, char* workspace_ptr, void* final_output,
              int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config,
              ActivationParameters activation_params, FusedRoutingParams fused_routing,
              cudaStream_t stream) override;

  // We make these GEMM1 & GEMM2 static because they need to be stateless for the profiler to work
  static void gemm1(MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
                    const T* const input, T* const output,
                    void* const intermediate_result, const int64_t* const expert_first_token_offset,
                    const TmaWarpSpecializedGroupedGemmInput tma_ws_input_template, const WeightType* const fc1_expert_weights,
                    const ScaleBiasType* const fc1_expert_biases, const int64_t* const num_valid_tokens_ptr,
                    const ScaleBiasType* const fc1_int_scales, const float* const fc1_fp8_dequant, const float* const fc2_fp8_quant,
                    const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_fp4_act_flat,
                    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params,
                    const int64_t num_rows, const int64_t expanded_num_rows, const int64_t hidden_size, const int64_t inter_size,
                    const int num_experts_per_node, ActivationType fc1_activation_type, const float** alpha_scale_ptr_array,
                    const int* permuted_row_to_expert, float* moe_gemv_splitk_partials,
                    const int* permuted_row_to_source_row, bool bias_is_broadcast,
                    cudaStream_t stream, MOEParallelismConfig parallelism_config,
                    cutlass_extensions::CutlassGemmConfig config,
                    ActivationParameters activation_params);

  static void gemm2(MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
                    const T* const input, void* const gemm_output,
                    OutputType* const final_output, const int64_t* const expert_first_token_offset,
                    const TmaWarpSpecializedGroupedGemmInput tma_ws_input_template, const WeightType* const fc2_expert_weights,
                    const ScaleBiasType* const fc2_expert_biases, const ScaleBiasType* const fc2_int_scales,
                    const float* const fc2_fp8_dequant, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat,
                    QuantParams quant_params, const float* const token_topk_unpermuted_scales,
                    const float* const token_topk_permuted_scales, const int* const unpermuted_row_to_permuted_row,
                    const int* permuted_row_to_unpermuted_row, const int* const token_selected_experts,
                    const int64_t* const num_valid_tokens_ptr, const int64_t num_rows, const int64_t expanded_num_rows,
                    const int64_t hidden_size, const int64_t inter_size, const int num_experts_per_node,
                    const int64_t experts_per_token, const float** alpha_scale_ptr_array,
                    const int* permuted_row_to_expert,
                    cudaStream_t stream, MOEParallelismConfig parallelism_config,
                    cutlass_extensions::CutlassGemmConfig config);

  // Overrides to allow us to forward on to the internal functions with the pointers using the correct type
  void gemm1(const void* const input, void* const output, void* const intermediate_result,
             const int64_t* const expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
             const void* const fc1_expert_weights, const void* const fc1_expert_biases,
             const int64_t* const num_valid_tokens_ptr, const void* const fc1_int_scales, const float* const fc1_fp8_dequant,
             const float* const fc2_fp8_quant, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_fp4_act_flat,
             TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params,
             const int64_t num_rows, const int64_t expanded_num_rows, const int64_t hidden_size, const int64_t inter_size,
             const int num_experts_per_node, ActivationType fc1_activation_type, const float** alpha_scale_ptr_array,
             bool bias_is_broadcast, cudaStream_t stream,
             cutlass_extensions::CutlassGemmConfig config,
             ActivationParameters activation_params) override {
    return Self::gemm1(moe_gemm_runner_, static_cast<const T*>(input),
                       static_cast<T*>(output), intermediate_result, expert_first_token_offset, tma_ws_input_template,
                       static_cast<const WeightType*>(fc1_expert_weights), static_cast<const ScaleBiasType*>(fc1_expert_biases),
                       num_valid_tokens_ptr, static_cast<const ScaleBiasType*>(fc1_int_scales), fc1_fp8_dequant, fc2_fp8_quant,
                       fc1_fp4_act_flat, fc2_fp4_act_flat, quant_params, num_rows, expanded_num_rows, hidden_size, inter_size,
                       num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array, nullptr, nullptr, nullptr,
                       bias_is_broadcast, stream, MOEParallelismConfig{}, config, activation_params);
  }

  void gemm2(const void* const input, void* const gemm_output, void* const final_output,
             const int64_t* const expert_first_token_offset, const TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
             const void* const fc2_expert_weights, const void* const fc2_expert_biases, const void* const fc2_int_scales,
             const float* const fc2_fp8_dequant, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat,
             QuantParams quant_params, const float* const token_topk_unpermuted_scales,
             const float* const token_topk_permuted_scales, const int* const unpermuted_row_to_permuted_row,
             const int* permuted_row_to_unpermuted_row, const int* const token_selected_experts,
             const int64_t* const num_valid_tokens_ptr, const int64_t num_rows, const int64_t expanded_num_rows,
             const int64_t hidden_size, const int64_t inter_size, const int num_experts_per_node,
             const int64_t experts_per_token, const float** alpha_scale_ptr_array,
             cudaStream_t stream, MOEParallelismConfig parallelism_config,
             cutlass_extensions::CutlassGemmConfig config) override {
    return Self::gemm2(moe_gemm_runner_, static_cast<const T*>(input), gemm_output,
                       static_cast<OutputType*>(final_output), expert_first_token_offset, tma_ws_input_template,
                       static_cast<const WeightType*>(fc2_expert_weights), static_cast<const ScaleBiasType*>(fc2_expert_biases),
                       static_cast<const ScaleBiasType*>(fc2_int_scales), fc2_fp8_dequant, fc2_fp4_act_flat, quant_params,
                       token_topk_unpermuted_scales, token_topk_permuted_scales, unpermuted_row_to_permuted_row,
                       permuted_row_to_unpermuted_row, token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
                       hidden_size, inter_size, num_experts_per_node, experts_per_token, alpha_scale_ptr_array, nullptr,
                       stream, parallelism_config, config);
  }

  virtual size_t getGemmWorkspaceSize(int num_experts_per_node) const override {
    return moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);
  }

  std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecializedDispatch(const int64_t* expert_first_token_offset,
                                           TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                           int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
                                           int64_t gemm2_k, const int num_experts_per_node, const void* gemm1_in, const void* gemm2_in,
                                           const void* weights1, const void* weights2, const float* alpha_scale_flat1, const float* alpha_scale_flat2,
                                           const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat1,
                                           const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat2, QuantParams quant_params, const void* bias1,
                                           const void* bias2, void* gemm1_output, void* gemm2_output, cudaStream_t stream) override {
    return Self::computeStridesTmaWarpSpecialized(expert_first_token_offset, layout_info1, layout_info2, num_tokens,
                                                  expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts_per_node,
                                                  reinterpret_cast<const T*>(gemm1_in), reinterpret_cast<const T*>(gemm2_in),
                                                  reinterpret_cast<const WeightType*>(weights1), reinterpret_cast<const WeightType*>(weights2),
                                                  alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2, quant_params,
                                                  reinterpret_cast<const ScaleBiasType*>(bias1), reinterpret_cast<const ScaleBiasType*>(bias2),
                                                  reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
                                                  reinterpret_cast<UnfusedGemmOutputType*>(gemm2_output), stream);
  }

 private:
  std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput> setupTmaWarpSpecializedInputs(
      int64_t num_rows, int64_t expanded_num_rows, ActivationType fc1_activation_type, bool use_fused_gated_activation,
      int64_t hidden_size, int64_t inter_size, int64_t num_experts_per_node, const void* input_activations_void,
      const TmaWarpSpecializedGroupedGemmInput::ElementSF* input_sf, void* final_output,
      const WeightType* fc1_expert_weights, const WeightType* fc2_expert_weights, QuantParams quant_params,
      const ScaleBiasType* fc1_expert_biases, const ScaleBiasType* fc2_expert_biases,
      int start_expert,
      MOEParallelismConfig parallelism_config, cudaStream_t stream);

  static std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecialized(const int64_t* expert_first_token_offset,
                                   TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                   int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
                                   int64_t gemm2_k, const int num_experts_per_node, const T* gemm1_in, const T* gemm2_in,
                                   const WeightType* weights1, const WeightType* weights2, const float* alpha_scale_flat1,
                                   const float* alpha_scale_flat2, const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat1,
                                   const TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_flat2, QuantParams quant_params,
                                   const ScaleBiasType* bias1, const ScaleBiasType* bias2, UnfusedGemmOutputType* gemm1_output,
                                   UnfusedGemmOutputType* gemm2_output, cudaStream_t stream);
  std::map<std::string, std::pair<size_t, size_t>> getWorkspaceDeviceBufferSizes(const int64_t num_rows,
                                                                                 const int64_t hidden_size, const int64_t inter_size, const int num_experts_per_node,
                                                                                 const int experts_per_token, ActivationType activation_type,
                                                                                 bool use_awq);
  void configureWsPtrs(char* ws_ptr, const int64_t num_rows, const int64_t hidden_size, const int64_t inter_size,
                       const int num_experts_per_node, const int experts_per_token, ActivationType activation_type,
                       MOEParallelismConfig parallelism_config,
                       bool use_awq);

 private:
  bool mayHaveDifferentGEMMOutputType() const {
    // We just check if its supported because we need to know when calculating workspace size
    return (
        (moe_gemm_runner_.supportsTmaWarpSpecialized() && !std::is_same_v<T, UnfusedGemmOutputType>) || use_fp8);
  }

  bool mayHaveFinalizeFused() const {
    return moe_gemm_runner_.supportsTmaWarpSpecialized() && moe_gemm_runner_.getSM() == 90 && !use_deterministic_hopper_reduce_ && !use_w4afp8 && !use_wfp4a16;
  }

  // TODO: This should eventually take the quant params to give more flexibility
  static auto getScalingType() {
    return use_wfp4afp8  ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX
           : use_fp4     ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
           : use_wfp4a16 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX
                         : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE;
  }

  const T* applyPrequantScale(void* smoothed_act, const void* permuted_data, const void* prequant_scales,
                              const int64_t* num_valid_tokens_ptr, const int64_t expanded_num_rows, const int64_t seq_len, const bool use_awq,
                              cudaStream_t stream);

  MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType> moe_gemm_runner_;

  std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config_;
  std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config_;

  // Pointers
  int* permuted_row_to_unpermuted_row_{};
  int* permuted_token_selected_experts_{};
  int* blocked_expert_counts_{};
  int* blocked_expert_counts_cumsum_{};
  int* blocked_row_to_unpermuted_row_{};
  T* permuted_data_{};
  float* permuted_token_final_scales_{};

  int64_t* expert_first_token_offset_{};

  void* glu_inter_result_{};
  void* fc2_result_{};
  T* fc1_result_{};
  // TODO If we fuse the quantization for GEMM2 into GEMM1 we will need two pointers
  TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_fp4_act_scale_;
  TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_scale_;
  const float** alpha_scale_ptr_array_fc1_ = nullptr;
  const float** alpha_scale_ptr_array_fc2_ = nullptr;
  float* moe_gemv_splitk_partials_{};
  void* smoothed_act_{};

  TmaWarpSpecializedGroupedGemmInput tma_ws_grouped_gemm1_input_;
  TmaWarpSpecializedGroupedGemmInput tma_ws_grouped_gemm2_input_;
};

struct GemmProfilerBackend {
 public:
  using Config = cutlass_extensions::CutlassGemmConfig;
  enum class GemmToProfile {
    Undefined = 0,
    GEMM_1,
    GEMM_2
  };

  void init(CutlassMoeFCRunnerInterface& runner, GemmToProfile gemm_to_profile, nvinfer::DataType dtype,
            nvinfer::DataType wtype, nvinfer::DataType otype, int num_experts, int k, int64_t hidden_size,
            int64_t inter_size, int64_t group_size, ActivationType activation_type, bool bias,
            bool need_weights, MOEParallelismConfig parallelism_config) {
    mInterface = &runner;
    mGemmToProfile = gemm_to_profile;
    mDType = dtype;
    mWType = wtype;
    mOType = otype;
    mNumExperts = num_experts;
    mNumExpertsPerNode = num_experts / parallelism_config.ep_size;
    mK = k;
    mExpertHiddenSize = hidden_size;
    mExpertInterSize = inter_size;  // Already divided by tp_size
    mGroupSize = group_size;
    mActivationType = activation_type;
    mBias = bias;
    mNeedWeights = need_weights;
    mParallelismConfig = parallelism_config;
    mSM = common::getSMVersion();

    mScalingType = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE;
    if (dtype == nvinfer::DataType::kFP8 && (wtype == nvinfer::DataType::kFP4 || wtype == nvinfer::DataType::kINT64)) {
      mScalingType = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
    } else if (dtype == nvinfer::DataType::kFP4 && (wtype == nvinfer::DataType::kFP4 || wtype == nvinfer::DataType::kINT64)) {
      mScalingType = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
    } else if ((wtype == nvinfer::DataType::kFP4 || wtype == nvinfer::DataType::kINT64)) {
      mScalingType = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
    }
  }

  void prepare(int num_tokens, char* workspace, const void* expert_weights, cudaStream_t stream);

  std::map<std::string, std::pair<size_t, size_t>> getProfilerWorkspaces(int maxM, bool is_tma_ws);
  size_t getWorkspaceSize(int maxM);

  void runProfiler(int num_tokens, const cutlass_extensions::CutlassGemmConfig& tactic, char* workspace_ptr_char, const void* expert_weights,
                   const cudaStream_t& stream);

  CutlassMoeFCRunnerInterface* mInterface;

  GemmToProfile mGemmToProfile = GemmToProfile::Undefined;
  std::vector<cutlass_extensions::CutlassGemmConfig> mAllTacticsSaved;
  int mSM{};
  int64_t mNumExperts{};
  int64_t mNumExpertsPerNode{};
  int64_t mK{};
  int64_t mExpertHiddenSize{};
  int64_t mExpertInterSize{};
  int64_t mGroupSize{};
  ActivationType mActivationType{};
  MOEParallelismConfig mParallelismConfig{};

  int mSampleIndex = 0;

  nvinfer::DataType mDType{};
  nvinfer::DataType mWType{};
  nvinfer::DataType mOType{};

  // This will be a unique value for every iteration of warmup and actual bench
  constexpr static int64_t NUM_ROUTING_SAMPLES = 16;

  std::array<TmaWarpSpecializedGroupedGemmInput, NUM_ROUTING_SAMPLES> mTmaInputCache;
  QuantParams mQuantParams;

  bool mBias{};
  bool mNeedWeights{};

  TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType mScalingType{};

 private:
  void prepareRouting(int num_tokens, char* workspace, cudaStream_t stream);
  void prepareQuantParams(int num_tokens, char* workspace, cudaStream_t stream);
  void prepareTmaWsInputs(int num_tokens, char* workspace, const void* expert_weights, cudaStream_t stream);
};

// Populates a buffer with random values for use with MOE benchmarking
void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream);

}  // namespace cutlass_kernels
}  // namespace onnxruntime::llm::kernels

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
