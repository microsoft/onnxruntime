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
#include "contrib_ops/cuda/llm/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "contrib_ops/cuda/llm/nv_infer_datatype.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
#include <cuda_runtime_api.h>

#include <array>
#include <map>
#include <optional>
#include <random>
#include <utility>

namespace onnxruntime::llm::kernels {
// Change to following declarations must sync with lora.h in public repo
class LoraImpl;

int Lora_run(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
             void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace, cudaStream_t stream);

struct LoraParams {
  using LoraImplPtr = std::shared_ptr<LoraImpl>;

  int32_t const* fc1_lora_ranks = nullptr;
  void const* const* fc1_lora_weight_ptrs = nullptr;

  int32_t const* fc2_lora_ranks = nullptr;
  void const* const* fc2_lora_weight_ptrs = nullptr;

  int32_t const* gated_lora_ranks = nullptr;
  void const* const* gated_lora_weight_ptrs = nullptr;

  // used to calculate split group gemm workspace
  int num_reqs;

  // fc1 and gated use the same impl
  LoraImplPtr fc1_lora_impl;
  LoraImplPtr fc2_lora_impl;

  void* workspace;

  cudaEvent_t* memcpy_event_ptr;

  LoraParams() = default;

  LoraParams(int num_reqs, int32_t const* fc1_lora_ranks, void const* const* fc1_lora_weight_ptrs,
             int32_t const* fc2_lora_ranks, void const* const* fc2_lora_weight_ptrs, LoraImplPtr fc1_lora_impl,
             LoraImplPtr fc2_lora_impl, void* workspace, cudaEvent_t* memcpy_event_ptr,
             int32_t const* gated_lora_ranks = nullptr, void const* const* gated_lora_weight_ptrs = nullptr)
      : fc1_lora_ranks(fc1_lora_ranks), fc1_lora_weight_ptrs(fc1_lora_weight_ptrs), fc2_lora_ranks(fc2_lora_ranks), fc2_lora_weight_ptrs(fc2_lora_weight_ptrs), gated_lora_ranks(gated_lora_ranks), gated_lora_weight_ptrs(gated_lora_weight_ptrs), num_reqs(num_reqs), fc1_lora_impl(fc1_lora_impl), fc2_lora_impl(fc2_lora_impl), workspace(workspace), memcpy_event_ptr(memcpy_event_ptr) {
  }
};

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

  bool operator==(MOEParallelismConfig const& other) const {
    return tp_size == other.tp_size && tp_rank == other.tp_rank && ep_size == other.ep_size && ep_rank == other.ep_rank && cluster_size == other.cluster_size && cluster_rank == other.cluster_rank;
  }

  friend std::ostream& operator<<(std::ostream& os, MOEParallelismConfig const& config) {
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
    void const* fc1_weight_scales = nullptr;
    void const* fc2_weight_scales = nullptr;
  } wo;

  // FP8 quantization params
  struct
  {
    bool fc2_use_per_expert_act_scale = false;
    float const* dequant_fc1 = nullptr;    // (num_experts_per_node, )
    float const* quant_fc2 = nullptr;      // (1, ) or (num_experts_per_node, ) based on fc2_use_per_expert_act_scale
    float const* dequant_fc2 = nullptr;    // (num_experts_per_node, )
    float const* quant_final = nullptr;    // (1, )
    float const* dequant_input = nullptr;  // (1, )
  } fp8;

  // FP8 MXFP4 quantization params
  // This mode uses regular global scale for FP8 activations and block scaling for MXFP4 weights
  struct FP8MXFP4Inputs {
    struct GemmInputs {
      bool use_per_expert_act_scale = false;
      float const* act_global_scale = nullptr;                                                 // (1, ) or (num_experts_per_node, ) based on use_per_expert_act_scale
      TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const* weight_block_scale = nullptr;  // (experts, n, k / 32)
      float const* global_scale = nullptr;                                                     // (num_experts_per_node, )
    };

    GemmInputs fc1;
    GemmInputs fc2;
  } fp8_mxfp4;

  // MXFP8 MXFP4 quantization params
  // This mode uses block scaled MXFP8 and MXFP4 weights
  struct MXFP8MXFP4Inputs {
    struct GemmInputs {
      TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const* weight_block_scale = nullptr;  // (experts, n, k / 32)
      float const* global_scale = nullptr;                                                     // (num_experts_per_node, )
    };

    GemmInputs fc1;
    GemmInputs fc2;
  } mxfp8_mxfp4;

  // FP4 quantization params
  struct FP4Inputs {
    struct GemmInputs {
      bool use_per_expert_act_scale = false;

      float const* act_global_scale = nullptr;                                                 // (1, ) or (num_experts_per_node, ) based on use_per_expert_act_scale
      TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const* weight_block_scale = nullptr;  // (experts, n, k / 16)
      float const* global_scale = nullptr;                                                     // (num_experts_per_node, )
    };

    GemmInputs fc1;
    GemmInputs fc2;
  } fp4;

  // GPTQ/AWQ quantization params
  struct GroupwiseInputs {
    struct GroupwiseGemmInputs {
      void const* act_scales = nullptr;
      void const* weight_scales = nullptr;
      void const* weight_zeros = nullptr;
      float const* alpha = nullptr;
    };

    int group_size = -1;
    GroupwiseGemmInputs fc1;
    GroupwiseGemmInputs fc2;
  } groupwise;

  // FP8 blockscaling params (for Deepseek)
  struct BlockScaleParams {
    float const* fc1_scales_ptrs = nullptr;
    float const* fc2_scales_ptrs = nullptr;

    BlockScaleParams() = default;

    BlockScaleParams(float const* fc1_scales_ptrs, float const* fc2_scales_ptrs)
        : fc1_scales_ptrs(fc1_scales_ptrs), fc2_scales_ptrs(fc2_scales_ptrs) {
    }
  } fp8_block_scaling;

  static QuantParams Int(void const* fc1_weight_scales, void const* fc2_weight_scales) {
    QuantParams qp;
    qp.wo = {fc1_weight_scales, fc2_weight_scales};
    return qp;
  }

  static QuantParams FP8(float const* dequant_fc1, float const* quant_fc2, float const* dequant_fc2,
                         float const* quant_final = nullptr, float const* dequant_input = nullptr,
                         bool fc2_use_per_expert_act_scale = false) {
    QuantParams qp;
    qp.fp8 = {fc2_use_per_expert_act_scale, dequant_fc1, quant_fc2, dequant_fc2, quant_final, dequant_input};
    return qp;
  }

  static QuantParams FP8MXFP4(float const* fc1_act_global_scale,
                              TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const* fc1_weight_block_scale,
                              float const* fc1_global_scale,  //
                              float const* fc2_act_global_scale,
                              TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const* fc2_weight_block_scale,
                              float const* fc2_global_scale,  //
                              bool fc1_use_per_expert_act_scale = false, bool fc2_use_per_expert_act_scale = false) {
    QuantParams qp;
    qp.fp8_mxfp4.fc1 = {fc1_use_per_expert_act_scale, fc1_act_global_scale, fc1_weight_block_scale, fc1_global_scale};
    qp.fp8_mxfp4.fc2 = {fc2_use_per_expert_act_scale, fc2_act_global_scale, fc2_weight_block_scale, fc2_global_scale};
    return qp;
  }

  static QuantParams MXFP8MXFP4(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const* fc1_weight_block_scale,
                                float const* fc1_global_scale,  //
                                TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const* fc2_weight_block_scale, float const* fc2_global_scale) {
    QuantParams qp;
    qp.mxfp8_mxfp4.fc1 = {fc1_weight_block_scale, fc1_global_scale};
    qp.mxfp8_mxfp4.fc2 = {fc2_weight_block_scale, fc2_global_scale};
    return qp;
  }

  static QuantParams FP4(float const* fc1_act_global_scale,
                         TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const* fc1_weight_block_scale,
                         float const* fc1_global_scale,  //
                         float const* fc2_act_global_scale,
                         TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const* fc2_weight_block_scale,
                         float const* fc2_global_scale,  //
                         bool fc1_use_per_expert_act_scale = false, bool fc2_use_per_expert_act_scale = false)

  {
    QuantParams qp;
    qp.fp4.fc1 = {fc1_use_per_expert_act_scale, fc1_act_global_scale, fc1_weight_block_scale, fc1_global_scale};
    qp.fp4.fc2 = {fc2_use_per_expert_act_scale, fc2_act_global_scale, fc2_weight_block_scale, fc2_global_scale};
    return qp;
  }

  static QuantParams GroupWise(int group_size, void const* fc1_weight_scales, void const* fc2_weight_scales,
                               void const* fc1_activation_scales = nullptr, void const* fc2_activation_scales = nullptr,
                               void const* fc1_weight_zeros = nullptr, void const* fc2_weight_zeros = nullptr,
                               float const* fc1_alpha = nullptr, float const* fc2_alpha = nullptr) {
    QuantParams qp;
    qp.groupwise.group_size = group_size;
    qp.groupwise.fc1 = {fc1_activation_scales, fc1_weight_scales, fc1_weight_zeros, fc1_alpha};
    qp.groupwise.fc2 = {fc2_activation_scales, fc2_weight_scales, fc2_weight_zeros, fc2_alpha};
    return qp;
  }

  static QuantParams FP8BlockScaling(float const* fc1_scales, float const* fc2_scales) {
    QuantParams qp;
    qp.fp8_block_scaling = {fc1_scales, fc2_scales};
    return qp;
  }
};

struct MoeMinLatencyParams {
  // All these are allocated on device memory
  // Number of active experts on current node; smaller than or equal to num_experts_per_node
  int* num_active_experts_per_node;
  // The score of each token for each activated expert. 0 if the expert is not chosen by the token.
  // Only the first num_active_experts_per_ rows are valid
  float* experts_to_token_score;
  // The global expert id for each activated expert
  // Only the first num_active_experts_per_ values are valid
  int* active_expert_global_ids;

  MoeMinLatencyParams()
      : num_active_experts_per_node(nullptr), experts_to_token_score(nullptr), active_expert_global_ids(nullptr) {
  }

  MoeMinLatencyParams(int* num_active_experts_per_node, float* experts_to_token_score, int* active_expert_global_ids)
      : num_active_experts_per_node(num_active_experts_per_node), experts_to_token_score(experts_to_token_score), active_expert_global_ids(active_expert_global_ids) {
  }
};

class CutlassMoeFCRunnerInterface {
 public:
  virtual ~CutlassMoeFCRunnerInterface() = default;
  virtual size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                                  int const num_experts, int const experts_per_token, ActivationType activation_type,
                                  MOEParallelismConfig parallelism_config, bool use_lora, bool use_deepseek_fp8_block_scale,
                                  bool min_latency_mode, bool use_awq) = 0;
  virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                         std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) = 0;
  virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

  virtual void runMoe(void const* input_activations, void const* input_sf, int const* token_selected_experts,
                      float const* token_final_scales, void const* fc1_expert_weights, void const* fc1_expert_biases,
                      ActivationType fc1_activation_type, void const* fc2_expert_weights, void const* fc2_expert_biases,
                      QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                      int const num_experts, int const experts_per_token, char* workspace_ptr, void* final_output,
                      int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
                      bool use_lora, LoraParams& lora_params, bool use_deepseek_fp8_block_scale, bool min_latency_mode,
                      MoeMinLatencyParams& min_latency_params, cudaStream_t stream) = 0;

  // Aliases for profiling the gemms
  virtual void gemm1(void const* const input, void* const output, void* const intermediate_result,
                     int64_t const* const expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
                     void const* const fc1_expert_weights, void const* const fc1_expert_biases,
                     int64_t const* const num_valid_tokens_ptr, void const* const fc1_int_scales, float const* const fc1_fp8_dequant,
                     float const* const fc2_fp8_quant, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
                     TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params,
                     int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
                     int const num_experts_per_node, ActivationType fc1_activation_type, float const** alpha_scale_ptr_array,
                     bool bias_is_broadcast, bool use_deepseek_fp8_block_scale, cudaStream_t stream,
                     cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode, int* num_active_experts_per,
                     int* active_expert_global_ids) = 0;

  virtual void gemm2(void const* const input, void* const gemm_output, void* const final_output,
                     int64_t const* const expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
                     void const* const fc2_expert_weights, void const* const fc2_expert_biases, void const* const fc2_int_scales,
                     float const* const fc2_fp8_dequant, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
                     QuantParams quant_params, float const* const token_topk_unpermuted_scales,
                     float const* const token_topk_permuted_scales, int const* const unpermuted_row_to_permuted_row,
                     int const* permuted_row_to_unpermuted_row, int const* const token_selected_experts,
                     int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
                     int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
                     int64_t const experts_per_token, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
                     bool use_deepseek_fp8_block_scale, cudaStream_t stream, MOEParallelismConfig parallelism_config,
                     bool const enable_alltoall, cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
                     int* num_active_experts_per, int* active_expert_global_ids) = 0;

  virtual std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecializedDispatch(int64_t const* expert_first_token_offset,
                                           TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                           int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
                                           int64_t gemm2_k, int const num_experts_per_node, void const* gemm1_in, void const* gemm2_in,
                                           void const* weights1, void const* weights2, float const* alpha_scale_flat1, float const* alpha_scale_flat2,
                                           TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
                                           TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params, void const* bias1,
                                           void const* bias2, void* gemm1_output, void* gemm2_output, cudaStream_t stream) = 0;

  virtual std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecializedLowLatencyDispatch(TmaWarpSpecializedGroupedGemmInput layout_info1,
                                                     TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
                                                     int64_t gemm2_n, int64_t gemm2_k, int const num_experts, void const* input1, void const* input2,
                                                     void const* weights1, void const* weights2, float const* fp8_dequant1, float const* fp8_dequant2,
                                                     TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
                                                     TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat, QuantParams quant_params,
                                                     void const* bias1, void const* bias2, void* output1, void* output2, int const* num_active_experts_per,
                                                     int const* active_expert_global_ids, int start_expert, cudaStream_t stream) = 0;

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
  using DeepSeekBlockScaleGemmRunner = onnxruntime::llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface;
  using ScaleBiasType = BackBoneType;
  using Self = CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType>;
#if defined(ENABLE_FP8)
  static constexpr bool use_fp8 = (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) && !std::is_same_v<WeightType, cutlass::uint4b_t>;
  static constexpr bool use_w4afp8 = std::is_same_v<WeightType, cutlass::uint4b_t> && std::is_same_v<T, __nv_fp8_e4m3>;
  static_assert(!std::is_same_v<BackBoneType, __nv_fp8_e4m3>, "Current logic requires backbone type to be >=16-bits");
  static_assert(!std::is_same_v<OutputType, __nv_fp8_e4m3>, "Current logic requires output type to be >=16-bits");
#else
  static constexpr bool use_fp8 = false;
  static constexpr bool use_w4afp8 = false;
#endif
#if defined(ENABLE_FP4)
  static constexpr bool act_fp4 = std::is_same_v<T, __nv_fp4_e2m1>;
  static constexpr bool weight_fp4 = std::is_same_v<WeightType, __nv_fp4_e2m1>;
  static constexpr bool use_wfp4afp8 = std::is_same_v<T, __nv_fp8_e4m3> && weight_fp4;
  static constexpr bool use_fp4 = act_fp4 && weight_fp4;
  static_assert(!std::is_same_v<BackBoneType, __nv_fp4_e2m1>, "Current logic requires backbone type to be >=16-bits");
  static_assert(!std::is_same_v<OutputType, __nv_fp4_e2m1>, "Current logic requires output type to be >=16-bits");
#else
  static constexpr bool act_fp4 = false;
  static constexpr bool weight_fp4 = false;
  static constexpr bool use_wfp4afp8 = false;
  static constexpr bool use_fp4 = false;
#endif

  // Added by ORT

  ActivationType activation_type_;
  bool has_fc3_;
  bool normalize_routing_weights_;
  bool use_sparse_mixer_;

  static constexpr bool use_block_scaling = use_fp4 || use_wfp4afp8;

  // This should leave the variable unchanged in any currently supported configuration
  using UnfusedGemmOutputType = BackBoneType;

  // We introduce this as a separate parameter, so that if we ever remove the above condition we can decouple
  // BackBoneType and OutputType easily. For now these are required to be equivalent
  static_assert(std::is_same_v<OutputType, BackBoneType>, "Scale and bias types must match OutputType");

 public:
  CutlassMoeFCRunner(int sm_version, ActivationType activation_type, bool has_fc3, bool normalize_routing_weights, bool use_sparse_mixer);

  ~CutlassMoeFCRunner() override = default;

  static_assert(
      std::is_same_v<T, WeightType> || !std::is_same_v<T, float>, "Does not support float with quantized weights");

  size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const fc1_output_size,
                          int const num_experts, int const experts_per_token, ActivationType activation_type,
                          MOEParallelismConfig parallelism_config, bool use_lora, bool use_deepseek_fp8_block_scale,
                          bool min_latency_mode, bool use_awq) override;

  void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                 std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) override {
    gemm1_config_ = std::move(gemm1_config);
    gemm2_config_ = std::move(gemm2_config);
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override {
    return moe_gemm_runner_.getConfigs();
  }

  static std::vector<cutlass_extensions::CutlassGemmConfig> getTactics(int sm) {
    using RunnerType = decltype(moe_gemm_runner_);
    return RunnerType::getConfigs(sm);
  }

  void runMoe(void const* input_activations, void const* input_sf, int const* token_selected_experts,
              float const* token_final_scales, void const* fc1_expert_weights, void const* fc1_expert_biases,
              ActivationType fc1_activation_type, void const* fc2_expert_weights, void const* fc2_expert_biases,
              QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
              int const num_experts, int const experts_per_token, char* workspace_ptr, void* final_output,
              int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
              bool use_lora, LoraParams& lora_params, bool use_deepseek_fp8_block_scale, bool min_latency_mode,
              MoeMinLatencyParams& min_latency_params, cudaStream_t stream) override;

  // We make these GEMM1 & GEMM2 static because they need to be stateless for the profiler to work
  static void gemm1(MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
                    // This argument must not be null if fp8 block scaling is being used.
                    // The gemm_runner will be ignored in that case. NOTE: it would
                    // be great if we could consolidate gemm_runner and fp8_blockscale_gemm_runner.
                    // For now, they don't share the same interface, so we just use two separate
                    // arguments.
                    DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input, T* const output,
                    void* const intermediate_result, int64_t const* const expert_first_token_offset,
                    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template, WeightType const* const fc1_expert_weights,
                    ScaleBiasType const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
                    ScaleBiasType const* const fc1_int_scales, float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
                    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
                    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params,
                    int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
                    int const num_experts_per_node, ActivationType fc1_activation_type, float const** alpha_scale_ptr_array,
                    bool bias_is_broadcast, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config,
                    bool min_latency_mode, int* num_active_experts_per, int* active_expert_global_ids);

  static void gemm2(MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner,
                    DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input, void* const gemm_output,
                    OutputType* const final_output, int64_t const* const expert_first_token_offset,
                    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template, WeightType const* const fc2_expert_weights,
                    ScaleBiasType const* const fc2_expert_biases, ScaleBiasType const* const fc2_int_scales,
                    float const* const fc2_fp8_dequant, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
                    QuantParams quant_params, float const* const token_topk_unpermuted_scales,
                    float const* const token_topk_permuted_scales, int const* const unpermuted_row_to_permuted_row,
                    int const* permuted_row_to_unpermuted_row, int const* const token_selected_experts,
                    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
                    int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
                    int64_t const experts_per_token, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
                    cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
                    cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode, int* num_active_experts_per,
                    int* active_expert_global_ids);

  // Overrides to allow us to forward on to the internal functions with the pointers using the correct type
  void gemm1(void const* const input, void* const output, void* const intermediate_result,
             int64_t const* const expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
             void const* const fc1_expert_weights, void const* const fc1_expert_biases,
             int64_t const* const num_valid_tokens_ptr, void const* const fc1_int_scales, float const* const fc1_fp8_dequant,
             float const* const fc2_fp8_quant, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
             TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params,
             int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
             int const num_experts_per_node, ActivationType fc1_activation_type, float const** alpha_scale_ptr_array,
             bool bias_is_broadcast, bool use_deepseek_fp8_block_scale, cudaStream_t stream,
             cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode, int* num_active_experts_per,
             int* active_expert_global_ids) override {
    auto* block_scale_gemm_runner = use_deepseek_fp8_block_scale ? getDeepSeekBlockScaleGemmRunner() : nullptr;
    return Self::gemm1(moe_gemm_runner_, block_scale_gemm_runner, static_cast<T const*>(input),
                       static_cast<T*>(output), intermediate_result, expert_first_token_offset, tma_ws_input_template,
                       static_cast<WeightType const*>(fc1_expert_weights), static_cast<ScaleBiasType const*>(fc1_expert_biases),
                       num_valid_tokens_ptr, static_cast<ScaleBiasType const*>(fc1_int_scales), fc1_fp8_dequant, fc2_fp8_quant,
                       fc1_fp4_act_flat, fc2_fp4_act_flat, quant_params, num_rows, expanded_num_rows, hidden_size, inter_size,
                       num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array, bias_is_broadcast, stream, config,
                       min_latency_mode, num_active_experts_per, active_expert_global_ids);
  }

  void gemm2(void const* const input, void* const gemm_output, void* const final_output,
             int64_t const* const expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
             void const* const fc2_expert_weights, void const* const fc2_expert_biases, void const* const fc2_int_scales,
             float const* const fc2_fp8_dequant, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
             QuantParams quant_params, float const* const token_topk_unpermuted_scales,
             float const* const token_topk_permuted_scales, int const* const unpermuted_row_to_permuted_row,
             int const* permuted_row_to_unpermuted_row, int const* const token_selected_experts,
             int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
             int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
             int64_t const experts_per_token, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
             bool use_deepseek_fp8_block_scale, cudaStream_t stream, MOEParallelismConfig parallelism_config,
             bool const enable_alltoall, cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
             int* num_active_experts_per, int* active_expert_global_ids) override {
    auto* block_scale_gemm_runner = use_deepseek_fp8_block_scale ? getDeepSeekBlockScaleGemmRunner() : nullptr;
    return Self::gemm2(moe_gemm_runner_, block_scale_gemm_runner, static_cast<T const*>(input), gemm_output,
                       static_cast<OutputType*>(final_output), expert_first_token_offset, tma_ws_input_template,
                       static_cast<WeightType const*>(fc2_expert_weights), static_cast<ScaleBiasType const*>(fc2_expert_biases),
                       static_cast<ScaleBiasType const*>(fc2_int_scales), fc2_fp8_dequant, fc2_fp4_act_flat, quant_params,
                       token_topk_unpermuted_scales, token_topk_permuted_scales, unpermuted_row_to_permuted_row,
                       permuted_row_to_unpermuted_row, token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
                       hidden_size, inter_size, num_experts_per_node, experts_per_token, alpha_scale_ptr_array, use_lora, fc2_lora,
                       stream, parallelism_config, enable_alltoall, config, min_latency_mode, num_active_experts_per,
                       active_expert_global_ids);
  }

  virtual size_t getGemmWorkspaceSize(int num_experts_per_node) const override {
    return moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);
  }

  std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecializedDispatch(int64_t const* expert_first_token_offset,
                                           TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                           int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
                                           int64_t gemm2_k, int const num_experts_per_node, void const* gemm1_in, void const* gemm2_in,
                                           void const* weights1, void const* weights2, float const* alpha_scale_flat1, float const* alpha_scale_flat2,
                                           TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
                                           TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params, void const* bias1,
                                           void const* bias2, void* gemm1_output, void* gemm2_output, cudaStream_t stream) override {
    return Self::computeStridesTmaWarpSpecialized(expert_first_token_offset, layout_info1, layout_info2, num_tokens,
                                                  expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts_per_node,
                                                  reinterpret_cast<T const*>(gemm1_in), reinterpret_cast<T const*>(gemm2_in),
                                                  reinterpret_cast<WeightType const*>(weights1), reinterpret_cast<WeightType const*>(weights2),
                                                  alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2, quant_params,
                                                  reinterpret_cast<ScaleBiasType const*>(bias1), reinterpret_cast<ScaleBiasType const*>(bias2),
                                                  reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
                                                  reinterpret_cast<UnfusedGemmOutputType*>(gemm2_output), stream);
  }

  std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecializedLowLatencyDispatch(TmaWarpSpecializedGroupedGemmInput layout_info1,
                                                     TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
                                                     int64_t gemm2_n, int64_t gemm2_k, int const num_experts, void const* input1, void const* input2,
                                                     void const* weights1, void const* weights2, float const* fp8_dequant1, float const* fp8_dequant2,
                                                     TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
                                                     TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat, QuantParams quant_params,
                                                     void const* bias1, void const* bias2, void* output1, void* output2, int const* num_active_experts_per,
                                                     int const* active_expert_global_ids, int start_expert, cudaStream_t stream) override {
    return Self::computeStridesTmaWarpSpecializedLowLatency(layout_info1, layout_info2, num_tokens, gemm1_n,
                                                            gemm1_k, gemm2_n, gemm2_k, num_experts, reinterpret_cast<T const*>(input1),
                                                            reinterpret_cast<T const*>(input2), reinterpret_cast<WeightType const*>(weights1),
                                                            reinterpret_cast<WeightType const*>(weights2), fp8_dequant1, fp8_dequant2, fc1_fp4_act_flat,
                                                            fc2_fp4_act_flat, quant_params, reinterpret_cast<ScaleBiasType const*>(bias1),
                                                            reinterpret_cast<ScaleBiasType const*>(bias2), reinterpret_cast<UnfusedGemmOutputType*>(output1),
                                                            reinterpret_cast<UnfusedGemmOutputType*>(output2), num_active_experts_per, active_expert_global_ids,
                                                            start_expert, stream);
  }

 private:
  std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput> setupTmaWarpSpecializedInputs(
      int64_t num_rows, int64_t expanded_num_rows, ActivationType fc1_activation_type, int64_t hidden_size,
      int64_t inter_size, int64_t num_experts_per_node, void const* input_activations_void,
      TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, void* final_output,
      WeightType const* fc1_expert_weights, WeightType const* fc2_expert_weights, QuantParams quant_params,
      ScaleBiasType const* fc1_expert_biases, ScaleBiasType const* fc2_expert_biases, bool min_latency_mode,
      MoeMinLatencyParams& min_latency_params, bool use_lora, int start_expert,
      MOEParallelismConfig parallelism_config, cudaStream_t stream);

  static std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecialized(int64_t const* expert_first_token_offset,
                                   TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
                                   int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
                                   int64_t gemm2_k, int const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
                                   WeightType const* weights1, WeightType const* weights2, float const* alpha_scale_flat1,
                                   float const* alpha_scale_flat2, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
                                   TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
                                   ScaleBiasType const* bias1, ScaleBiasType const* bias2, UnfusedGemmOutputType* gemm1_output,
                                   UnfusedGemmOutputType* gemm2_output, cudaStream_t stream);
  static std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
  computeStridesTmaWarpSpecializedLowLatency(TmaWarpSpecializedGroupedGemmInput layout_info1,
                                             TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
                                             int64_t gemm2_n, int64_t gemm2_k, int const num_experts, T const* input1, T const* input2,
                                             WeightType const* weights1, WeightType const* weights2, float const* fp8_dequant1, float const* fp8_dequant2,
                                             TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
                                             TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat, QuantParams quant_params,
                                             ScaleBiasType const* bias1, ScaleBiasType const* bias2, UnfusedGemmOutputType* output1,
                                             UnfusedGemmOutputType* output2, int const* num_active_experts_per, int const* active_expert_global_ids,
                                             int start_expert, cudaStream_t stream);
  std::map<std::string, std::pair<size_t, size_t>> getWorkspaceDeviceBufferSizes(int64_t const num_rows,
                                                                                 int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
                                                                                 int const experts_per_token, ActivationType activation_type, bool use_lora, bool use_deepseek_fp8_block_scale,
                                                                                 bool min_latency_mode, bool use_awq);
  void configureWsPtrs(char* ws_ptr, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                       int const num_experts_per_node, int const experts_per_token, ActivationType activation_type,
                       MOEParallelismConfig parallelism_config, bool use_lora, bool use_deepseek_fp8_block_scale,
                       bool min_latency_mode, bool use_awq);

 private:
  bool mayHaveDifferentGEMMOutputType() const {
    // We just check if its supported because we need to know when calculating workspace size
    return (
        (moe_gemm_runner_.supportsTmaWarpSpecialized() && !std::is_same_v<T, UnfusedGemmOutputType>) || use_fp8);
  }

  bool mayHaveFinalizeFused() const {
    return moe_gemm_runner_.supportsTmaWarpSpecialized() && moe_gemm_runner_.getSM() == 90 && !use_deterministic_hopper_reduce_ && !use_w4afp8;
  }

  // TODO: This should eventually take the quant params to give more flexibility
  static auto getScalingType() {
    return use_wfp4afp8 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX
           : use_fp4    ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                        : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE;
  }

  bool setupLoraWorkspace(int64_t expanded_num_rows, int64_t num_rows, int64_t inter_size, int64_t hidden_size,
                          int start_expert, bool is_gated_activation, int num_experts_per_node, bool needs_num_valid,
                          LoraParams& lora_params, cudaStream_t stream);

  ScaleBiasType const* loraFC1(int64_t expanded_num_rows, int64_t inter_size, int64_t hidden_size,
                               int num_experts_per_node, int start_expert, int64_t const* num_valid_tokens_ptr, bool is_gated_activation,
                               ScaleBiasType const* fc1_expert_biases, LoraParams& lora_params, float const* input_fp8_dequant,
                               cudaStream_t stream);

  void loraFC2(int64_t inter_size, int64_t hidden_size, int num_experts_per_node, int start_expert,
               int64_t const* num_valid_tokens_ptr, int64_t num_tokens, LoraParams& lora_params, float const* fc2_fp8_quant,
               cudaStream_t stream);

  DeepSeekBlockScaleGemmRunner* getDeepSeekBlockScaleGemmRunner() const;

  static void BlockScaleFC1(DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input, T* const output,
                            void* const intermediate_result, int64_t const* const expert_first_token_offset,
                            WeightType const* const fc1_expert_weights, ScaleBiasType const* const fc1_expert_biases,
                            float const* const fc2_fp8_quant, int64_t const num_rows, int64_t const expanded_num_rows,
                            int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
                            ActivationType fc1_activation_type, QuantParams& quant_params, cudaStream_t stream);

  static void BlockScaleFC2(DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input, void* const gemm_output,
                            OutputType* const final_output, int64_t const* const expert_first_token_offset,
                            WeightType const* const fc2_expert_weights, ScaleBiasType const* const fc2_expert_biases,
                            float const* const token_topk_unpermuted_scales, int const* const unpermuted_row_to_permuted_row,
                            int const* const permuted_row_to_unpermuted_row, int const* const token_selected_experts,
                            int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
                            int64_t const hidden_size, int64_t const inter_size, int64_t const num_experts_per_node, int64_t const k,
                            MOEParallelismConfig parallelism_config, bool const enable_alltoall, QuantParams& quant_params,
                            cudaStream_t stream);

  T const* applyPrequantScale(void* smoothed_act, void const* permuted_data, void const* prequant_scales,
                              int64_t const* num_valid_tokens_ptr, int64_t const expanded_num_rows, int64_t const seq_len, bool const use_awq,
                              cudaStream_t stream);

  MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType> moe_gemm_runner_;
  std::unique_ptr<DeepSeekBlockScaleGemmRunner> blockscale_gemm_runner_;

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
  float const** alpha_scale_ptr_array_fc1_ = nullptr;
  float const** alpha_scale_ptr_array_fc2_ = nullptr;
  ScaleBiasType* lora_input_{};
  ScaleBiasType* lora_fc1_result_{};
  ScaleBiasType* lora_add_bias_{};
  ScaleBiasType* lora_fc2_result_{};
  void* smoothed_act_{};

  TmaWarpSpecializedGroupedGemmInput tma_ws_grouped_gemm1_input_;
  TmaWarpSpecializedGroupedGemmInput tma_ws_grouped_gemm2_input_;

  struct HostLoraWorkspace {
    std::vector<int> host_permuted_rows;
    std::vector<void const*> host_permuted_fc1_weight_ptrs;
    std::vector<void const*> host_permuted_fc2_weight_ptrs;
    std::vector<void const*> host_permuted_gated_weight_ptrs;
    std::vector<int32_t> host_permuted_fc1_lora_ranks;
    std::vector<int32_t> host_permuted_fc2_lora_ranks;
    std::vector<int32_t> host_permuted_gated_lora_ranks;
    std::vector<int64_t> host_expert_first_token_offset;
  };

  HostLoraWorkspace host_lora_workspace_;
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
            int64_t inter_size, int64_t group_size, ActivationType activation_type, bool bias, bool use_lora,
            bool min_latency_mode, bool need_weights, MOEParallelismConfig parallelism_config, bool const enable_alltoall) {
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
    mUseLora = false;
    mMinLatencyMode = min_latency_mode;
    mNeedWeights = need_weights;
    mParallelismConfig = parallelism_config;
    mEnableAlltoall = enable_alltoall;
    mSM = common::getSMVersion();

    mScalingType = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE;
    if (dtype == nvinfer::DataType::kFP8 && (wtype == nvinfer::DataType::kFP4 || wtype == nvinfer::DataType::kINT64)) {
      mScalingType = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
    } else if ((dtype == nvinfer::DataType::kFP4 || dtype == nvinfer::DataType::kINT64) && (wtype == nvinfer::DataType::kFP4 || wtype == nvinfer::DataType::kINT64)) {
      mScalingType = TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
    }
  }

  void prepare(int num_tokens, char* workspace, void const* expert_weights, cudaStream_t stream);

  std::map<std::string, std::pair<size_t, size_t>> getProfilerWorkspaces(int maxM, bool is_tma_ws);
  size_t getWorkspaceSize(int maxM);

  void runProfiler(int num_tokens, Config const& tactic, char* workspace_ptr_char, void const* expert_weights,
                   cudaStream_t const& stream);

  CutlassMoeFCRunnerInterface* mInterface;

  GemmToProfile mGemmToProfile = GemmToProfile::Undefined;
  std::vector<Config> mAllTacticsSaved;
  int mSM{};
  int64_t mNumExperts{};
  int64_t mNumExpertsPerNode{};
  int64_t mK{};
  int64_t mExpertHiddenSize{};
  int64_t mExpertInterSize{};
  int64_t mGroupSize{};
  ActivationType mActivationType{};
  MOEParallelismConfig mParallelismConfig{};
  bool mEnableAlltoall = false;

  int mSampleIndex = 0;

  nvinfer::DataType mDType{};
  nvinfer::DataType mWType{};
  nvinfer::DataType mOType{};

  // This will be a unique value for every iteration of warmup and actual bench
  constexpr static int64_t NUM_ROUTING_SAMPLES = 16;

  std::array<TmaWarpSpecializedGroupedGemmInput, NUM_ROUTING_SAMPLES> mTmaInputCache;
  QuantParams mQuantParams;

  bool mBias{};
  bool mUseLora{};
  bool mMinLatencyMode{};
  bool mNeedWeights{};

  TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType mScalingType{};

 private:
  void prepareRouting(int num_tokens, char* workspace, cudaStream_t stream);
  void prepareQuantParams(int num_tokens, char* workspace, cudaStream_t stream);
  void prepareTmaWsInputs(int num_tokens, char* workspace, void const* expert_weights, cudaStream_t stream);
};

// Populates a buffer with random values for use with MOE benchmarking
void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream);

}  // namespace cutlass_kernels
}  // namespace onnxruntime::llm::kernels

#ifdef __GNUC__ 
#pragma GCC diagnostic pop
#endif