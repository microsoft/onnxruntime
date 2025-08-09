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
#include <array>
#include <cuda_runtime_api.h>
#include <optional>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/layout/layout.h"
#include "contrib_ops/cuda/llm/common/cuda_fp8_utils.h"
#include "contrib_ops/cuda/llm/common/workspace.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"

#include "contrib_ops/cuda/llm/moe_gemm/common.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

namespace onnxruntime::llm::kernels::cutlass_kernels {
template <class T>
constexpr auto transpose_stride(T const& t) {
  return cute::prepend(cute::prepend(cute::take<2, cute::rank_v<T>>(t), cute::get<0>(t)), cute::get<1>(t));
}

template <typename AType, typename BType, typename BScaleType, typename OType>
struct GroupedGemmInput {
  AType const* A = nullptr;
  int64_t const* total_tokens_including_expert = nullptr;
  BType const* B = nullptr;
  BScaleType const* scales = nullptr;
  BScaleType const* zeros = nullptr;
  OType const* biases = nullptr;
  OType* C = nullptr;
  float const** alpha_scales = nullptr;
  int* occupancy = nullptr;

  ActivationType activation_type = ActivationType::InvalidType;
  int64_t num_rows = 0;
  int64_t n = 0;
  int64_t k = 0;
  int num_experts = 0;
  int const groupwise_quant_group_size = 0;

  bool bias_is_broadcast = true;
  bool use_fused_moe = false;

  cudaStream_t stream = 0;
  cutlass_extensions::CutlassGemmConfig gemm_config;
};

struct TmaWarpSpecializedGroupedGemmInput {
  template <class T>
  using TransposeStride = decltype(transpose_stride<T>(T{}));
  template <class Tag>
  using TransposeLayoutTag = std::conditional_t<std::is_same_v<Tag, cutlass::layout::RowMajor>,
                                                cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;

  static_assert(std::is_same_v<cutlass::layout::RowMajor, TransposeLayoutTag<cutlass::layout::ColumnMajor>>);
  static_assert(std::is_same_v<cutlass::layout::ColumnMajor, TransposeLayoutTag<cutlass::layout::RowMajor>>);

  // Layout for A and B is transposed and then swapped in the implementation
  // This uses B^T * A^T = (A * B)^T to get a better layout for the GEMM
  using LayoutA = TransposeLayoutTag<cutlass::layout::RowMajor>;     // Layout type for A matrix operand
  using LayoutB = TransposeLayoutTag<cutlass::layout::ColumnMajor>;  // Layout type for B matrix operand
  using LayoutC = TransposeLayoutTag<cutlass::layout::RowMajor>;     // Layout type for C matrix operand

  constexpr static int NVFP4BlockScaleVectorSize = 16;
  constexpr static int MXFPXBlockScaleVectorSize = 32;

  using NVFP4BlockScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<NVFP4BlockScaleVectorSize>;
  using MXFPXBlockScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<MXFPXBlockScaleVectorSize>;

  // 128
  // This is the alignment of the weight matrix the fully padded SF will refer to.
  // We require the SFs to be aligned to this value (zero padded as needed)
  // The weights do not need to be aligned to this value, CUTLASS will handle extra padding
  // N here is a short hand for the outer dimension of the GEMM, this applies to both M & N dimension of the GEMM
  constexpr static int MinNDimAlignmentNVFP4 = cute::size<0>(NVFP4BlockScaledConfig::SfAtom{});
  constexpr static int MinNDimAlignmentMXFPX = cute::size<0>(MXFPXBlockScaledConfig::SfAtom{});

  // Block scale vector size * 4
  // This is the alignment of the weight matrix the fully padded SF will refer to.
  // We should never actually need to pad a buffer to this alignment
  // The weights only need to be aligned to BlockScaleVectorSize, CUTLASS will handle extra padding
  // The SFs only need to be aligned to 4 (zero padded as needed)
  // K here is a short hand for the inner dimension of the GEMM
  constexpr static int MinKDimAlignmentNVFP4 = cute::size<1>(NVFP4BlockScaledConfig::SfAtom{});
  constexpr static int MinKDimAlignmentMXFPX = cute::size<1>(MXFPXBlockScaledConfig::SfAtom{});

  // Helper function to align a dimension to the SF alignment
  constexpr static int64_t alignToSfDim(int64_t dim, int64_t alignment) {
    return (dim + alignment - 1) / alignment * alignment;
  }

  using StrideA = std::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutA*>>;  // Use B because they will be swapped
  using StrideB = std::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutB*>>;  // Use A because they will be swapped
  using StrideC = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutC*>>;

#ifdef ENABLE_FP8
  template <class T>
  constexpr static bool IsFP8_v = std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;
#else
  template <class T>
  constexpr static bool IsFP8_v = false;
#endif

  // Currently this should always just be T
  template <class T>
  using OutputTypeAdaptor_t = std::conditional_t<IsFP8_v<T>, nv_bfloat16, T>;

  using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int64_t, int64_t, int64_t>>;

  ProblemShape shape_info{};
  StrideA* stride_a = nullptr;
  StrideB* stride_b = nullptr;

  void const** ptr_a = nullptr;
  void const** ptr_b = nullptr;

  // C is currently the same in both epilogues
  StrideC* stride_c = nullptr;
  void const** ptr_c = nullptr;

  struct DefaultEpilogue {
    using LayoutD = TransposeLayoutTag<cutlass::layout::RowMajor>;  // Layout type for D matrix operand
    using StrideD = std::remove_pointer_t<cutlass::detail::TagToStrideC_t<LayoutD*>>;

    StrideD* stride_d = nullptr;
    void** ptr_d = nullptr;
  };

  struct FusedFinalizeEpilogue {
    using StrideFinalOutput = DefaultEpilogue::StrideD;
    using StrideBias = TransposeStride<cute::Stride<cute::_0, cute::_1, int>>;
    using StrideRouterScales = TransposeStride<cute::Stride<cute::_1, cute::_0>>;

    void* ptr_final_output = nullptr;
    StrideFinalOutput stride_final_output{};

    void const* ptr_bias = nullptr;
    StrideBias stride_bias{};

    float const* ptr_router_scales = nullptr;
    StrideRouterScales stride_router_scales{};

    int64_t const* ptr_expert_first_token_offset = nullptr;
    int const* ptr_source_token_index = nullptr;

    size_t num_rows_in_final_output = 0;
  };

  DefaultEpilogue default_epilogue;
  FusedFinalizeEpilogue fused_finalize_epilogue;

  enum class EpilogueFusion {
    NONE,
    ACTIVATION,
    GATED_ACTIVATION,
    FINALIZE
  };
  EpilogueFusion fusion = EpilogueFusion::NONE;

  float const** alpha_scale_ptr_array = nullptr;

  using ElementSF = uint8_t;
  using MXFPXElementSF = ElementSF;  // Just an alias for now
  using NVFP4ElementSF = ElementSF;  // Just an alias for now
  ElementSF const** fpX_block_scaling_factors_A = nullptr;
  ElementSF const** fpX_block_scaling_factors_B = nullptr;

  void* fpX_block_scaling_factors_stride_A = nullptr;
  void* fpX_block_scaling_factors_stride_B = nullptr;

  enum class FpXBlockScalingType {
    MXFPX,
    NVFP4,
    NONE
  };
  FpXBlockScalingType fpX_block_scaling_type = FpXBlockScalingType::NONE;

  struct INT4GroupwiseParams {
    constexpr static int group_size = 128;  // Unused, hard-coded to 128
    bool enabled = false;
    using SFA = __nv_bfloat16;
    using SFB = __nv_bfloat16;  // Unused
    using ProblemShapeInt = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
    using LayoutSFA = typename cutlass::layout::ColumnMajor;
    using LayoutSFB = typename cutlass::layout::ColumnMajor;  // Unused
    using StrideSFA = cute::Stride<cute::Int<1>, int64_t, int64_t>;
    using StrideSFB = cute::Stride<cute::Int<1>, int64_t, int64_t>;  // Unused
    StrideSFA* stride_s_a = nullptr;
    StrideSFB* stride_s_b = nullptr;  // Unused
    const SFA** ptr_s_a = nullptr;
    const SFA** ptr_z_a = nullptr;  // Unused
    const SFB** ptr_s_b = nullptr;  // Unused
    const SFB** ptr_z_b = nullptr;  // Unused
    ProblemShapeInt shape{};
  };

  INT4GroupwiseParams int4_groupwise_params;

  uint8_t* gemm_workspace = nullptr;
  size_t gemm_workspace_size = 0;

  static std::array<size_t, 17> workspaceBuffers(int num_experts, FpXBlockScalingType scaling_type);

  static size_t workspaceSize(int num_experts, FpXBlockScalingType scaling_type);

  void configureWorkspace(int8_t* start_ptr, int num_experts, void* gemm_workspace, size_t gemm_workspace_size,
                          FpXBlockScalingType scaling_type);

  bool isValid() const {
    return stride_a != nullptr && ptr_a != nullptr;
  }

  void setFinalizeFusionParams(void* final_output, float const* router_scales,
                               int64_t const* expert_first_token_offset, int const* source_token_index, void const* bias, int hidden_size,
                               int num_output_tokens);

  std::string toString() const;
};

constexpr bool isGatedActivation(ActivationType activation_type) {
  return activation_type == ActivationType::Swiglu || activation_type == ActivationType::Geglu;
}

template <typename T,                         /*The type used for activations/scales/compute*/
          typename WeightType,                /* The type for the MoE weights */
          typename OutputType,                /* The output type for the GEMM */
          typename ScaleBiasType = OutputType /* The type for the scales/bias */
          >
class MoeGemmRunner {
 public:
  MoeGemmRunner();

#if defined(ENABLE_FP8)
  static constexpr bool use_fp8 = (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) && !std::is_same_v<WeightType, cutlass::uint4b_t>
#if defined(ENABLE_FP4)
                                  && !std::is_same_v<WeightType, __nv_fp4_e2m1>
#endif
      ;
  static constexpr bool use_w4afp8 = std::is_same_v<T, __nv_fp8_e4m3> && std::is_same_v<WeightType, cutlass::uint4b_t>;
#else
  static constexpr bool use_fp8 = false;
  static constexpr bool use_w4afp8 = false;
  static constexpr bool use_wfp4afp4 = false;
#endif

#if defined(ENABLE_FP4)
  static constexpr bool use_fp4 = std::is_same_v<T, __nv_fp4_e2m1>;
  static constexpr bool use_wfp4afp4 = std::is_same_v<T, __nv_fp8_e4m3> && std::is_same_v<WeightType, __nv_fp4_e2m1>;
#else
  static constexpr bool use_fp4 = false;
  static constexpr bool use_wfp4afp4 = false;
#endif

  void moeGemmBiasAct(GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs,
                      TmaWarpSpecializedGroupedGemmInput hopper_inputs);

  void moeGemm(GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs,
               TmaWarpSpecializedGroupedGemmInput hopper_inputs);

  std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs() const;
  static std::vector<cutlass_extensions::CutlassGemmConfig> getConfigs(int sm);
  static std::vector<cutlass_extensions::CutlassGemmConfig> getTmaWarpSpecializedConfigs(int sm);
  static std::vector<cutlass_extensions::CutlassGemmConfig> getBlackwellConfigs(int sm);
  static std::vector<cutlass_extensions::CutlassGemmConfig> getHopperConfigs(int sm);
  static std::vector<cutlass_extensions::CutlassGemmConfig> getAmpereConfigs(int sm);

  [[nodiscard]] bool isTmaWarpSpecialized(cutlass_extensions::CutlassGemmConfig gemm_config) const;
  [[nodiscard]] bool supportsTmaWarpSpecialized() const;
  [[nodiscard]] bool isFusedGatedActivation(
      cutlass_extensions::CutlassGemmConfig gemm_config, bool is_gated_activation, int gemm_n, int gemm_k) const;
  [[nodiscard]] bool supportsFusedGatedActivation(bool is_gated_activation, int gemm_n, int gemm_k) const;

  size_t getMaxWorkspaceSize(int num_experts) const;

  [[nodiscard]] int getSM() const;

 private:
  template <typename EpilogueTag>
  void dispatchToArch(GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs,
                      TmaWarpSpecializedGroupedGemmInput hopper_inputs);

  template <typename EpilogueTag>
  void runGemm(GroupedGemmInput<T, WeightType, ScaleBiasType, OutputType> inputs,
               TmaWarpSpecializedGroupedGemmInput hopper_inputs);

 private:
  int sm_{};
  int multi_processor_count_{};
  mutable int num_experts_ = 0;
  mutable size_t gemm_workspace_size_ = 0;
  size_t calcMaxWorkspaceSize(int num_experts) const;
};

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
