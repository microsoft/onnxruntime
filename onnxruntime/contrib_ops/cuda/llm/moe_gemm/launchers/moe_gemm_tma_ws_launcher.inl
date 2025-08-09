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
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

#include "contrib_ops/cuda/llm/cutlass_extensions/compute_occupancy.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue/collective/epilogue_moe_finalize.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/threadblock/default_mma.h"

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/cutlass_heuristic.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_tma_warp_specialized_traits.h"
#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_launcher.h"

#include <cuda.h>
#include <cuda_fp16.h>
// #include <cutlass/arch/arch.h>
#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>
#include <math.h>
#include <sstream>

namespace onnxruntime::llm
{
namespace kernels
{
namespace cutlass_kernels
{
using EpilogueFusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion;

// Constructs an object with specific arguments only if flag is true
// This forces the if constexpr branch to properly pruned be when called from in non-template functions
template <bool FLAG, class ReturnType, class... Args>
ReturnType construct_if_true(Args&&... args)
{
    if constexpr (FLAG)
    {
        return ReturnType{std::forward<Args>(args)...};
    }
    else
    {
        return ReturnType{};
    }
}

template <bool FLAG, class GemmGrouped, bool A>
auto deduce_layout_sf()
{
    if constexpr (FLAG && A)
    {
        return typename GemmGrouped::GemmKernel::CollectiveMainloop::LayoutSFA{};
    }
    else if constexpr (FLAG && !A)
    {
        return typename GemmGrouped::GemmKernel::CollectiveMainloop::LayoutSFB{};
    }
    else
    {
        return (void*) nullptr;
    }
}

template <typename ArchTag, typename T, typename WeightType, typename OutputType, typename EpilogueTag,
    EpilogueFusion FUSION, typename TileShape, typename ClusterShape, bool IsMXFPX, bool BIAS>
struct DispatchToTmaWSFunction
{
};

// TMA WS specialized version
template <typename ArchTag, typename T, typename WeightType, typename OutputType, typename EpilogueTag,
    EpilogueFusion FUSION, typename TileShape, typename ClusterShape, bool IsMXFPX, bool BIAS>
void tma_warp_specialized_generic_moe_gemm_kernelLauncher(TmaWarpSpecializedGroupedGemmInput tma_ws_input,
    int num_experts, int const multi_processor_count, cudaStream_t stream, int* kernel_occupancy,
    size_t* workspace_size)
{
    if constexpr (ArchTag::kMinComputeCapability < 90)
    {
        ORT_THROW("Invalid architecture instantiated");
    }
#ifndef COMPILE_HOPPER_TMA_GROUPED_GEMMS
    else if constexpr (ArchTag::kMinComputeCapability >= 90 && ArchTag::kMinComputeCapability < 100)
    {
        ORT_THROW("Please recompile with support for hopper by passing 90-real as an arch to build_wheel.py.");
    }
#endif
#ifndef COMPILE_BLACKWELL_TMA_GROUPED_GEMMS
    else if constexpr (ArchTag::kMinComputeCapability >= 100 && ArchTag::kMinComputeCapability < 120)
    {
        ORT_THROW("Please recompile with support for blackwell by passing 100-real as an arch to build_wheel.py.");
    }
#endif
#ifndef COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
    else if constexpr (ArchTag::kMinComputeCapability >= 120)
    {
        ORT_THROW("Please recompile with support for blackwell by passing 120-real as an arch to build_wheel.py.");
    }
#endif
    else
    {
        return DispatchToTmaWSFunction<ArchTag, T, WeightType, OutputType, EpilogueTag, FUSION, TileShape, ClusterShape,
            IsMXFPX, BIAS>::op(tma_ws_input, num_experts, multi_processor_count, stream, kernel_occupancy,
            workspace_size);
    }
}

#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
constexpr bool COMPILE_HOPPER_TMA_GROUPED_GEMMS_ENABLED = true;
#else
constexpr bool COMPILE_HOPPER_TMA_GROUPED_GEMMS_ENABLED = false;
#endif

#ifdef COMPILE_BLACKWELL_TMA_GROUPED_GEMMS
constexpr bool COMPILE_BLACKWELL_TMA_GROUPED_GEMMS_ENABLED = true;
#else
constexpr bool COMPILE_BLACKWELL_TMA_GROUPED_GEMMS_ENABLED = false;
#endif

#ifdef COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
constexpr bool COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS_ENABLED = true;
#else
constexpr bool COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS_ENABLED = false;
#endif

#ifdef ENABLE_FP8
using SafeFP8 = __nv_fp8_e4m3;
#else
using SafeFP8 = void;
#endif
#ifdef ENABLE_FP4
using SafeFP4 = __nv_fp4_e2m1;
#else
struct SafeFP4
{
};
#endif
#ifdef ENABLE_BF16
using SafeBF16 = __nv_bfloat16;
#else
using SafeBF16 = void;
#endif

// TODO Revert this back to a template instantiation once compiler bug is resolved
#define INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(ArchTag_, DataType_, WeightType_, OutputType_, EpilogueTag_,                                                                                                                                    \
    FUSION_, CTA_M_, CTA_N_, CTA_K_, CGA_M_, CGA_N_, CGA_K_, MXFPX_, BIAS_)                                                                                                                                                                       \
    static void                                                                                                                                                                                                                                   \
        tma_warp_specialized_generic_moe_gemm_kernelLauncher_##ArchTag_##_##DataType_##_##WeightType_##_##OutputType_##_##EpilogueTag_##_##FUSION_##_##CTA_M_##_##CTA_N_##_##CTA_K_##_##CGA_M_##_##CGA_N_##_##CGA_K_##_##MXFPX_##_##BIAS_(        \
            TmaWarpSpecializedGroupedGemmInput tma_ws_input, int num_experts, int const multi_processor_count,                                                                                                                                    \
            cudaStream_t stream, int* kernel_occupancy, size_t* workspace_size)                                                                                                                                                                   \
    {                                                                                                                                                                                                                                             \
        constexpr static EpilogueFusion FUSION = EpilogueFusion::FUSION_;                                                                                                                                                                         \
        /* constexpr static bool BIAS = BIAS_; */ /* Always false */                                                                                                                                                                              \
        using ArchTag = cutlass::arch::ArchTag_;                                                                                                                                                                                                  \
        using T = DataType_;                                                                                                                                                                                                                      \
        using WeightType = WeightType_;                                                                                                                                                                                                           \
        using OutputType = OutputType_;                                                                                                                                                                                                           \
        using EpilogueTag = onnxruntime::llm::cutlass_extensions::EpilogueTag_;                                                                                                                                                                       \
        using TileShape = cute::Shape<cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>>;                                                                                                                                                   \
        using ClusterShape = cute::Shape<cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>;                                                                                                                                                \
        constexpr static bool IsMXFPX = MXFPX_;                                                                                                                                                                                                   \
                                                                                                                                                                                                                                                  \
        if constexpr (!COMPILE_HOPPER_TMA_GROUPED_GEMMS_ENABLED && ArchTag::kMinComputeCapability >= 90                                                                                                                                           \
            && ArchTag::kMinComputeCapability < 100)                                                                                                                                                                                              \
        {                                                                                                                                                                                                                                         \
            ORT_THROW("Please recompile with support for hopper by passing 90-real as an arch to build_wheel.py.");                                                                                                                              \
        }                                                                                                                                                                                                                                         \
        else if constexpr (!COMPILE_BLACKWELL_TMA_GROUPED_GEMMS_ENABLED && ArchTag::kMinComputeCapability >= 100                                                                                                                                  \
            && ArchTag::kMinComputeCapability < 120)                                                                                                                                                                                              \
        {                                                                                                                                                                                                                                         \
            ORT_THROW(                                                                                                                                                                                                                           \
                "Please recompile with support for blackwell by passing 100-real as an arch to build_wheel.py.");                                                                                                                                 \
        }                                                                                                                                                                                                                                         \
        else if constexpr (!COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS_ENABLED                                                                                                                                                                     \
            && ArchTag::kMinComputeCapability >= 120)                                                                                                                                                                                             \
        {                                                                                                                                                                                                                                         \
            ORT_THROW(                                                                                                                                                                                                                           \
                "Please recompile with support for blackwell by passing 120-real as an arch to build_wheel.py.");                                                                                                                                 \
        }                                                                                                                                                                                                                                         \
        else if constexpr (!should_filter_tma_warp_specialized_gemm_problem_shape_v<ArchTag, TileShape, ClusterShape,                                                                                                                             \
                               T>)                                                                                                                                                                                                                \
        {                                                                                                                                                                                                                                         \
            using namespace cute;                                                                                                                                                                                                                 \
            /* Helper class for defining all the cutlass types                                                                                                                                                                                    \
            // template <typename ArchTag, typename T, typename WeightType, typename OutputType, typename EpilogueTag,                                                                                                                            \
            //    typename TileShape, typename ClusterShape, bool BIAS, EpilogueFusion FUSION>                                                                                                                                                    \
            // struct TmaWarpSpecializedGroupedGemmInfo                                                                                                                                                                                           \
            { */                                                                                                                                                                                                                                  \
            using Arch = ArchTag;                                                                                                                                                                                                                 \
            constexpr static bool IsBlackwell = Arch::kMinComputeCapability >= 100;                                                                                                                                                               \
            constexpr static bool IsSM120 = Arch::kMinComputeCapability == 120 || Arch::kMinComputeCapability == 121;                                                                                                                             \
            constexpr static bool IsWFP4AFP8 = cutlass::platform::is_same<WeightType, SafeFP4>::value                                                                                                                                             \
                && cutlass::platform::is_same<T, SafeFP8>::value;                                                                                                                                                                                 \
            constexpr static bool IsFP4 = cutlass::platform::is_same<T, SafeFP4>::value;                                                                                                                                                          \
            static_assert(!IsFP4 || IsBlackwell, "FP4 is only supported by SM100");                                                                                                                                                               \
                                                                                                                                                                                                                                                  \
            constexpr static bool IsFP8 = cutlass::platform::is_same<T, SafeFP8>::value;                                                                                                                                                          \
                                                                                                                                                                                                                                                  \
            /* TODO Update once mixed input support is added */                                                                                                                                                                                   \
            static_assert(cutlass::platform::is_same<T, WeightType>::value || IsWFP4AFP8,                                                                                                                                                         \
                "TMA warp specialized MOE implementation does not support mixed input types");                                                                                                                                                    \
                                                                                                                                                                                                                                                  \
            constexpr static bool IsBlockScaled = IsFP4 || IsWFP4AFP8;                                                                                                                                                                            \
            static_assert(!IsBlockScaled || IsBlackwell, "Block scaled is only implemented for SM100");                                                                                                                                           \
                                                                                                                                                                                                                                                  \
            static_assert(cutlass::platform::is_same<T, SafeBF16>::value || cutlass::platform::is_same<T, half>::value                                                                                                                            \
                    || cutlass::platform::is_same<T, float>::value || IsFP8 || IsFP4,                                                                                                                                                             \
                "Specialized for bfloat16, half, float, fp8, fp4");                                                                                                                                                                               \
                                                                                                                                                                                                                                                  \
            /* The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.*/                                                                                                                              \
            using ElementType = typename CudaToCutlassTypeAdapter<T>::type;                                                                                                                                                                       \
                                                                                                                                                                                                                                                  \
            /* TODO The below never trigger, and are incorrect for int8 types anyway                                                                                                                                                              \
            //    using CutlassWeightTypeMaybeUint4 = typename CudaToCutlassTypeAdapter<WeightType>::type;                                                                                                                                        \
            //    // For legacy reasons we convert unsigned 8-bit to signed                                                                                                                                                                       \
            //    using CutlassWeightTypeMaybeUint8                                                                                                                                                                                               \
            //        = std::conditional_t<std::is_same_v<CutlassWeightTypeMaybeUint4, cutlass::uint4b_t>,                                                                                                                                        \
            cutlass::int4b_t,                                                                                                                                                                                                                     \
            //            CutlassWeightTypeMaybeUint4>;                                                                                                                                                                                           \
            //    using CutlassWeightType                                                                                                                                                                                                         \
            //        = std::conditional_t<std::is_same_v<CutlassWeightTypeMaybeUint8, uint8_t>, int8_t,                                                                                                                                          \
            //        CutlassWeightTypeMaybeUint8>; */                                                                                                                                                                                            \
            using CutlassWeightType = typename CudaToCutlassTypeAdapter<WeightType>::type;                                                                                                                                                        \
                                                                                                                                                                                                                                                  \
            using ElementA = ElementType;                                                                                                                                                                                                         \
            using ElementB = CutlassWeightType;                                                                                                                                                                                                   \
                                                                                                                                                                                                                                                  \
            using ElementD = typename CudaToCutlassTypeAdapter<                                                                                                                                                                                   \
                TmaWarpSpecializedGroupedGemmInput::OutputTypeAdaptor_t<OutputType>>::type;                                                                                                                                                       \
            using ElementFinalOutput = typename CudaToCutlassTypeAdapter<OutputType>::type;                                                                                                                                                       \
                                                                                                                                                                                                                                                  \
            /* using ElementC = std::conditional_t<BIAS, ElementType, void>; */                                                                                                                                                                   \
            /* using ElementCSafe = std::conditional_t<BIAS, ElementType, ElementD>; */                                                                                                                                                           \
            using ElementC = void;                                                                                                                                                                                                                \
            using ElementCSafe = ElementD;                                                                                                                                                                                                        \
                                                                                                                                                                                                                                                  \
            using ElementAccumulator = float;                                                                                                                                                                                                     \
                                                                                                                                                                                                                                                  \
            using ElementBias = ElementFinalOutput;                                                                                                                                                                                               \
            using ElementRouterScales = float;                                                                                                                                                                                                    \
                                                                                                                                                                                                                                                  \
            using ElementSF = std::conditional_t<IsMXFPX, cutlass::float_ue8m0_t,                                                                                                                                                                 \
                cutlass::float_ue4m3_t>; /*TmaWarpSpecializedGroupedGemmInput::ElementSF;*/                                                                                                                                                       \
            using ElementABlockScaled                                                                                                                                                                                                             \
                = std::conditional_t<IsSM120, cutlass::nv_float4_t<ElementA>, cute::tuple<ElementA, ElementSF>>;                                                                                                                                  \
            using ElementBBlockScaled                                                                                                                                                                                                             \
                = std::conditional_t<IsSM120, cutlass::nv_float4_t<ElementB>, cute::tuple<ElementB, ElementSF>>;                                                                                                                                  \
                                                                                                                                                                                                                                                  \
            /* A matrix configuration - this is transposed and swapped with B */                                                                                                                                                                  \
            using LayoutA = TmaWarpSpecializedGroupedGemmInput::LayoutA;                                                                                                                                                                          \
            constexpr static int AlignmentA                                                                                                                                                                                                       \
                = 128 / cutlass::sizeof_bits<ElementA>::value; /* Memory access granularity/alignment of A matrix in                                                                                                                              \
                                                               units of elements (up to 16 bytes) */                                                                                                                                              \
            /* B matrix configuration - this is transposed and swapped with A */                                                                                                                                                                  \
            using LayoutB = TmaWarpSpecializedGroupedGemmInput::LayoutB; /* Layout type for B matrix operand */                                                                                                                                   \
            constexpr static int AlignmentB = IsWFP4AFP8                                                                                                                                                                                          \
                ? 128                                                                                                                                                                                                                             \
                : (128 / cutlass::sizeof_bits<ElementB>::value); /* Memory access granularity/alignment of B matrix in                                                                                                                            \
                                                                units                                                                                                                                                                             \
                                                                // of elements (up to 16 bytes)*/                                                                                                                                                 \
                                                                                                                                                                                                                                                  \
            /* C matrix configuration */                                                                                                                                                                                                          \
            using LayoutC = TmaWarpSpecializedGroupedGemmInput::LayoutC; /* Layout type for C matrix operand */                                                                                                                                   \
            using StrideC = TmaWarpSpecializedGroupedGemmInput::StrideC;                                                                                                                                                                          \
            /* Note we use ElementType here deliberately, so we don't break when BIAS is disabled */                                                                                                                                              \
            constexpr static int AlignmentC = 128                                                                                                                                                                                                 \
                / cutlass::sizeof_bits<ElementType>::value; /* Memory access granularity/alignment of C matrix in                                                                                                                                 \
                                                            // units of elements (up to 16 bytes)*/                                                                                                                                               \
                                                                                                                                                                                                                                                  \
            /* D matrix configuration */                                                                                                                                                                                                          \
            using LayoutD = TmaWarpSpecializedGroupedGemmInput::DefaultEpilogue::LayoutD;                                                                                                                                                         \
            using StrideD = TmaWarpSpecializedGroupedGemmInput::DefaultEpilogue::StrideD;                                                                                                                                                         \
            constexpr static int AlignmentD                                                                                                                                                                                                       \
                = 128 / cutlass::sizeof_bits<ElementD>::value; /* Memory access granularity/alignment of D matrix                                                                                                                                 \
                                                               // in units of elements (up to 16 bytes) */                                                                                                                                        \
                                                                                                                                                                                                                                                  \
            static_assert(                                                                                                                                                                                                                        \
                cutlass::platform::is_same<EpilogueTag, onnxruntime::llm::cutlass_extensions::EpilogueOpDefault>::value,                                                                                                                              \
                "TMA Warp Specialized Grouped GEMM specialisation doesn't support fused activation");                                                                                                                                             \
                                                                                                                                                                                                                                                  \
            using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator, ElementC,                                                                                                                               \
                ElementAccumulator>;                                                                                                                                                                                                              \
                                                                                                                                                                                                                                                  \
            /* TODO Add mode for fused activation once CUTLASS adds support                                                                                                                                                                       \
            //  using EpilogueSchedule = cutlass::platform::conditional_t<                                                                                                                                                                        \
            //        cutlass::platform::is_same<EpilogueOp, EpilogueOpDefault>::value,                                                                                                                                                           \
            //        cutlass::epilogue::PtrArrayNoSmemWarpSpecialized,                                                                                                                                                                           \
            //        cutlass::epilogue::??????????????????             /// <<<<<< what supports activations                                                                                                                                      \
            //        >;*/                                                                                                                                                                                                                        \
            using EpilogueScheduleSM90 = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;                                                                                                                                                        \
                                                                                                                                                                                                                                                  \
            constexpr static bool Is2SM = IsBlackwell && (cute::size<0>(ClusterShape{}) % 2) == 0;                                                                                                                                                \
            using EpilogueScheduleSM100 = std::conditional_t<Is2SM, cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,                                                                                                                             \
                cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>;                                                                                                                                                                                \
            using EpilogueScheduleSM120 = cutlass::epilogue::TmaWarpSpecialized;                                                                                                                                                                  \
            using EpilogueScheduleBW = std ::conditional_t<IsSM120, EpilogueScheduleSM120, EpilogueScheduleSM100>;                                                                                                                                \
            using EpilogueSchedule = std::conditional_t<IsBlackwell, EpilogueScheduleBW, EpilogueScheduleSM90>;                                                                                                                                   \
                                                                                                                                                                                                                                                  \
            using EpilogueTileShapeSm90 = TileShape;                                                                                                                                                                                              \
            using AtomClusterDiv = std::conditional_t<Is2SM, _2, _1>;                                                                                                                                                                             \
            using AtomThrShape = decltype(shape_div(ClusterShape{}, Shape<AtomClusterDiv, _1, _1>{}));                                                                                                                                            \
            using EpilogueTileShapeSm100 = decltype(shape_div(TileShape{}, AtomThrShape{}));                                                                                                                                                      \
            using EpilogueTileShape = std::conditional_t<IsBlackwell, EpilogueTileShapeSm100, EpilogueTileShapeSm90>;                                                                                                                             \
            using EpilogueElementC = std::conditional_t<IsSM120, ElementCSafe, ElementC>;                                                                                                                                                         \
            using EpilogueTensorOp = std::conditional_t<IsBlackwell && IsBlockScaled,                                                                                                                                                             \
                cutlass::arch::OpClassBlockScaledTensorOp, cutlass::arch::OpClassTensorOp>;                                                                                                                                                       \
            using EpilogueSubTile                                                                                                                                                                                                                 \
                = std::conditional_t<Arch::kMinComputeCapability == 100 && IsFP4 && CTA_N_ == 256, /* SM100 Exactly */                                                                                                                            \
                    cute::Shape<cute::_128, cute::_64>, cutlass::epilogue::collective::EpilogueTileAuto>;                                                                                                                                         \
            /* Epilogue For Default Finalize */                                                                                                                                                                                                   \
            using CollectiveEpilogueDefault = typename cutlass::epilogue::collective::CollectiveBuilder</**/                                                                                                                                      \
                Arch, EpilogueTensorOp,                                                                 /**/                                                                                                                                      \
                EpilogueTileShape, ClusterShape,                                                        /**/                                                                                                                                      \
                EpilogueSubTile,                                                                        /**/                                                                                                                                      \
                ElementAccumulator, ElementAccumulator,                                                 /**/                                                                                                                                      \
                EpilogueElementC, LayoutC*, AlignmentC,                                                 /**/                                                                                                                                      \
                ElementD, LayoutD*, AlignmentD,                                                         /**/                                                                                                                                      \
                EpilogueSchedule>::CollectiveOp;                                                                                                                                                                                                  \
                                                                                                                                                                                                                                                  \
            /* Epilogue For Fused Finalize */                                                                                                                                                                                                     \
            using CollectiveEpilogueFinalize =                                                                                                                                                                                                    \
                typename cutlass::epilogue::collective::EpilogueMoeFusedFinalizeBuilder<                /**/                                                                                                                                      \
                    Arch, EpilogueTileShape,                                                            /**/                                                                                                                                      \
                    ElementCSafe, StrideC*,                                                             /**/                                                                                                                                      \
                    ElementFinalOutput,                                                                                                                                                                                                           \
                    TmaWarpSpecializedGroupedGemmInput::FusedFinalizeEpilogue::StrideFinalOutput,       /**/                                                                                                                                      \
                    ElementAccumulator,                                                                 /**/                                                                                                                                      \
                    ElementAccumulator,                                                                 /**/                                                                                                                                      \
                    ElementBias, TmaWarpSpecializedGroupedGemmInput::FusedFinalizeEpilogue::StrideBias, /**/                                                                                                                                      \
                    ElementRouterScales,                                                                                                                                                                                                          \
                    TmaWarpSpecializedGroupedGemmInput::FusedFinalizeEpilogue::StrideRouterScales       /**/                                                                                                                                      \
                    >::CollectiveOp;                                                                                                                                                                                                              \
                                                                                                                                                                                                                                                  \
            using CollectiveEpilogue = std::conditional_t<FUSION == EpilogueFusion::FINALIZE,                                                                                                                                                     \
                CollectiveEpilogueFinalize, CollectiveEpilogueDefault>;                                                                                                                                                                           \
                                                                                                                                                                                                                                                  \
            using StageCountAutoCarveout = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(                                                                                                                                    \
                sizeof(typename CollectiveEpilogue::SharedStorage))>;                                                                                                                                                                             \
                                                                                                                                                                                                                                                  \
            using KernelScheduleSM90                                                                                                                                                                                                              \
                = std::conditional_t<IsFP8, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,                                                                                                                               \
                    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;                                                                                                                                                                  \
                                                                                                                                                                                                                                                  \
            using KernelSchedule2SmSm100BlockScaled                                                                                                                                                                                               \
                = std::conditional_t<IsMXFPX, cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100,                                                                                                                                    \
                    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmNvf4Sm100>;                                                                                                                                                                 \
            using KernelSchedule1SmSm100BlockScaled                                                                                                                                                                                               \
                = std::conditional_t<IsMXFPX, cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100,                                                                                                                                    \
                    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100>;                                                                                                                                                                 \
                                                                                                                                                                                                                                                  \
            /* TRT-LLM uses vector size 16 for block scaled */                                                                                                                                                                                    \
            using KernelScheduleSM100 = std::conditional_t<Is2SM,                                                                                                                                                                                 \
                std::conditional_t<IsBlockScaled, KernelSchedule2SmSm100BlockScaled,                                                                                                                                                              \
                    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>,                                                                                                                                                                     \
                std::conditional_t<IsBlockScaled, KernelSchedule1SmSm100BlockScaled,                                                                                                                                                              \
                    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>>;                                                                                                                                                                    \
            using KernelScheduleSM120 = cutlass ::gemm ::collective::KernelScheduleAuto;                                                                                                                                                          \
            using KernelScheduleBW = std::conditional_t<IsSM120, KernelScheduleSM120, KernelScheduleSM100>;                                                                                                                                       \
                                                                                                                                                                                                                                                  \
            using KernelSchedule = std::conditional_t<IsBlackwell, KernelScheduleBW, KernelScheduleSM90>;                                                                                                                                         \
                                                                                                                                                                                                                                                  \
            using TensorOp = std::conditional_t<IsBlackwell && IsBlockScaled,                                                                                                                                                                     \
                cutlass::arch::OpClassBlockScaledTensorOp, cutlass::arch::OpClassTensorOp>;                                                                                                                                                       \
                                                                                                                                                                                                                                                  \
            using MainloopElementA = std::conditional_t<IsBlackwell && IsBlockScaled, ElementABlockScaled, ElementA>;                                                                                                                             \
            using MainloopElementB = std::conditional_t<IsBlackwell && IsBlockScaled, ElementBBlockScaled, ElementB>;                                                                                                                             \
                                                                                                                                                                                                                                                  \
            using MainloopTileShapeSm90 = TileShape;                                                                                                                                                                                              \
            using MainloopTileShapeSm100 = decltype(shape_div(TileShape{}, AtomThrShape{}));                                                                                                                                                      \
            using MainloopTileShape = std::conditional_t<IsBlackwell, MainloopTileShapeSm100, MainloopTileShapeSm90>;                                                                                                                             \
                                                                                                                                                                                                                                                  \
            using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder</**/                                                                                                                                                 \
                Arch, TensorOp,                                                              /**/                                                                                                                                                 \
                MainloopElementB, LayoutB*, AlignmentB,                                      /* A & B swapped here */                                                                                                                             \
                MainloopElementA, LayoutA*, AlignmentA,                                      /**/                                                                                                                                                 \
                ElementAccumulator,                                                          /**/                                                                                                                                                 \
                MainloopTileShape, ClusterShape,                                             /**/                                                                                                                                                 \
                StageCountAutoCarveout, KernelSchedule>::CollectiveOp;                                                                                                                                                                            \
                                                                                                                                                                                                                                                  \
            using GemmKernel = cutlass::gemm::kernel::GemmUniversal<TmaWarpSpecializedGroupedGemmInput::ProblemShape,                                                                                                                             \
                CollectiveMainloop, CollectiveEpilogue, void, void>;                                                                                                                                                                              \
                                                                                                                                                                                                                                                  \
            using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                                                                                                                                                          \
            /*};                                                                                                                                                                                                                                  \
                                                                                                                                                                                           \                                                      \
            //        using namespace cute;                                                                                                                                                                                                       \
            //        using GemmInfo = TmaWarpSpecializedGroupedGemmInfo;<ArchTag, T, WeightType, OutputType,                                                                                                                                     \
            EpilogueTag,                                                                                                                                                                                                                          \
            //        TileShape,                                                                                                                                                                                                                  \
            //            ClusterShape, BIAS, FUSION>;                                                                                                                                                                                            \
            //                                                                                                                                                                                                                                    \
            //        using ElementAccumulator = typename GemmInfo::ElementAccumulator;                                                                                                                                                           \
            //        using ElementA = typename GemmInfo::ElementA;                                                                                                                                                                               \
            //        using ElementB = typename GemmInfo::ElementB;                                                                                                                                                                               \
            //        using ElementC = typename GemmInfo::ElementC;                                                                                                                                                                               \
            //        using ElementCSafe = typename GemmInfo::ElementCSafe;                                                                                                                                                                       \
            //        using ElementD = typename GemmInfo::ElementD;                                                                                                                                                                               \
            //        using ElementFinalOutput = typename GemmInfo::ElementFinalOutput;                                                                                                                                                           \
            //        using ElementBias = typename GemmInfo::ElementBias;                                                                                                                                                                         \
            //                                                                                                                                                                                                                                    \
            //        using CollectiveMainloop = typename GemmInfo::CollectiveMainloop;                                                                                                                                                           \
            //        using CollectiveEpilogue = typename GemmInfo::CollectiveEpilogue;                                                                                                                                                           \
            //        using GemmKernel = typename GemmInfo::GemmKernel;                                                                                                                                                                           \
            //        using GemmGrouped = typename GemmInfo::GemmGrouped;*/                                                                                                                                                                       \
                                                                                                                                                                                                                                                  \
            if (kernel_occupancy != nullptr)                                                                                                                                                                                                      \
            {                                                                                                                                                                                                                                     \
                ORT_THROW("TMA WS kernels do not support calculating occupancy");                                                                                                                                                                \
                return;                                                                                                                                                                                                                           \
            }                                                                                                                                                                                                                                     \
                                                                                                                                                                                                                                                  \
            cutlass::KernelHardwareInfo hw_info;                                                                                                                                                                                                  \
            hw_info.device_id = 0;                                                                                                                                                                                                                \
            hw_info.sm_count = multi_processor_count;                                                                                                                                                                                             \
                                                                                                                                                                                                                                                  \
            GemmGrouped gemm;                                                                                                                                                                                                                     \
                                                                                                                                                                                                                                                  \
            if (workspace_size != nullptr)                                                                                                                                                                                                        \
            {                                                                                                                                                                                                                                     \
                /* Make a mock problem shape with just the minimal information actually required to get the workspace                                                                                                                             \
                // size This makes some assumptions about CUTLASS's implementation which is suboptimal. We have a                                                                                                                                 \
                check                                                                                                                                                                                                                             \
                // later to catch future cutlass updates causing silent breakages, but that is not fool proof. The                                                                                                                                \
                // alternative is to wait until we have data and then dynamically allocate the workspace*/                                                                                                                                        \
                typename TmaWarpSpecializedGroupedGemmInput::ProblemShape shape_info{num_experts, nullptr, nullptr};                                                                                                                              \
                                                                                                                                                                                                                                                  \
                typename GemmKernel::TileScheduler::Arguments scheduler_args{                                                                                                                                                                     \
                    1, GemmKernel::TileScheduler::RasterOrderOptions::AlongN};                                                                                                                                                                    \
                const typename GemmGrouped::Arguments args{                                                                                                                                                                                       \
                    cutlass::gemm::GemmUniversalMode::kGrouped, shape_info, {}, {}, hw_info, scheduler_args};                                                                                                                                     \
                *workspace_size = gemm.get_workspace_size(args);                                                                                                                                                                                  \
                return;                                                                                                                                                                                                                           \
            }                                                                                                                                                                                                                                     \
                                                                                                                                                                                                                                                  \
            using MainloopArguments = typename CollectiveMainloop::Arguments;                                                                                                                                                                     \
            ORT_ENFORCE(tma_ws_input.stride_a);                                                                                                                                                                                                    \
            ORT_ENFORCE(tma_ws_input.stride_b);                                                                                                                                                                                                    \
            ORT_ENFORCE(tma_ws_input.ptr_a);                                                                                                                                                                                                       \
            ORT_ENFORCE(tma_ws_input.ptr_b);                                                                                                                                                                                                       \
                                                                                                                                                                                                                                                  \
            auto make_mainloop_params = [&]() -> MainloopArguments                                                                                                                                                                                \
            {                                                                                                                                                                                                                                     \
                if constexpr (IsBlockScaled)                                                                                                                                                                                                      \
                {                                                                                                                                                                                                                                 \
                    return construct_if_true<IsBlockScaled, MainloopArguments>(                                                                                                                                                                   \
                        reinterpret_cast<ElementB const**>(tma_ws_input.ptr_b), tma_ws_input.stride_b,                                                                                                                                            \
                        reinterpret_cast<ElementA const**>(tma_ws_input.ptr_a), tma_ws_input.stride_a,                                                                                                                                            \
                        reinterpret_cast<ElementSF const**>(tma_ws_input.fpX_block_scaling_factors_B),                                                                                                                                            \
                        reinterpret_cast<decltype(deduce_layout_sf<IsBlockScaled, GemmGrouped, false>())>(                                                                                                                                        \
                            tma_ws_input.fpX_block_scaling_factors_stride_B),                                                                                                                                                                     \
                        reinterpret_cast<ElementSF const**>(tma_ws_input.fpX_block_scaling_factors_A),                                                                                                                                            \
                        reinterpret_cast<decltype(deduce_layout_sf<IsBlockScaled, GemmGrouped, true>())>(                                                                                                                                         \
                            tma_ws_input.fpX_block_scaling_factors_stride_A));                                                                                                                                                                    \
                }                                                                                                                                                                                                                                 \
                else                                                                                                                                                                                                                              \
                {                                                                                                                                                                                                                                 \
                    return construct_if_true<!IsBlockScaled, MainloopArguments>(                                                                                                                                                                  \
                        reinterpret_cast<ElementB const**>(tma_ws_input.ptr_b), tma_ws_input.stride_b,                                                                                                                                            \
                        reinterpret_cast<ElementA const**>(tma_ws_input.ptr_a), tma_ws_input.stride_a);                                                                                                                                           \
                }                                                                                                                                                                                                                                 \
            };                                                                                                                                                                                                                                    \
                                                                                                                                                                                                                                                  \
            auto const mainloop_params = make_mainloop_params();                                                                                                                                                                                  \
                                                                                                                                                                                                                                                  \
            using EpilogueArguments = typename CollectiveEpilogue::Arguments;                                                                                                                                                                     \
            using EpilogueScalars = decltype(EpilogueArguments{}.thread);                                                                                                                                                                         \
            auto make_epilogue_scalars = [&]()                                                                                                                                                                                                    \
            {                                                                                                                                                                                                                                     \
                if constexpr (IsBlackwell)                                                                                                                                                                                                        \
                {                                                                                                                                                                                                                                 \
                    return construct_if_true<IsBlackwell, EpilogueScalars>(ElementAccumulator(1.f),                                                                                                                                               \
                        tma_ws_input.ptr_c ? ElementAccumulator(1.f) : ElementAccumulator(0.f), nullptr, nullptr,                                                                                                                                 \
                        tma_ws_input.alpha_scale_ptr_array, nullptr,                                                                                                                                                                              \
                        cute::Shape<_0, _0, int64_t>{                                                                                                                                                                                             \
                            cute::_0{}, cute::_0{}, (tma_ws_input.alpha_scale_ptr_array != nullptr) ? 1 : 0},                                                                                                                                     \
                        cute::Shape<_0, _0, int64_t>{cute::_0{}, cute::_0{}, 0});                                                                                                                                                                 \
                }                                                                                                                                                                                                                                 \
                else if (tma_ws_input.alpha_scale_ptr_array)                                                                                                                                                                                      \
                {                                                                                                                                                                                                                                 \
                    return construct_if_true<!IsBlackwell, EpilogueScalars>(tma_ws_input.alpha_scale_ptr_array);                                                                                                                                  \
                }                                                                                                                                                                                                                                 \
                else                                                                                                                                                                                                                              \
                {                                                                                                                                                                                                                                 \
                    return construct_if_true<!IsBlackwell, EpilogueScalars>(ElementAccumulator(1.f),                                                                                                                                              \
                        tma_ws_input.ptr_c ? ElementAccumulator(1.f) : ElementAccumulator(0.f));                                                                                                                                                  \
                }                                                                                                                                                                                                                                 \
            };                                                                                                                                                                                                                                    \
            auto epilogue_scalars = make_epilogue_scalars();                                                                                                                                                                                      \
            /* TODO ptr_c casts to ElementCSafe** because there is a workaround in CUTLASS */                                                                                                                                                     \
            auto make_epi_args = [&]()                                                                                                                                                                                                            \
            {                                                                                                                                                                                                                                     \
                static_assert(FUSION == EpilogueFusion::NONE || FUSION == EpilogueFusion::FINALIZE,                                                                                                                                               \
                    "Unimplemented fusion provided to TMA WS MoE gemm launcher");                                                                                                                                                                 \
                                                                                                                                                                                                                                                  \
                if constexpr (FUSION == EpilogueFusion::NONE)                                                                                                                                                                                     \
                {                                                                                                                                                                                                                                 \
                    auto epi_params = tma_ws_input.default_epilogue;                                                                                                                                                                              \
                    return construct_if_true<FUSION == EpilogueFusion::NONE, EpilogueArguments>(epilogue_scalars,                                                                                                                                 \
                        nullptr, tma_ws_input.stride_c, reinterpret_cast<ElementD**>(epi_params.ptr_d),                                                                                                                                           \
                        epi_params.stride_d);                                                                                                                                                                                                     \
                }                                                                                                                                                                                                                                 \
                else if constexpr (FUSION == EpilogueFusion::FINALIZE)                                                                                                                                                                            \
                {                                                                                                                                                                                                                                 \
                    /* Parameters for fused finalize */                                                                                                                                                                                           \
                    auto epi_params = tma_ws_input.fused_finalize_epilogue;                                                                                                                                                                       \
                    return construct_if_true<FUSION == EpilogueFusion::FINALIZE, EpilogueArguments>(                                                                                                                                              \
                        epilogue_scalars,               /* Parameters to underlying epilogue */                                                                                                                                                   \
                        nullptr, tma_ws_input.stride_c, /* C params */                                                                                                                                                                            \
                        reinterpret_cast<ElementFinalOutput*>(epi_params.ptr_final_output),                                                                                                                                                       \
                        epi_params.stride_final_output, /* D (output) params */                                                                                                                                                                   \
                        reinterpret_cast<ElementBias const*>(epi_params.ptr_bias),                                                                                                                                                                \
                        epi_params.stride_bias,         /* Bias params */                                                                                                                                                                         \
                        epi_params.ptr_router_scales, epi_params.stride_router_scales, /* Router scales */                                                                                                                                        \
                        epi_params.ptr_expert_first_token_offset, /* Offset of this expert's token in the                                                                                                                                         \
                                                                     router scales */                                                                                                                                                             \
                        epi_params.ptr_source_token_index,        /* Index of the source token to sum into */                                                                                                                                     \
                        epi_params.num_rows_in_final_output       /* Number of tokens in the output buffer */                                                                                                                                     \
                    );                                                                                                                                                                                                                            \
                }                                                                                                                                                                                                                                 \
            };                                                                                                                                                                                                                                    \
            EpilogueArguments const epilogue_params = make_epi_args();                                                                                                                                                                            \
            /*        EpilogueArguments const epilogue_params = make_epi_args<EpilogueArguments, EpilogueScalars,                                                                                                                                 \
            ElementCSafe, ElementD, ElementFinalOutput, ElementBias, FUSION>(                                                                                                                                                                     \
            //            tma_ws_input, epilogue_scalars                                                                                                                                                                                          \
            //        );*/                                                                                                                                                                                                                        \
                                                                                                                                                                                                                                                  \
            typename GemmKernel::TileScheduler::Arguments scheduler_args{                                                                                                                                                                         \
                1, GemmKernel::TileScheduler::RasterOrderOptions::AlongN};                                                                                                                                                                        \
                                                                                                                                                                                                                                                  \
            const typename GemmGrouped::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,                                                                                                                                                \
                tma_ws_input.shape_info, mainloop_params, epilogue_params, hw_info, scheduler_args};                                                                                                                                              \
                                                                                                                                                                                                                                                  \
            size_t calculated_ws_size = gemm.get_workspace_size(args);                                                                                                                                                                            \
            ORT_ENFORCE(calculated_ws_size <= tma_ws_input.gemm_workspace_size,                                                                                                                                                          \
                "Workspace is size %zu but only %zu were allocated", calculated_ws_size,                                                                                                                                                          \
                tma_ws_input.gemm_workspace_size);                                                                                                                                                                                                \
                                                                                                                                                                                                                                                  \
            auto can_implement = gemm.can_implement(args);                                                                                                                                                                                        \
            ORT_ENFORCE(can_implement == cutlass::Status::kSuccess,                                                                                                                                                                      \
                "Grouped GEMM kernel will fail for params. Error: "                                                                                                                                                                               \
                    + std::string(cutlass::cutlassGetStatusString(can_implement)));                                                                                                                                                               \
                                                                                                                                                                                                                                                  \
            auto init_status = gemm.initialize(args, tma_ws_input.gemm_workspace);                                                                                                                                                                \
            ORT_ENFORCE(init_status == cutlass::Status::kSuccess,                                                                                                                                                                        \
                "Failed to initialize cutlass TMA WS grouped gemm. Error: "                                                                                                                                                                       \
                    + std::string(cutlass::cutlassGetStatusString(init_status)));                                                                                                                                                                 \
            auto run_status = gemm.run(stream, nullptr, onnxruntime::llm::common::getEnvEnablePDL());                                                                                                                                                 \
            ORT_ENFORCE(run_status == cutlass::Status::kSuccess,                                                                                                                                                                         \
                "Failed to run cutlass TMA WS grouped gemm. Error: "                                                                                                                                                                              \
                    + std::string(cutlass::cutlassGetStatusString(run_status)));                                                                                                                                                                  \
            sync_check_cuda_error(stream);                                                                                                                                                                                                        \
        }                                                                                                                                                                                                                                         \
        else                                                                                                                                                                                                                                      \
        {                                                                                                                                                                                                                                         \
            ORT_THROW("Configuration was disabled by FAST_BUILD");                                                                                                                                                                               \
        }                                                                                                                                                                                                                                         \
                                                                                                                                                                                                                                                  \
        return;                                                                                                                                                                                                                                   \
    }                                                                                                                                                                                                                                             \
                                                                                                                                                                                                                                                  \
    template <>                                                                                                                                                                                                                                   \
    struct DispatchToTmaWSFunction<cutlass::arch::ArchTag_, DataType_, WeightType_, OutputType_,                                                                                                                                                  \
        onnxruntime::llm::cutlass_extensions::EpilogueTag_, EpilogueFusion::FUSION_,                                                                                                                                                                  \
        cute::Shape<cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>>,                                                                                                                                                                     \
        cute::Shape<cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>, MXFPX_, BIAS_>                                                                                                                                                      \
    {                                                                                                                                                                                                                                             \
        constexpr static auto* op                                                                                                                                                                                                                 \
            = &tma_warp_specialized_generic_moe_gemm_kernelLauncher_##ArchTag_##_##DataType_##_##WeightType_##_##OutputType_##_##EpilogueTag_##_##FUSION_##_##CTA_M_##_##CTA_N_##_##CTA_K_##_##CGA_M_##_##CGA_N_##_##CGA_K_##_##MXFPX_##_##BIAS_; \
    };                                                                                                                                                                                                                                            \
    template void tma_warp_specialized_generic_moe_gemm_kernelLauncher<cutlass::arch::ArchTag_, DataType_,                                                                                                                                        \
        WeightType_, OutputType_, onnxruntime::llm::cutlass_extensions::EpilogueTag_, EpilogueFusion::FUSION_,                                                                                                                                        \
        cute::Shape<cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>>,                                                                                                                                                                     \
        cute::Shape<cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>, MXFPX_, BIAS_>(                                                                                                                                                     \
        TmaWarpSpecializedGroupedGemmInput tma_ws_input, int num_experts, int const multi_processor_count,                                                                                                                                        \
        cudaStream_t stream, int* kernel_occupancy, size_t* workspace_size);

} // namespace cutlass_kernels
} // namespace kernels
} // namespace onnxruntime::llm
