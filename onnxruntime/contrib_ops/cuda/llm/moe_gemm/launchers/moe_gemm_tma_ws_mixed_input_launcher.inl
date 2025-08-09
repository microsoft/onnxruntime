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

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "contrib_ops/cuda/llm/cutlass_extensions/compute_occupancy.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/cutlass_heuristic.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"
#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.h"

namespace  onnxruntime::llm
{
namespace kernels
{
namespace cutlass_kernels
{
namespace tk = onnxruntime::llm::common;
namespace tkc = onnxruntime::llm::cutlass_extensions;

using namespace cute;

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag, typename CTAShape,
    typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
    cutlass::WeightOnlyQuantOp QuantOp>
void sm90_generic_mixed_moe_gemm_kernelLauncher(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size)
{
    ORT_LLM_LOG_ENTRY();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // A matrix configuration
    // using ElementA = typename CudaToCutlassTypeAdapter<T>::type;
    using ElementA = cutlass::float_e4m3_t;
    using LayoutA = cutlass::layout::RowMajor;         // Layout type for A matrix operand
    constexpr int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    // using ElementB = typename CudaToCutlassTypeAdapter<WeightType>::type;
    using ElementB = typename cutlass::int4b_t;
    using LayoutB = cutlass::layout::ColumnMajor;      // Layout type for B matrix operand
    constexpr int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B matrix in units of
                                                       // elements (up to 16 bytes)

    // This example manually swaps and transposes, so keep transpose of input layouts
    using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
    using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

    // Need to pass a pointer type to make the 3rd dimension of Stride be _0
    using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
    using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

    // Scale configuration
    constexpr int PackedScalesNum = get<2>(CTAShape{}) / 128;
    using ElementScalePacked
        = cutlass::Array<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA, PackedScalesNum>;
    using LayoutScale = cutlass::layout::RowMajor;

    // C/D matrix configuration
    using ElementC = typename CudaToCutlassTypeAdapter<GemmOutputType>::type;
    using LayoutC = cutlass::layout::RowMajor;         // Layout type for C and D matrix operands
    constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C matrix in units of
                                                       // elements (up to 16 bytes)

    // D matrix configuration
    using ElementD = ElementC;
    using LayoutD = LayoutC;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Core kernel configurations
    using ElementAccumulator = float;    // Element type for internal accumulation
    using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp;             // Operator class tag
    using TileShape = CTAShape;                                       // Threadblock-level tile size
    using StageCountType = cutlass::gemm::collective::StageCountAuto; // Stage count maximized based on the tile size
    using KernelSchedule
        = std::conditional_t<std::is_same_v<MainloopScheduleType, cutlass::gemm::KernelTmaWarpSpecializedPingpong>,
            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
    using EpilogueSchedule
        = std::conditional_t<std::is_same_v<KernelSchedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>,
            cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
            cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>; // Epilogue to launch

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type*,
        AlignmentC, ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type*, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    // =========================================================== MIXED INPUT WITH SCALES
    // =========================================================================== The Scale information must get paired
    // with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the
    // scale information.
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<ArchTag, OperatorClass,
        cute::tuple<ElementB, ElementScalePacked>, LayoutB_Transpose*, AlignmentB, ElementA, LayoutA_Transpose*,
        AlignmentA, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<Shape<int, int, int>>,
        CollectiveMainloop, CollectiveEpilogue>;

    using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;

    GemmGrouped gemm;
    using Args = typename GemmGrouped::Arguments;
    Args arguments;

    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = inputs.alpha_scales;
    fusion_args.beta_ptr_array = nullptr;
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = sm_count_;

    if (workspace_size != nullptr)
    {
        const Args args{cutlass::gemm::GemmUniversalMode::kGrouped,
            {inputs.num_experts, hopper_inputs.int4_groupwise_params.shape.problem_shapes, nullptr},
            {reinterpret_cast<ElementB const**>(hopper_inputs.ptr_b), hopper_inputs.stride_b,
                reinterpret_cast<ElementA const**>(hopper_inputs.ptr_a), hopper_inputs.stride_a,
                reinterpret_cast<ElementScalePacked const**>(hopper_inputs.int4_groupwise_params.ptr_s_a),
                hopper_inputs.int4_groupwise_params.stride_s_a, int(inputs.groupwise_quant_group_size)},
            {fusion_args, reinterpret_cast<ElementC const**>(hopper_inputs.ptr_c), hopper_inputs.stride_c,
                reinterpret_cast<ElementD**>(hopper_inputs.default_epilogue.ptr_d),
                hopper_inputs.default_epilogue.stride_d},
            hw_info};
        *workspace_size = gemm.get_workspace_size(args);
        return;
    }

    arguments = Args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {inputs.num_experts, hopper_inputs.int4_groupwise_params.shape.problem_shapes, nullptr},
        {reinterpret_cast<ElementB const**>(hopper_inputs.ptr_b), hopper_inputs.stride_b,
            reinterpret_cast<ElementA const**>(hopper_inputs.ptr_a), hopper_inputs.stride_a,
            reinterpret_cast<ElementScalePacked const**>(hopper_inputs.int4_groupwise_params.ptr_s_a),
            hopper_inputs.int4_groupwise_params.stride_s_a, int(inputs.groupwise_quant_group_size)},
        {fusion_args, reinterpret_cast<ElementC const**>(hopper_inputs.ptr_c), hopper_inputs.stride_c,
            reinterpret_cast<ElementD**>(hopper_inputs.default_epilogue.ptr_d),
            hopper_inputs.default_epilogue.stride_d},
        hw_info};

    if (gemm.get_workspace_size(arguments) > hopper_inputs.gemm_workspace_size)
    {
        LLM_LOG_ERROR("[Mixed dtype WS grouped GEMM] given workspace size insufficient, %d < %d.",
            gemm.get_workspace_size(arguments), hopper_inputs.gemm_workspace_size);
    }

    auto can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string err_msg = "mixed dtype WS grouped cutlass kernel will fail for params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        std::cout << err_msg << std::endl;
        ORT_THROW("[Mixed dtype WS grouped GEMM] " + err_msg);
    }

    auto init_status = gemm.initialize(arguments, hopper_inputs.gemm_workspace, inputs.stream);
    if (init_status != cutlass::Status::kSuccess)
    {
        std::string err_msg = "Failed to initialize cutlass mixed dtype WS grouped gemm. Error: "
            + std::string(cutlassGetStatusString(init_status));
        ORT_THROW("[Mixed dtype WS grouped GEMM] " + err_msg);
    }

    auto run_status = gemm.run(inputs.stream);
    if (run_status != cutlass::Status::kSuccess)
    {
        std::string err_msg = "Failed to run cutlass mixed dtype WS grouped gemm. Error: "
            + std::string(cutlassGetStatusString(run_status));
        ORT_THROW("[Mixed dtype WS grouped GEMM] " + err_msg);
    }
    return;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace  onnxruntime::llm
