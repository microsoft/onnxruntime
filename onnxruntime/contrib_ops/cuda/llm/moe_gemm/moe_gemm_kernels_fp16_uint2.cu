#include <limits>

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_int2.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {

bool IsInt2GroupedGemmSupported(const Int2GroupedGemmParams& params) {
  return params.sm == 80 && params.block_size == 64 && params.num_rows > 0 &&
         params.num_rows <= std::numeric_limits<int>::max() && params.num_experts > 0 &&
         params.num_columns > 0 && params.num_columns % 64 == 0 &&
         params.reduction_size > 0 && params.reduction_size % 64 == 0 &&
         params.multiprocessor_count > 0 &&
         (params.tile_rows == 32 || params.tile_rows == 64);
}

void RunInt2GroupedGemm(const Int2GroupedGemmParams& params) {
  ORT_ENFORCE(IsInt2GroupedGemmSupported(params), "Unsupported SM80 FP16 INT2 grouped GEMM configuration");
  ORT_ENFORCE(params.activations && params.packed_weights && params.block_scales &&
                  params.expert_row_ends && params.output,
              "INT2 grouped GEMM requires non-null device buffers");
  GroupedGemmInput<half, cutlass::uint2b_t, half, half> inputs{
      params.activations, params.expert_row_ends, reinterpret_cast<const cutlass::uint2b_t*>(params.packed_weights), params.block_scales, nullptr, nullptr, params.output, nullptr, nullptr, ActivationType::Identity, params.num_rows, params.num_columns, params.reduction_size, params.num_experts, params.block_size, true, false, params.stream, {}, {}};
  if (params.tile_rows == 64) {
    genericMoeGemmKernelLauncher<half, cutlass::uint2b_t, half, cutlass::arch::Sm80,
                                 cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
                                 cutlass_extensions::EpilogueOpDefault,
                                 cutlass::gemm::GemmShape<64, 128, 64>,
                                 cutlass::gemm::GemmShape<32, 64, 64>, 3>::call(inputs, params.multiprocessor_count);
  } else {
    genericMoeGemmKernelLauncher<half, cutlass::uint2b_t, half, cutlass::arch::Sm80,
                                 cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
                                 cutlass_extensions::EpilogueOpDefault,
                                 cutlass::gemm::GemmShape<32, 128, 64>,
                                 cutlass::gemm::GemmShape<32, 32, 64>, 3>::call(inputs, params.multiprocessor_count);
  }
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels