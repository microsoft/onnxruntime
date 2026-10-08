// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_int2_prefill.h"
#include "cutlass/epilogue/thread/activation.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_int2.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_activation_kernels.cuh"
#include "core/common/safeint.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
namespace {

struct PrefillWorkspace {
  size_t bytes = 0;
  size_t permuted_rows;
  size_t permuted_experts;
  size_t expert_offsets;
  size_t counts;
  size_t cumulative_counts;
  size_t blocked_rows;
  size_t expanded_input;
  size_t fc1_output;
  size_t activated_output;
  size_t fc2_output;

  size_t Reserve(size_t count, size_t element_size) {
    const size_t offset = bytes;
    const size_t end = SafeInt<size_t>(bytes) + SafeInt<size_t>(count) * element_size;
    bytes = (SafeInt<size_t>(end) + 255) / 256 * 256;
    return offset;
  }

  explicit PrefillWorkspace(const Int2MoePrefillParams& params) {
    const size_t expanded = SafeInt<size_t>(params.num_rows) * params.top_k;
    const int64_t block_rows = computeNumTokensPerBlock(params.num_rows, params.num_experts);
    const size_t blocks = (params.num_rows + block_rows - 1) / block_rows;
    const size_t count_elements = SafeInt<size_t>(params.num_experts) * blocks;
    permuted_rows = Reserve(expanded, sizeof(int));
    permuted_experts = Reserve(expanded, sizeof(int));
    expert_offsets = Reserve(SafeInt<size_t>(params.num_experts) + 1, sizeof(int64_t));
    counts = Reserve(count_elements, sizeof(int));
    cumulative_counts = Reserve(count_elements, sizeof(int));
    blocked_rows = Reserve(SafeInt<size_t>(params.num_rows) * params.num_experts, sizeof(int));
    expanded_input = Reserve(SafeInt<size_t>(expanded) * params.hidden_size, sizeof(half));
    fc1_output = Reserve(SafeInt<size_t>(expanded) * params.inter_size * 2, sizeof(half));
    activated_output = Reserve(SafeInt<size_t>(expanded) * params.inter_size, sizeof(half));
    fc2_output = expanded_input;
  }
};

template <typename ElementType>
void RunPackedGroupedGemm(const Int2MoePrefillParams& params,
                          const ElementType* activations, const uint8_t* weights,
                          const void* scales, const int64_t* offsets, ElementType* output,
                          int64_t num_rows, int num_columns, int reduction_size, int weight_bits) {
  if (weight_bits == 2) {
    Int2GroupedGemmParamsT<ElementType> gemm;
    gemm.activations = activations;
    gemm.packed_weights = weights;
    gemm.block_scales = static_cast<const ElementType*>(scales);
    gemm.expert_row_ends = offsets + 1;
    gemm.output = output;
    gemm.num_rows = num_rows;
    gemm.num_columns = num_columns;
    gemm.reduction_size = reduction_size;
    gemm.block_size = params.block_size;
    gemm.num_experts = params.num_experts;
    gemm.sm = params.sm;
    gemm.multiprocessor_count = params.multiprocessor_count;
    gemm.stream = params.stream;
    RunInt2GroupedGemm(gemm);
    return;
  }
  ORT_ENFORCE(weight_bits == 4, "Packed INT prefill requires INT2 or INT4 weights");
  GroupedGemmInput<ElementType, cutlass::uint4b_t, ElementType, ElementType> gemm{
      activations, offsets + 1, reinterpret_cast<const cutlass::uint4b_t*>(weights), static_cast<const ElementType*>(scales), nullptr, nullptr, output, nullptr, nullptr, ActivationType::Identity, num_rows, num_columns, reduction_size, params.num_experts, params.block_size, true, false, params.stream, {}, {}};
  gemm.gemm_config = cutlass_extensions::CutlassGemmConfig(
      cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
      cutlass_extensions::SplitKStyle::NO_SPLIT_K, 1, 4);
  MoeGemmRunner<ElementType, cutlass::uint4b_t, ElementType> runner(params.sm, params.multiprocessor_count);
  runner.moeGemm(gemm, {});
}

template <typename ElementType>
void RunInt2MoePrefillImpl(const Int2MoePrefillParams& params, void* workspace) {
  const PrefillWorkspace layout(params);
  auto* storage = static_cast<char*>(workspace);
  auto* permuted_rows = reinterpret_cast<int*>(storage + layout.permuted_rows);
  auto* permuted_experts = reinterpret_cast<int*>(storage + layout.permuted_experts);
  auto* offsets = reinterpret_cast<int64_t*>(storage + layout.expert_offsets);
  auto* expanded_input = reinterpret_cast<ElementType*>(storage + layout.expanded_input);
  auto* fc1_output = reinterpret_cast<ElementType*>(storage + layout.fc1_output);
  auto* activated_output = reinterpret_cast<ElementType*>(storage + layout.activated_output);
  auto* fc2_output = reinterpret_cast<ElementType*>(storage + layout.fc2_output);
  const int64_t expanded = params.num_rows * params.top_k;

  threeStepBuildExpertMapsSortFirstToken(
      params.selected_experts, permuted_experts, permuted_rows, params.unpermuted_to_permuted, offsets,
      reinterpret_cast<int*>(storage + layout.counts), reinterpret_cast<int*>(storage + layout.cumulative_counts),
      reinterpret_cast<int*>(storage + layout.blocked_rows), params.num_rows, params.num_experts, params.top_k,
      0, params.stream);
  const QuantParams quant_params{};
  expandInputRowsKernelLauncher<ElementType, ElementType>(
      static_cast<const ElementType*>(params.input), expanded_input, nullptr, nullptr, permuted_rows,
      params.num_rows, params.hidden_size,
      params.top_k, params.num_experts, quant_params, false, offsets, nullptr, nullptr, nullptr, params.stream);

  RunPackedGroupedGemm(params, expanded_input, params.fc1_weights, params.fc1_scales,
                       offsets, fc1_output, expanded, params.inter_size * 2,
                       params.hidden_size, params.fc1_weight_bits);

  ActivationParams activation(params.activation_type);
  activation.alpha = params.alpha;
  activation.beta = params.beta;
  activation.limit = params.limit;
  activation.swiglu_fusion = 1;
  doActivation<ElementType, ElementType, ElementType>(
      activated_output, fc1_output, nullptr, static_cast<const ElementType*>(params.fc1_bias), true, offsets,
      params.num_experts, params.inter_size, expanded, params.activation_type,
      quant_params, false, nullptr, params.stream, activation);

  RunPackedGroupedGemm(params, activated_output, params.fc2_weights, params.fc2_scales,
                       offsets, fc2_output, expanded, params.hidden_size,
                       params.inter_size, params.fc2_weight_bits);
  finalizeMoeRoutingKernelLauncher<ElementType, ElementType, ElementType>(
      fc2_output, static_cast<ElementType*>(params.output), static_cast<const ElementType*>(params.fc2_bias),
      params.routing_weights, params.unpermuted_to_permuted,
      permuted_rows, params.selected_experts, offsets, params.num_rows, params.hidden_size, params.top_k,
      params.num_experts, MOEParallelismConfig{}, false, params.stream);
}

}  // namespace

size_t GetInt2MoePrefillWorkspaceSize(const Int2MoePrefillParams& params) {
  return PrefillWorkspace(params).bytes;
}

void RunInt2MoePrefill(const Int2MoePrefillParams& params, void* workspace) {
  if (params.is_bf16) {
#ifdef ENABLE_BF16
    RunInt2MoePrefillImpl<__nv_bfloat16>(params, workspace);
#else
    ORT_THROW("BF16 INT2 prefill requires ENABLE_BF16");
#endif
    return;
  }
  RunInt2MoePrefillImpl<half>(params, workspace);
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
