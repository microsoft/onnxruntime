// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"

#include <cuda_fp16.h>
#include <type_traits>

// Shared device-side kernels + launch/dispatch helpers (fpA_intB_gemv namespace).
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_device.cuh"
// Include env_var_utils.h after the device header: dispatcher.h (pulled in there) transitively
// includes provider_api.h, which defines SHARED_PROVIDER. That guard suppresses env_var_utils.h's
// own logging.h include and avoids redefining CREATE_MESSAGE / LOGS_CATEGORY (which would fail
// under -Werror).
#include "core/platform/env_var_utils.h"

namespace onnxruntime::llm {
namespace kernels {

namespace moe_gemv {

namespace fiv = onnxruntime::llm::kernels::fpA_intB_gemv;

// CtaN/Threads match the dense per-channel (EnableZero=false) m-tile config.
static constexpr int kCtaN = 8;
static constexpr int kThreads = 128;
// int4 ColumnMajorInterleave (Sm80) tile width along N.
static constexpr int kTileSizeK = 64;
static constexpr int kInt4Interleave = 128 * 8 / (kTileSizeK * 4);  // = 4
static constexpr int kInt8Interleave = 128 * 8 / (kTileSizeK * 8);  // = 2

// Maps a runtime accumulator-precision choice to a compile-time type tag so the
// host launcher can select the fp32- or 16-bit-accumulation kernel instantiation.
template <typename T>
struct TypeTag {
  using type = T;
};

// Opt-in: accumulate the GEMV inner product in fp32 instead of the default fp16
// for fp16 activations. bf16 always accumulates in fp32 because 16-bit bf16
// accumulation is too lossy.
inline bool MoeGemvUseFp32Accum() {
  // Parsed once via ORT's environment helper (consistent parsing/thread-safety across platforms).
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_MOE_GEMV_FP32_ACCUM", 0) == 1;
  return enabled;
}

inline bool MoeGemvUseSplitK2SwiGLU() {
  // Parsed once via ORT's environment helper (consistent parsing/thread-safety across platforms).
  static bool const enabled = [] {
    auto const value = onnxruntime::ParseEnvironmentVariable<int>("ORT_MOE_GEMV_SPLITK2_SWIGLU");
    if (!value.has_value()) {
      return false;
    }
    ORT_ENFORCE(*value == 0 || *value == 1,
                "ORT_MOE_GEMV_SPLITK2_SWIGLU must be 0 or 1, but got ", *value);
    return *value == 1;
  }();
  return enabled;
}

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k,
                           int weight_bits, int group_size) {
  if (sm < 80) {
    return false;
  }
  if (weight_bits != 4 && weight_bits != 8) {
    return false;
  }
  // group_size <= 0 selects the per-column (per-channel) path; block-wise scales must be 32, 64, or 128.
  if (group_size > 0 && group_size != 32 && group_size != 64 && group_size != 128) {
    return false;
  }
  // Keep the first block-wise GEMV implementation on complete K blocks.
  if (group_size > 0 && k % group_size != 0) {
    return false;
  }
  if (expanded_num_rows <= 0 || expanded_num_rows > kMaxProfiledExpandedRows) {
    return false;
  }
  if (n < kMinProfiledProblemDim || k < kMinProfiledProblemDim) {
    return false;
  }
  if (expanded_num_rows > kMaxProfiledExpandedRowsForSmallProblemDim &&
      (n < kMinProfiledProblemDimForExpandedRowsAbove4 || k < kMinProfiledProblemDimForExpandedRowsAbove4)) {
    return false;
  }
  // n must tile evenly; k must tile evenly into StepK along interleaved-K.
  int const interleave = weight_bits == 4 ? kInt4Interleave : kInt8Interleave;
  if (n % (kCtaN * interleave) != 0) {
    return false;
  }
  int64_t const interleaved_k = k * interleave;
  int const step_k = 128 / weight_bits;
  if (interleaved_k % step_k != 0) {
    return false;
  }

  // The interleaved kernel reads K in whole tiles of kTileSizeK (64). Each group of participating
  // threads covers one kTileSizeK-wide K tile via sub-offsets {0, kTileSizeK/2, ...}; the per-thread
  // activation iterator unconditionally loads StepK elements at real_offset_k. When k is not a multiple
  // of kTileSizeK the final partial tile makes the upper-half threads read past the valid k range of the
  // activation row (e.g. k=32 reads act[..32..63]), yielding garbage/NaN. Require complete K tiles.
  if (k % kTileSizeK != 0) {
    return false;
  }

  return true;
}

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k) {
  return is_moe_gemv_supported(sm, expanded_num_rows, n, k, 4, 0);
}

template <typename T, typename WeightType>
struct DetailsForTAndWeight;

template <>
struct DetailsForTAndWeight<half, cutlass::uint4b_t> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
  static constexpr int kWeightBits = 4;
};

template <>
struct DetailsForTAndWeight<half, uint8_t> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int8DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
  static constexpr int kWeightBits = 8;
};

#ifdef ENABLE_BF16
template <>
struct DetailsForTAndWeight<__nv_bfloat16, cutlass::uint4b_t> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
  static constexpr int kWeightBits = 4;
};

template <>
struct DetailsForTAndWeight<__nv_bfloat16, uint8_t> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int8DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
  static constexpr int kWeightBits = 8;
};
#endif

template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric(T const* act, WeightType const* weight, T const* scales, T const* bias, T* out,
                                   int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                                   int num_experts,
                                   int64_t expanded_num_rows, int64_t n, int64_t k, int group_size, int sm,
                                   cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForTAndWeight<T, WeightType>::Details;
  using TypeA = typename DetailsForTAndWeight<T, WeightType>::TypeA;
  // Accumulate fp16 activations in fp16 by default. ORT_MOE_GEMV_FP32_ACCUM=1
  // restores the previous fp32 accumulation path; bf16 always uses fp32.
  bool const use_fp32_accum = !std::is_same_v<T, half> || MoeGemvUseFp32Accum();
  auto launch = [&](auto acc_tag) {
    using AccT = typename decltype(acc_tag)::type;
    fiv::dispatch_moe_gemv_group_size<Details, kCtaN, kThreads, TypeA, AccT>(
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
        const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(weight)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(bias)),
        reinterpret_cast<TypeA*>(out),
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
  };
  if (use_fp32_accum) {
    launch(TypeTag<float>{});
  } else {
    launch(TypeTag<TypeA>{});
  }
}

template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric_interleaved_swiglu(
    T const* act, WeightType const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size, int sm,
    cutlass_kernels::ActivationParams activation_params, void* splitk_partials, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForTAndWeight<T, WeightType>::Details;
  using TypeA = typename DetailsForTAndWeight<T, WeightType>::TypeA;
  // Accumulation policy matches launch_moe_gemv_int_symmetric.
  bool const use_fp32_accum = !std::is_same_v<T, half> || MoeGemvUseFp32Accum();
  if (splitk_partials != nullptr && MoeGemvUseSplitK2SwiGLU()) {
    if constexpr (std::is_same_v<T, half>) {
      auto launch_splitk = [&](auto acc_tag) {
        using AccT = typename decltype(acc_tag)::type;
        using PartialT = AccT;
        fiv::dispatch_moe_gemv_splitk_twopass_swiglu_group_size<Details, kCtaN, kThreads, 2, TypeA, AccT, PartialT>(
            const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
            const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(weight)),
            const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
            const_cast<TypeA*>(reinterpret_cast<TypeA const*>(bias)),
            reinterpret_cast<TypeA*>(out),
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
            activation_params, reinterpret_cast<PartialT*>(splitk_partials), stream);
      };
      if (use_fp32_accum) {
        launch_splitk(TypeTag<float>{});
      } else {
        launch_splitk(TypeTag<TypeA>{});
      }
      return;
    }
  }
  auto launch = [&](auto acc_tag) {
    using AccT = typename decltype(acc_tag)::type;
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<Details, kCtaN, kThreads, TypeA, AccT>(
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
        const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(weight)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(bias)),
        reinterpret_cast<TypeA*>(out),
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
        activation_params, stream);
  };
  if (use_fp32_accum) {
    launch(TypeTag<float>{});
  } else {
    launch(TypeTag<TypeA>{});
  }
}

template <typename T>
void launch_moe_gemv_int4_per_channel(T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
                                      int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                                      int num_experts,
                                      int64_t expanded_num_rows, int64_t n, int64_t k, int sm, cudaStream_t stream) {
  launch_moe_gemv_int_symmetric<T, cutlass::uint4b_t>(
      act, reinterpret_cast<cutlass::uint4b_t const*>(weight), scales, bias, out, expert_first_token_offset,
      permuted_row_to_expert, num_experts, expanded_num_rows, n, k, 0, sm, stream);
}

template <typename T>
void launch_moe_gemv_int4_per_channel_interleaved_swiglu(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int sm,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  launch_moe_gemv_int_symmetric_interleaved_swiglu<T, cutlass::uint4b_t>(
      act, reinterpret_cast<cutlass::uint4b_t const*>(weight), scales, bias, out, expert_first_token_offset,
      permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, 0, sm, activation_params,
      nullptr, stream);
}

template void launch_moe_gemv_int_symmetric<half, cutlass::uint4b_t>(
    half const*, cutlass::uint4b_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric<half, uint8_t>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<half, cutlass::uint4b_t>(
    half const*, cutlass::uint4b_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, void*, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<half, uint8_t>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, void*, cudaStream_t);

template void launch_moe_gemv_int4_per_channel<half>(half const*, uint8_t const*, half const*, half const*, half*,
                                                     int64_t const*, int const*, int, int64_t, int64_t, int64_t,
                                                     int, cudaStream_t);
template void launch_moe_gemv_int4_per_channel_interleaved_swiglu<half>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int, int64_t,
    int64_t, int64_t, int, cutlass_kernels::ActivationParams, cudaStream_t);

#ifdef ENABLE_BF16
template void launch_moe_gemv_int_symmetric<__nv_bfloat16, cutlass::uint4b_t>(
    __nv_bfloat16 const*, cutlass::uint4b_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric<__nv_bfloat16, uint8_t>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<__nv_bfloat16, cutlass::uint4b_t>(
    __nv_bfloat16 const*, cutlass::uint4b_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    void*, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<__nv_bfloat16, uint8_t>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    void*, cudaStream_t);

template void launch_moe_gemv_int4_per_channel<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, cudaStream_t);
template void launch_moe_gemv_int4_per_channel_interleaved_swiglu<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, cutlass_kernels::ActivationParams,
    cudaStream_t);
#endif
}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
