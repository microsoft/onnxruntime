// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dynamic_quantize_matmul_nbits.h"

#include <cfloat>

#if defined(ORT_DYNAMIC_QUANTIZED_SM90_IMPL)
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#endif

namespace onnxruntime::contrib::cuda {
#if defined(ORT_DYNAMIC_QUANTIZED_SM90_IMPL)
namespace {
using namespace cute;
namespace fusion = cutlass::epilogue::fusion;

// BF16 activation scales can make INT32-accumulator * scale leave the FP32 normal range, so the
// first product follows the operator contract: non-normal results are formed in FP64 from the inputs.
// For 2^-126 <= |a| < 2^96 any nonzero accumulator (|acc| <= 2^31) gives a normal finite product, and a
// zero accumulator gives zero, so one integer range test on the scale selects the plain two-multiply path.
__device__ __forceinline__ bool PlainScaleProduct(float a) {
  const uint32_t magnitude = __float_as_uint(a) & 0x7fffffffu;
  return magnitude - 0x00800000u < 0x6f800000u - 0x00800000u;
}

__device__ __noinline__ float StrictScaleProductSlow(float value, float a, float b) {
  if (isnan(a)) return a;
  const float first = __fmul_rn(value, a);
  if (value == 0.0f || (isfinite(first) && fabsf(first) >= FLT_MIN)) return __fmul_rn(first, b);
  return __double2float_rn(__dmul_rn(__dmul_rn(static_cast<double>(value), static_cast<double>(a)),
                                     static_cast<double>(b)));
}

template <class T>
struct StrictScaleProductOp {};

template <int N>
struct StrictScaleProductOp<cutlass::Array<float, N>> {
  CUTLASS_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& acc,
                                                     cutlass::Array<float, N> const& a,
                                                     cutlass::Array<float, N> const& b) const {
    cutlass::Array<float, N> result;
    bool plain = true;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) plain &= PlainScaleProduct(a[i]);
    if (__builtin_expect(plain, 1)) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) result[i] = __fmul_rn(__fmul_rn(acc[i], a[i]), b[i]);
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) result[i] = StrictScaleProductSlow(acc[i], a[i], b[i]);
    }
    return result;
  }
};

template <int TileM, typename Scale, typename D>
struct DynamicSm90Gemm {
  using Tile = Shape<Int<TileM>, _128, _128>;
  using Cluster = Shape<_2, _1, _1>;
  using StrideD = Stride<int64_t, _1, _0>;
  static constexpr auto Round = cutlass::FloatRoundStyle::round_to_nearest;
  using ScaleA = fusion::Sm90ColBroadcast<0, Tile, float, float, Stride<bool, _0, _0>>;
  using ScaleB = fusion::Sm90RowBroadcast<0, Tile, Scale, float>;
  using Bias = fusion::Sm90RowBroadcast<0, Tile, D, float>;
  // FP16 activations: the first product is always a normal FP32 value, so two plain multiplies are exact.
  using First = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, float, float, Round>,
                                fusion::Sm90AccFetch, ScaleA>;
  using Second = cute::conditional_t<
      cute::is_same_v<D, cutlass::half_t>,
      fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, float, float, Round>, First, ScaleB>,
      fusion::Sm90EVT<fusion::Sm90Compute<StrictScaleProductOp, float, float, Round>,
                      fusion::Sm90AccFetch, ScaleA, ScaleB>>;
  using EpilogueTree = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::plus, D, float, Round>, Second, Bias>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      int32_t, float, void, StrideD, 8, D, StrideD, 8,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueTree>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16,
      int32_t, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static Status Run(const int8_t* a, const float* a_scales, const int8_t* b, const void* b_scales,
                    const void* bias, void* y, bool per_tensor, int rows, int n, int k,
                    int64_t b_stride, int device_id, int sm_count, cudaStream_t stream) {
    typename ScaleA::Arguments a_args{a_scales, 1.0f, {!per_tensor, _0{}, _0{}}};
    typename ScaleB::Arguments b_args{static_cast<const Scale*>(b_scales)};
    typename Bias::Arguments bias_args{static_cast<const D*>(bias)};
    typename Second::Arguments second = [&] {
      if constexpr (cute::is_same_v<D, cutlass::half_t>) {
        return typename Second::Arguments{typename First::Arguments{{}, a_args, {}}, b_args, {}};
      } else {
        return typename Second::Arguments{{}, a_args, b_args, {}};
      }
    }();
    typename EpilogueTree::Arguments epilogue{second, bias_args, {}};
    cutlass::KernelHardwareInfo hardware;
    hardware.device_id = device_id;
    hardware.sm_count = sm_count;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {rows, n, k, 1}, {a, {int64_t{k}, _1{}, _0{}}, b, {b_stride, _1{}, _0{}}}, {epilogue, nullptr, {int64_t{n}, _1{}, _0{}}, static_cast<D*>(y), {int64_t{n}, _1{}, _0{}}}, hardware};
    Gemm gemm;
    auto check = [](cutlass::Status status) {
      return status == cutlass::Status::kSuccess
                 ? Status::OK()
                 : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DynamicQuantizeMatMulNBits SM90 GEMM: ",
                                   cutlassGetStatusString(status));
    };
    ORT_RETURN_IF_ERROR(check(gemm.can_implement(args)));
    ORT_RETURN_IF_NOT(gemm.get_workspace_size(args) == 0,
                      "DynamicQuantizeMatMulNBits SM90 GEMM unexpectedly requires workspace.");
    // Per-call stack parameters bind current scratch/output pointers without a shared mutable cache.
    ORT_RETURN_IF_ERROR(check(gemm.initialize(args, nullptr, stream)));
    ORT_RETURN_IF_ERROR(check(gemm.run(stream)));
    return CUDA_CALL(cudaGetLastError());
  }
};
}  // namespace
#endif

Status DynamicQuantizedSm90Available(bool& available) {
  available = false;
#if defined(ORT_DYNAMIC_QUANTIZED_SM90_IMPL)
  cudaFuncAttributes attributes{};
  const auto error = cudaFuncGetAttributes(
      &attributes, cutlass::device_kernel<typename DynamicSm90Gemm<128, float, cutlass::half_t>::Kernel>);
  if (error == cudaErrorNoKernelImageForDevice || error == cudaErrorInvalidDeviceFunction) {
    (void)cudaGetLastError();
    return Status::OK();
  }
  CUDA_RETURN_IF_ERROR(error);
  available = attributes.binaryVersion == 90 && attributes.ptxVersion >= 90;
#endif
  return Status::OK();
}

Status LaunchDynamicQuantizedSm90(const int8_t* a, const float* a_scales, const int8_t* b,
                                  const void* b_scales, DynamicWeightScaleType scale_type,
                                  const void* bias, void* y, bool bf16, bool per_tensor, int rows, int n, int k,
                                  int64_t b_stride, int device_id, int sm_count, cudaStream_t stream) {
#if defined(ORT_DYNAMIC_QUANTIZED_SM90_IMPL)
  const bool large_tile = rows > 128 && n > 2048;
#define LAUNCH_SM90_TYPE(Scale, D)                                                                           \
  return large_tile ? DynamicSm90Gemm<128, Scale, D>::Run(a, a_scales, b, b_scales, bias, y, per_tensor,     \
                                                          rows, n, k, b_stride, device_id, sm_count, stream) \
                    : DynamicSm90Gemm<64, Scale, D>::Run(a, a_scales, b, b_scales, bias, y, per_tensor,      \
                                                         rows, n, k, b_stride, device_id, sm_count, stream)
#define LAUNCH_SM90(Scale)                                \
  if (bf16) LAUNCH_SM90_TYPE(Scale, cutlass::bfloat16_t); \
  LAUNCH_SM90_TYPE(Scale, cutlass::half_t)
  switch (scale_type) {
    case DynamicWeightScaleType::Float:
      LAUNCH_SM90(float);
    case DynamicWeightScaleType::Half:
      LAUNCH_SM90(cutlass::half_t);
    case DynamicWeightScaleType::BFloat16:
      LAUNCH_SM90(cutlass::bfloat16_t);
  }
#undef LAUNCH_SM90
#undef LAUNCH_SM90_TYPE
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid SM90 weight scale type.");
#else
  ORT_UNUSED_PARAMETER(a);
  ORT_UNUSED_PARAMETER(a_scales);
  ORT_UNUSED_PARAMETER(b);
  ORT_UNUSED_PARAMETER(b_scales);
  ORT_UNUSED_PARAMETER(scale_type);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(bf16);
  ORT_UNUSED_PARAMETER(per_tensor);
  ORT_UNUSED_PARAMETER(rows);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(b_stride);
  ORT_UNUSED_PARAMETER(device_id);
  ORT_UNUSED_PARAMETER(sm_count);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "DynamicQuantizeMatMulNBits SM90 GEMM was not built.");
#endif
}
}  // namespace onnxruntime::contrib::cuda
