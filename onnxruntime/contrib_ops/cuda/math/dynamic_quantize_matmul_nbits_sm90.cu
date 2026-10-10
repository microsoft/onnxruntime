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
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
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

// Grouped-scale INT8 WGMMA GEMM for weight block 128 (W2/W4 expanded to signed INT8, or W8 with block 128).
// The K tile equals the scale block: every tile accumulates into a fresh INT32 accumulator that is promoted
// into the FP32 running sum as sum += (float(acc) * a_scale) * b_scale, in K order and without FMA, so the
// result is the operator's exact interval semantics. One producer warpgroup issues TMA for A/B and gathers the
// [N, G] scales, and BM / 64 consumer warpgroups run the WGMMAs. There is no split-K (it would reassociate).
// The running sum starts at -0.0f, so the first term is added without a special case ((-0.0f) + x == x).
template <typename T>
__device__ __forceinline__ float GroupedToFloat(T value) {
  return float(value);
}
__device__ __forceinline__ float GroupedToFloat(__half value) { return __half2float(value); }
__device__ __forceinline__ float GroupedToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }
__device__ __forceinline__ __half2 GroupedPack(__half*, float x, float y) { return __floats2half2_rn(x, y); }
__device__ __forceinline__ __nv_bfloat162 GroupedPack(__nv_bfloat16*, float x, float y) {
  return __floats2bfloat162_rn(x, y);
}

template <int BM_, int BN_, int ST_, class SbT_, class OutT_, int MINB_>
struct GroupedCfg {
  static constexpr int BM = BM_, BN = BN_, ST = ST_, BK = 128;
  static constexpr int MINB = MINB_;
  static constexpr int NWG = BM / 64;
  static constexpr int NTHREADS = (NWG + 1) * 128;
  using SbT = SbT_;
  using OutT = OutT_;
  static constexpr bool kStrict = !cute::is_same_v<OutT_, __half>;
  static constexpr bool kRegRealloc = (MINB_ == 1);

  using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{},
                                             make_shape(Int<BM>{}, Int<BK>{}, Int<ST>{})));
  using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<int8_t>{},
                                             make_shape(Int<BN>{}, Int<BK>{}, Int<ST>{})));
  // Per-stage scale rows are padded to 128 bytes.
  static constexpr int SBS = int(((size_t(BN) * sizeof(SbT_) + 127) / 128) * 128 / sizeof(SbT_));
  using MmaOp = decltype(GMMA::ss_op_selector<int8_t, int8_t, int32_t, Shape<Int<64>, Int<BN>, _32>>());
  using TiledMma = decltype(make_tiled_mma(MmaOp{}, Layout<Shape<Int<NWG>, _1, _1>>{}));

  static constexpr size_t kBytesA = size_t(BM) * BK * ST;
  static constexpr size_t kBytesB = size_t(BN) * BK * ST;
  static constexpr size_t kBytesS = size_t(SBS) * sizeof(SbT) * ST;
  static constexpr int ESTR = BN + 8;  // epilogue staging row stride; 16 bytes of padding avoids bank conflicts
  static constexpr size_t kBytesE = size_t(NWG) * 64 * ESTR * 2;
  static constexpr size_t kOffE = kBytesA + kBytesB + ((kBytesS + 127) / 128) * 128;
  static constexpr size_t kSmemData = kOffE + kBytesE;
  static constexpr size_t kSmem = 1024 + kSmemData + 2 * ST * 8 + 64;
  static constexpr uint32_t kTxBytes = uint32_t(BM * BK + BN * BK);
};

template <class C, class TmaA, class TmaB>
__global__ void __launch_bounds__(C::NTHREADS, C::MINB)
    GroupedInt8Kernel(__grid_constant__ TmaA const tma_a, __grid_constant__ TmaB const tma_b,
                      const float* __restrict__ sa, bool per_tensor,
                      const typename C::SbT* __restrict__ sb,  // [N, G]
                      const typename C::OutT* __restrict__ bias, typename C::OutT* __restrict__ Y,
                      int M, int N, int K) {
  constexpr int BM = C::BM, BN = C::BN, ST = C::ST, NWG = C::NWG;
  using SbT = typename C::SbT;
  using OutT = typename C::OutT;
  const int G = K / 128;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  int8_t* smemA = reinterpret_cast<int8_t*>(base);
  int8_t* smemB = reinterpret_cast<int8_t*>(base + C::kBytesA);
  SbT* smemS = reinterpret_cast<SbT*>(base + C::kBytesA + C::kBytesB);
  OutT* smemE = reinterpret_cast<OutT*>(base + C::kOffE);
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(base + C::kSmemData);
  uint64_t* empty_bar = full_bar + ST;

  auto sA = make_tensor(make_smem_ptr(smemA), typename C::SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(smemB), typename C::SmemLayoutB{});

  const int wg = threadIdx.x / 128;
  const int warp_in_wg = (threadIdx.x / 32) % 4;
  const int lane = threadIdx.x % 32;

  using FullBar = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBar = cutlass::arch::ClusterBarrier;

  if (threadIdx.x == 0) {
    cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
    for (int s = 0; s < ST; ++s) {
      FullBar::init(&full_bar[s], 2);  // the TMA thread and the scale-gather warp
      EmptyBar::init(&empty_bar[s], NWG * 4);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  if (wg == 0) {
    if constexpr (C::kRegRealloc) cutlass::arch::warpgroup_reg_dealloc<40>();
    if (warp_in_wg == 0 && cute::elect_one_sync()) {
      auto mA = tma_a.get_tma_tensor(make_shape(M, K));
      auto mB = tma_b.get_tma_tensor(make_shape(N, K));
      auto gA = local_tile(mA, make_shape(Int<BM>{}, Int<128>{}), make_coord(blockIdx.y, _));
      auto gB = local_tile(mB, make_shape(Int<BN>{}, Int<128>{}), make_coord(blockIdx.x, _));
      auto tA = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));
      auto tB = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));
      auto tAgA = get<0>(tA);
      auto tAsA = get<1>(tA);
      auto tBgB = get<0>(tB);
      auto tBsB = get<1>(tB);
      for (int kt = 0; kt < G; ++kt) {
        const int s = kt % ST;
        const uint32_t ph = (kt / ST) & 1;
        EmptyBar::wait(&empty_bar[s], ph ^ 1);
        FullBar::arrive_and_expect_tx(&full_bar[s], C::kTxBytes);
        copy(tma_a.with(full_bar[s]), tAgA(_, kt), tAsA(_, s));
        copy(tma_b.with(full_bar[s]), tBgB(_, kt), tBsB(_, s));
      }
    }
    if (warp_in_wg == 1) {
      for (int kt = 0; kt < G; ++kt) {
        const int s = kt % ST;
        const uint32_t ph = (kt / ST) & 1;
        EmptyBar::wait(&empty_bar[s], ph ^ 1);
#pragma unroll
        for (int i = 0; i < (BN + 31) / 32; ++i) {
          const int nl = i * 32 + lane;
          const int n = n0 + nl;
          if (nl < BN) smemS[s * C::SBS + nl] = (n < N) ? sb[size_t(n) * G + kt] : SbT(0.f);
        }
        __syncwarp();
        if (lane == 0) FullBar::arrive(&full_bar[s]);
      }
    }
  } else {
    if constexpr (C::kRegRealloc) cutlass::arch::warpgroup_reg_alloc<232>();
    const int ctid = threadIdx.x - 128;
    typename C::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(ctid);
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);
    auto tCrA = thr_mma.make_fragment_A(tCsA);
    auto tCrB = thr_mma.make_fragment_B(tCsB);
    // Two INT32 accumulators: the MMA of tile kt + 1 overlaps the FP32 promotion of tile kt.
    auto acc0 = partition_fragment_C(tiled_mma, Shape<Int<BM>, Int<BN>>{});
    auto acc1 = partition_fragment_C(tiled_mma, Shape<Int<BM>, Int<BN>>{});
    constexpr int NACC = BN / 2;
    static_assert(decltype(size(acc0))::value == NACC, "unexpected accumulator fragment size");
    float sum[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) sum[i] = -0.0f;

    // Accumulator layout of wgmma m64nNk32 per thread: index i = 4 * j + 2 * h + e,
    // row = 16 * warp + lane / 4 + 8 * h, column = 8 * j + 2 * (lane % 4) + e.
    const int wg_c = ctid / 128;
    const int row_base = wg_c * 64 + warp_in_wg * 16 + lane / 4;
    const int col_base = 2 * (lane % 4);
    float sa_r[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = m0 + row_base + 8 * h;
      sa_r[h] = per_tensor ? sa[0] : (r < M ? sa[r] : 0.f);
    }
    // BF16 scales can leave the FP32 normal range: such rows take the contract's FP64 rule.
    const bool plain = !C::kStrict || (PlainScaleProduct(sa_r[0]) && PlainScaleProduct(sa_r[1]));

    auto issue = [&](int kt, auto& acc) {
      const int s = kt % ST;
      const uint32_t ph = (kt / ST) & 1;
      FullBar::wait(&full_bar[s], ph);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        cute::gemm(tiled_mma, tCrA(_, _, k, s), tCrB(_, _, k, s), acc);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_fence_operand(acc);
    };
    auto promote = [&](int kt, auto& acc) {
      const int s = kt % ST;
      warpgroup_fence_operand(acc);
      float sbv[BN / 4];
      const SbT* sp = smemS + s * C::SBS + col_base;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        sbv[2 * j] = GroupedToFloat(sp[8 * j]);
        sbv[2 * j + 1] = GroupedToFloat(sp[8 * j + 1]);
      }
      // The MMA has completed and the scales are in registers, so this stage can be refilled.
      __syncwarp();
      if (lane == 0) EmptyBar::arrive(&empty_bar[s]);
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int i = 4 * j + 2 * h + e;
            const float f = __int2float_rn(acc(i));
            float term;
            if (plain) {
              term = __fmul_rn(__fmul_rn(f, sa_r[h]), sbv[2 * j + e]);
            } else {
              term = StrictScaleProductSlow(f, sa_r[h], sbv[2 * j + e]);
            }
            sum[i] = __fadd_rn(sum[i], term);
          }
        }
      }
    };

    // All waits are compile-time constants in straight-line code so ptxas can prove which accumulator is
    // in flight; otherwise it serializes the WGMMAs (warning C7514).
    issue(0, acc0);
    int kt = 0;
    for (; kt + 2 < G; kt += 2) {
      issue(kt + 1, acc1);
      warpgroup_wait<1>();
      promote(kt, acc0);
      issue(kt + 2, acc0);
      warpgroup_wait<1>();
      promote(kt + 1, acc1);
    }
    if (kt + 1 < G) {
      issue(kt + 1, acc1);
      warpgroup_wait<1>();
      promote(kt, acc0);
      warpgroup_wait<0>();
      promote(kt + 1, acc1);
    } else {
      warpgroup_wait<0>();
      promote(kt, acc0);
    }

    // y = convert(sum + bias) staged through shared memory for coalesced 16-byte stores.
    OutT* e = smemE + wg_c * 64 * C::ESTR;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int n = n0 + 8 * j + col_base;
      float b0 = 0.f, b1 = 0.f;
      if (bias != nullptr) {
        if (n < N) b0 = GroupedToFloat(bias[n]);
        if (n + 1 < N) b1 = GroupedToFloat(bias[n + 1]);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float y0 = __fadd_rn(sum[4 * j + 2 * h], b0);
        const float y1 = __fadd_rn(sum[4 * j + 2 * h + 1], b1);
        const int rl = warp_in_wg * 16 + lane / 4 + 8 * h;
        *reinterpret_cast<decltype(GroupedPack(static_cast<OutT*>(nullptr), 0.f, 0.f))*>(
            e + rl * C::ESTR + 8 * j + col_base) = GroupedPack(static_cast<OutT*>(nullptr), y0, y1);
      }
    }
    cutlass::arch::NamedBarrier::sync(128, 1 + wg_c);
    constexpr int CPR = BN / 8;
    for (int idx = ctid % 128; idx < 64 * CPR; idx += 128) {
      const int rl = idx / CPR, c = idx % CPR;
      const int r = m0 + wg_c * 64 + rl;
      const int n = n0 + 8 * c;
      if (r < M && n < N) {  // N % 8 == 0, so a chunk is entirely in or out
        *reinterpret_cast<uint4*>(Y + size_t(r) * N + n) = *reinterpret_cast<const uint4*>(e + rl * C::ESTR + 8 * c);
      }
    }
  }
}

template <int BM, int BN, int ST, int MINB, class SbT, class OutT>
Status RunGrouped(const int8_t* a, const float* a_scales, const int8_t* b, const void* b_scales,
                  const void* bias, void* y, bool per_tensor, int rows, int n, int k, cudaStream_t stream) {
  using C = GroupedCfg<BM, BN, ST, SbT, OutT, MINB>;
  auto mA = make_tensor(a, make_shape(rows, k), make_stride(int64_t(k), _1{}));
  auto mB = make_tensor(b, make_shape(n, k), make_stride(int64_t(k), _1{}));
  auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, mA, typename C::SmemLayoutA{}(_, _, 0), make_shape(Int<BM>{}, Int<128>{}));
  auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, mB, typename C::SmemLayoutB{}(_, _, 0), make_shape(Int<BN>{}, Int<128>{}));
  using TmaA = decltype(tma_a);
  using TmaB = decltype(tma_b);
  auto kernel = &GroupedInt8Kernel<C, TmaA, TmaB>;
  CUDA_RETURN_IF_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(C::kSmem)));
  const dim3 grid((n + BN - 1) / BN, (rows + BM - 1) / BM, 1);
  kernel<<<grid, C::NTHREADS, C::kSmem, stream>>>(
      tma_a, tma_b, a_scales, per_tensor, static_cast<const SbT*>(b_scales), static_cast<const OutT*>(bias),
      static_cast<OutT*>(y), rows, n, k);
  return CUDA_CALL(cudaGetLastError());
}

template <class SbT, class OutT>
Status DispatchGrouped(const int8_t* a, const float* a_scales, const int8_t* b, const void* b_scales,
                       const void* bias, void* y, bool per_tensor, int rows, int n, int k, cudaStream_t stream) {
  // Tile choice from the measured H200 sweep: narrow tiles for few rows or columns, one wave at M=1536, N=1024.
  if (rows <= 128 || n <= 512) {
    return RunGrouped<64, 32, 4, 2, SbT, OutT>(a, a_scales, b, b_scales, bias, y, per_tensor, rows, n, k, stream);
  }
  if (n <= 1024) {
    return RunGrouped<128, 96, 4, 1, SbT, OutT>(a, a_scales, b, b_scales, bias, y, per_tensor, rows, n, k, stream);
  }
  return RunGrouped<128, 128, 4, 1, SbT, OutT>(a, a_scales, b, b_scales, bias, y, per_tensor, rows, n, k, stream);
}
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
Status LaunchDynamicQuantizedSm90Grouped(const int8_t* a, const float* a_scales, const int8_t* b,
                                         const void* b_scales, DynamicWeightScaleType scale_type,
                                         const void* bias, void* y, bool bf16, bool per_tensor, int rows, int n,
                                         int k, cudaStream_t stream) {
#if defined(ORT_DYNAMIC_QUANTIZED_SM90_IMPL)
#define DISPATCH_GROUPED(SbT, OutT) \
  return DispatchGrouped<SbT, OutT>(a, a_scales, b, b_scales, bias, y, per_tensor, rows, n, k, stream)
  switch (scale_type) {
    case DynamicWeightScaleType::Float:
      if (bf16) DISPATCH_GROUPED(float, __nv_bfloat16);
      DISPATCH_GROUPED(float, __half);
    case DynamicWeightScaleType::Half:
      if (bf16) DISPATCH_GROUPED(__half, __nv_bfloat16);
      DISPATCH_GROUPED(__half, __half);
    case DynamicWeightScaleType::BFloat16:
      if (bf16) DISPATCH_GROUPED(__nv_bfloat16, __nv_bfloat16);
      DISPATCH_GROUPED(__nv_bfloat16, __half);
  }
#undef DISPATCH_GROUPED
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
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "DynamicQuantizeMatMulNBits SM90 grouped GEMM was not built.");
#endif
}
}  // namespace onnxruntime::contrib::cuda
