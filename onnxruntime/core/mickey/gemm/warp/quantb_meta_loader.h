/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file warp/quantb_meta_loader.h
 * @brief Load quantization scales and offsets from global memory to fragments.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"

#include "int_util.h"

namespace mickey {
namespace gemm {
namespace warp {

namespace detail {

/**
 * @brief Convert (4b weights - 8) to fp16 using bits operations.
*/
CUTLASS_DEVICE
void weightsMinuEight2Half(uint32_t const &weights,
                  cutlass::Array<cutlass::half_t, 8>& dest)
{
  // 4b weights are arranged as [0, 2, 4, 6, 1, 3, 5, 7], so that adjacent
  // weights are in adjacent 16b half words.
  //   w & 0x000f000f --> take out element 0, 1
  //   w & 0x00f000f0 --> take out element 2, 3
  //   (w >> 8) & 0x000f000f --> take out element 4, 5
  //   (w >> 8) & 0x00f000f0 --> take out element 6, 7
  //
  // For element 0, 1, 4, 5, we have 0x000?000?, set the high bits
  // to 0x6400, essentially we set the exponent bits to 25, effective
  // exp = 25 - 15 = 10, with explicity hight bit, the value is
  //   2^10 + q_w.
  //
  // Similarly for element 2, 3, 6, 7, we have 0x00?000?, set the
  // high bits to 0x5400, essentially we set the exponent bits to 21,
  // effective exp = 21 - 15 = 6, with explicity hight bit, the value
  // is 2^6 + q_w.
  //
  // 1.125 instruction per weight, 9 instructions in total.

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 500))
  uint32_t*      b32s   = reinterpret_cast<uint32_t*>(dest.data());
  const uint32_t high_8s = weights >> 8;
  asm volatile(
    "  lop3.b32      %0, %4, 0x000f000f, %6, 0xea;\n"
    "  lop3.b32      %1, %4, 0x00f000f0, %7, 0xea;\n"
    "  lop3.b32      %2, %5, 0x000f000f, %6, 0xea;\n"
    "  lop3.b32      %3, %5, 0x00f000f0, %7, 0xea;\n"
    "  sub.rn.f16x2  %0, %0, %10;\n"         // q_w - 1032.0
    "  fma.rn.f16x2  %1, %1, %8, %9;\n"     // 1.0 * q_w + (-72.0)
    "  sub.rn.f16x2  %2, %2, %10;\n"
    "  fma.rn.f16x2  %3, %3, %8, %9;\n"
    : "=r"(b32s[0]), "=r"(b32s[1]), "=r"(b32s[2]), "=r"(b32s[3])
    : "r"(weights), "r"(high_8s),
      "r"(0x64006400), "r"(0x54005400)
      "r"(0x3c003c00), "r"(0xd480d480),
      "r"(0x64086408));
#else
  assert(false);
  (void)(weights);
  (void)(dest);
#endif
}

/**
 * @brief Convert 4b weights to fp16 using bits operations.
*/
CUTLASS_DEVICE
void weights2Half([[maybe_unused]] uint32_t const &weights,
                  [[maybe_unused]] cutlass::Array<cutlass::half_t, 8>& dest)
{
  // 4b weights are arranged as [0, 2, 4, 6, 1, 3, 5, 7], so that adjacent
  // weights are in adjacent 16b half words.
  //   w & 0x000f000f --> take out element 0, 1
  //   w & 0x00f000f0 --> take out element 2, 3
  //   (w >> 8) & 0x000f000f --> take out element 4, 5
  //   (w >> 8) & 0x00f000f0 --> take out element 6, 7
  //
  // For element 0, 1, 4, 5, we have 0x000?000?, set the high bits
  // to 0x6400, essentially we set the exponent bits to 25, effective
  // exp = 25 - 15 = 10, with explicity hight bit, the value is
  //   2^10 + q_w.
  //
  // Similarly for element 2, 3, 6, 7, we have 0x00?000?, set the
  // high bits to 0x5400, essentially we set the exponent bits to 21,
  // effective exp = 21 - 15 = 6, with explicity hight bit, the value
  // is 2^6 + q_w.
  //
  // 1.125 instruction per weight, 9 instructions in total.

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 500))
  uint32_t*      b32s   = reinterpret_cast<uint32_t*>(dest.data());
  const uint32_t high_8s = weights >> 8;

  asm volatile(
    "  lop3.b32      %0, %4, 0x000f000f, %6, 0xea;\n"
    "  lop3.b32      %1, %4, 0x00f000f0, %7, 0xea;\n"
    "  lop3.b32      %2, %5, 0x000f000f, %6, 0xea;\n"
    "  lop3.b32      %3, %5, 0x00f000f0, %7, 0xea;\n"
    "  sub.rn.f16x2  %0, %0, %6;\n"         // q_w - 1024.0
    "  fma.rn.f16x2  %1, %1, %8, %9;\n"     // 1.0 * q_w + (-64.0)
    "  sub.rn.f16x2  %2, %2, %6;\n"
    "  fma.rn.f16x2  %3, %3, %8, %9;\n"
    : "=r"(b32s[0]), "=r"(b32s[1]), "=r"(b32s[2]), "=r"(b32s[3])
    : "r"(weights), "r"(high_8s),
      "r"(0x64006400), "r"(0x54005400)
      "r"(0x3c003c00), "r"(0xd400d400));
#else
  assert(false);
#endif
}

}  // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Loader for blockwise quantization scales
template <
    typename QuantBlocking_,  ///! Shape of the quant block (concept: MatrixShape)
    typename WarpShape_,      ///! Shape of the warp tile (concept: GemmShape kM ignored)
    typename ElementT_ = cutlass::half_t,  ///! Data type of the scales and dequantized B
    bool has_offsets = false,  ///! Whether the quantization has offsets
    bool DebugPrint = false>
struct QuantBScaleLoader;

/// Specialization for column-wise quantization, i.e. QuantBlocking::kColumn == 1
template <
    int block_size_,
    typename WarpShape_,
    typename ElementT_,
    bool has_offsets,
    bool DebugPrint>
struct QuantBScaleLoader<cutlass::MatrixShape<block_size_, 1>, WarpShape_, ElementT_, has_offsets, DebugPrint> {
  //
  // Type definitions
  //
  using QuantBlocking = cutlass::MatrixShape<block_size_, 1>;
  using WarpShape = WarpShape_;
  using ElementT = ElementT_;
  using OffsetT = uint8_t;

  static_assert((WarpShape::kN % 16) == 0 && (WarpShape::kK % 16) == 0,
                "Warp tile size must be multiple of 16x16, the unit of packed weights.");
  static_assert(sizeof(ElementT) == 2, "Quantization only supports 16-bit float types");

  //
  // Column-wise blocking --> kColumn == 1, every column has its own
  // scale/offset, there are far less rows than columns in a warp tile.
  // So we use row-major layout to maximize continuous memory access in
  // a warp.
  //
  // Warp thread layout: As dictated by 16b tensor core layout, 32
  // threads in a warp is divided int 8 groups of 4 threads, each group
  // is responsible for a column, and each thread is responsible for 2
  // rows, forming a 8x8 tile.
  //

  // Number of continuous elements of scale/offset that a warp need to load.
  static constexpr int kMetaFragSize = WarpShape::kN / 8;
  static constexpr int kMetaChunkCount = div_up(WarpShape::kK, QuantBlocking::kRow);

  // HBM -> SMEM, 16 bytes per load, no leftover since WarpShape::kN is multiple of 16
  static constexpr int kSmemSize = WarpShape::kN * kMetaChunkCount;
  static constexpr int kScaleLoadThreads = (kSmemSize * sizeof(ElementT)) / 16;
  static_assert(kScaleLoadThreads <= 16); // shape up to 64x64, 16 threads can load all scales

  using FragmentScales = cutlass::Array<ElementT, kMetaFragSize * kMetaChunkCount>;

  static constexpr int kOffsetLoadThreads = (kSmemSize * sizeof(OffsetT)) / 16;
  static_assert(kOffsetLoadThreads <= 16); // shape up to 64x64, 16 threads can load all offsets

  using FragmentOffsets = typename std::conditional<has_offsets,
                                           FragmentScales,
                                           std::monostate>::type;

  //
  // Data members
  //
  const int n_cnt;

  const uint8_t * const scales_byte_p;
  const int scales_byte_stride;
  const uint8_t * const offsets_byte_p;
  const int offsets_byte_stride;

  //
  // Methods
  //
  template <typename T>
  CUTLASS_DEVICE
  static const uint8_t* get_scales_p(const void* ptr_scales, int scales_byte_stride, int k, int n) {
    return (ptr_scales == nullptr) ? nullptr :
        reinterpret_cast<uint8_t const*>(ptr_scales) + k * scales_byte_stride + n * sizeof(T);
  }

  /// Initializes the scale loader, pointing to the start of the scales tensor
  CUTLASS_DEVICE
  QuantBScaleLoader(
      int lane_idx,
      void const *ptr_scales,
      int scales_byte_stride,
      void const *ptr_offsets,  // dummy to make the interface consistent with QuantBScaleOffsetLoader
      int offsets_byte_stride,
      int start_n,
      int end_n)
      : n_cnt(end_n - start_n),
        scales_byte_p(get_scales_p<ElementT>(ptr_scales, scales_byte_stride, 0, start_n)),
        scales_byte_stride(scales_byte_stride),
        offsets_byte_p(get_scales_p<OffsetT>(ptr_offsets, offsets_byte_stride, 0, start_n)),
        offsets_byte_stride(offsets_byte_stride)
  {
    assert(ptr_scales != nullptr);
    assert(scales_byte_stride > 0 && mod_power2<16>(scales_byte_stride) == 0);
    assert(scales_byte_stride >= end_n * sizeof(ElementT));
    if constexpr(has_offsets) {
      assert(ptr_offsets != nullptr);
      assert(offsets_byte_stride > 0 && mod_power2<16>(offsets_byte_stride) == 0);
      assert(offsets_byte_stride >= end_n * sizeof(OffsetT));
    } else {
      assert(ptr_offsets == nullptr);
      assert(offsets_byte_stride == 0);
    }
  }

  /// Loads [start_k, end_k) x [start_n, end_n) scales from global memory to fragment
  /// [start_n, end_n) was specified in the constructor
  CUTLASS_DEVICE
  void load_to_smem(const int lane_idx, const int start_k, const int k_cnt, ElementT* smem, OffsetT* offset_smem) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    { // Load scales to smem
      int lane_ptr_offset = mul_power2<16 / sizeof(ElementT)>(lane_idx);

      // Column-wise quantization, every column has its own scale/offset
      const uint8_t* scales_ptr = scales_byte_p + (div_power2<QuantBlocking::kRow>(start_k)) * scales_byte_stride;
      const int k_loads = div_up(k_cnt, QuantBlocking::kRow);
      const int k_idx = div_power2<WarpShape::kN>(lane_ptr_offset);
      const int n_idx = mod_power2<WarpShape::kN>(lane_ptr_offset);

      unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(&smem[lane_ptr_offset]);
      int src_in_bytes = ((k_idx < k_loads && n_idx < n_cnt) ? 16 : 0);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
          "}\n"
          ::"r"((int)(kScaleLoadThreads > lane_idx)),
            "r"(smem_int_ptr),
            "l"(&scales_ptr[k_idx * scales_byte_stride + n_idx * sizeof(ElementT)]),
            "n"(16), "r"(src_in_bytes));
    }
    if constexpr(has_offsets) { // Load offset to smem
      int lane_ptr_offset = mul_power2<16 / sizeof(OffsetT)>(lane_idx);

      // Column-wise quantization, every column has its own scale/offset
      const uint8_t* offsets_ptr = offsets_byte_p + (div_power2<QuantBlocking::kRow>(start_k)) * offsets_byte_stride;
      const int k_loads = div_up(k_cnt, QuantBlocking::kRow);

      const int k_idx = div_power2<WarpShape::kN>(lane_ptr_offset);
      const int n_idx = mod_power2<WarpShape::kN>(lane_ptr_offset);

      unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(&offset_smem[lane_ptr_offset]);
      int src_in_bytes = ((k_idx < k_loads && n_idx < n_cnt) ? 16 : 0);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
          "}\n"
          ::"r"((int)(kOffsetLoadThreads > lane_idx)),
            "r"(smem_int_ptr),
            "l"(&offsets_ptr[k_idx * offsets_byte_stride + n_idx * sizeof(OffsetT)]),
            "n"(16), "r"(src_in_bytes));
    }
#else
      assert(false);
      (void)(lane_idx);
      (void)(start_k);
      (void)(k_cnt);
      (void)(smem);
      (void)(offset_smem);
#endif
  }

  CUTLASS_DEVICE
  static void load_fragment(const int lane_idx,
      FragmentScales &frag_scales, const ElementT* smem,
      FragmentOffsets &frag_offsets, const OffsetT* offset_smem) {
    const int n_idx = div_power2<4>(lane_idx);
    ElementT const* scales_ptr = smem + n_idx;
    [[maybe_unused]] OffsetT const* offset_ptr = offset_smem + n_idx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kSmemSize / 8; ++i) {
      frag_scales[i] = scales_ptr[i << 3];
      if constexpr(has_offsets) {
        uint16_t v = offset_ptr[i << 3];
        frag_offsets[i] = cutlass::half_t(__ushort2half_rn(v));
      }
    }
  }

  using FragmentB = cutlass::Array<ElementT, 2 * (WarpShape::kN / 8) * 2>;

  /// Dequantize a block of (16, WarpShape::kN) packed int4 weights to 16b float.
  /// This block has (WarpShape::kN / 8) * 2 tiles, each tile has 2 elements per thread,
  /// thus the FragmentB has (WarpShape::kN / 8) * 2 * 2 elements.

  template <int PackedBSize>
  CUTLASS_DEVICE
  static void dequant_k16(
      const int k_iter,
      cutlass::Array<unsigned, PackedBSize> const &frag_pack_b,
      FragmentScales const &frag_scales,
      FragmentOffsets const &frag_offsets,
      FragmentB &frag_b) {
    // Each 32b number in packed B represent a 16x16 tile
    constexpr int kPackedBNTiles = WarpShape::kN / 16;
    constexpr int kPackedBKStride = PackedBSize / kPackedBNTiles;
    static_assert(kPackedBKStride * kPackedBNTiles == PackedBSize);

    // We are processing 16xWarpShape::kN weights at a time, assuming each column has
    // only one scale/offset, so the block size cannot be smaller than 16.
    static_assert(QuantBlocking::kRow % 16 == 0);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const int meta_k = k_iter / (QuantBlocking::kRow / 16);
    half const* scales = reinterpret_cast<half const*>(frag_scales.data() + meta_k * kMetaFragSize);
    [[maybe_unused]] half const* offsets = nullptr;
    if constexpr(has_offsets) {
      offsets = reinterpret_cast<half const*>(frag_offsets.data() + meta_k * kMetaFragSize);
    }

    // Column-wise quantization, every column has its own scale/offset
    CUTLASS_PRAGMA_UNROLL
    for (int nn = 0; nn < (WarpShape::kN / 16); ++nn) {
      const int b_idx = (k_iter % kPackedBKStride) * kPackedBNTiles + nn;
      half2* fb_pair = reinterpret_cast<half2*>(frag_b.data() + nn * 8);

      half2 scale_pair = __half2half2(scales[nn * 2]);
      half2 scale_pair1 = __half2half2(scales[nn * 2 + 1]);

      cutlass::Array<ElementT, 8> ws;
      half2* weight_pair = reinterpret_cast<half2*>(ws.data());
      if constexpr(has_offsets) {
        detail::weights2Half(frag_pack_b[b_idx], ws);
        half2 offset_pair = __half2half2(offsets[nn * 2]);
        half2 offset_pair1 = __half2half2(offsets[nn * 2 + 1]);
        weight_pair[0] = __hsub2(weight_pair[0], offset_pair);
        weight_pair[1] = __hsub2(weight_pair[1], offset_pair);
        weight_pair[2] = __hsub2(weight_pair[2], offset_pair1);
        weight_pair[3] = __hsub2(weight_pair[3], offset_pair1);
      } else {
        detail::weightsMinuEight2Half(frag_pack_b[b_idx], ws);
      }

      fb_pair[0] = __hmul2(scale_pair, weight_pair[0]);
      fb_pair[1] = __hmul2(scale_pair, weight_pair[1]);
      fb_pair[2] = __hmul2(scale_pair1, weight_pair[2]);
      fb_pair[3] = __hmul2(scale_pair1, weight_pair[3]);

      if constexpr (DebugPrint) {
        const int lane_id = threadIdx.x % 32;
        const char* const format = ((lane_id % 4) == 3) ? "%f=%fx%f, %f=%fx%f\n" : "%f=%fx%f, %f=%fx%f, ";
        printf(format, float(fb_pair[0].x), float(weight_pair[0].x), float(scale_pair.x),
               float(fb_pair[0].y), float(weight_pair[0].y), float(scale_pair.y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[1].x), float(weight_pair[1].x), float(scale_pair.x),
               float(fb_pair[1].y), float(weight_pair[1].y), float(scale_pair.y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[2].x), float(weight_pair[2].x), float(scale_pair1.x),
               float(fb_pair[2].y), float(weight_pair[2].y), float(scale_pair1.y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[3].x), float(weight_pair[3].x), float(scale_pair1.x),
               float(fb_pair[3].y), float(weight_pair[3].y), float(scale_pair1.y));
        if (lane_id == 31) {
          printf("\n");
        }
      }
    }
#else
    assert(false);
    (void)(k_iter);
    (void)(frag_pack_b);
    (void)(frag_scales);
    (void)(frag_offsets);
    (void)(frag_b);
#endif  // __CUDA_ARCH__
  }

};


/// Specialization for row-wise quantization, i.e. QuantBlocking::kRow == 1
template <
    int block_size_,
    typename WarpShape_,
    typename ElementT_,
    bool has_offsets,
    bool DebugPrint>
struct QuantBScaleLoader<cutlass::MatrixShape<1, block_size_>, WarpShape_, ElementT_, has_offsets, DebugPrint> {
  //
  // Type definitions
  //
  using QuantBlocking = cutlass::MatrixShape<1, block_size_>;
  using WarpShape = WarpShape_;
  using ElementT = ElementT_;
  using OffsetT = uint8_t;

  static_assert((WarpShape::kN % 16) == 0 && (WarpShape::kK % 16) == 0,
                "Warp tile size must be multiple of 16x16, the unit of packed weights.");
  static_assert(sizeof(ElementT) == 2, "Quantization only supports 16-bit float types");

  //
  // Row-wise blocking --> kRow == 1, every row has its own
  // scale/offset, there are far less columns than rows in a warp tile.
  // So we use column-major layout to maximize continuous memory access in
  // a warp.
  //

  // Number of continuous elements of scale/offset that a warp need to load
  static constexpr int kMetaFragSize = (WarpShape::kK / 8) * 2;  // row wise quant, every row has its own scale/offset
  static constexpr int kMetaChunkCount = div_up(WarpShape::kN, QuantBlocking::kColumn);

  // HBM -> SMEM, 16 bytes per load, no leftover since WarpShape::kN is multiple of 16
  static constexpr int kSmemSize = WarpShape::kK * kMetaChunkCount;
  static constexpr int kScaleLoadThreads = (kSmemSize * sizeof(ElementT)) / 16;
  static_assert(kScaleLoadThreads <= 16); // shape up to 64x64, 16 threads can load all scales

  using FragmentScales = cutlass::Array<ElementT, kMetaFragSize * kMetaChunkCount>;

  static constexpr int kOffsetLoadThreads = (kSmemSize * sizeof(OffsetT)) / 16;
  static_assert(kOffsetLoadThreads <= 16); // shape up to 64x64, 16 threads can load all offsets

  using FragmentOffsets = typename std::conditional<has_offsets,
                                           FragmentScales,
                                           std::monostate>::type;

  //
  // Data members
  //
  const int n_cnt;

  const uint8_t * const scales_byte_p;
  const int scales_byte_stride;
  const uint8_t * const offsets_byte_p;
  const int offsets_byte_stride;

  //
  // Methods
  //
  template <typename T>
  CUTLASS_DEVICE
  static const uint8_t* get_scales_p(const void* ptr_scales, int scales_byte_stride, int k, int n) {
    return (ptr_scales == nullptr) ? nullptr :
           reinterpret_cast<uint8_t const*>(ptr_scales) + n * scales_byte_stride + k * sizeof(T);
  }

  /// Initializes the scale loader, pointing to the start of the scales tensor
  CUTLASS_DEVICE
  QuantBScaleLoader(
      int lane_idx,
      void const *ptr_scales,
      int scales_byte_stride,
      void const *ptr_offsets,  // dummy to make the interface consistent with QuantBScaleOffsetLoader
      int offsets_byte_stride,
      int start_n,
      int end_n)
      : n_cnt(div_up(end_n - start_n, QuantBlocking::kColumn)),
        scales_byte_p(get_scales_p<ElementT>(ptr_scales, scales_byte_stride, 0, start_n / QuantBlocking::kColumn)),
        scales_byte_stride(scales_byte_stride),
        offsets_byte_p(get_scales_p<OffsetT>(ptr_offsets, offsets_byte_stride, 0, start_n / QuantBlocking::kColumn)),
        offsets_byte_stride(offsets_byte_stride)
  {
    assert(ptr_scales != nullptr);
    assert(scales_byte_stride > 0 && mod_power2<16>(scales_byte_stride) == 0);
    if constexpr(has_offsets) {
      assert(ptr_offsets != nullptr);
      assert(offsets_byte_stride > 0 && mod_power2<16>(offsets_byte_stride) == 0);
    } else {
      assert(ptr_offsets == nullptr);
      assert(offsets_byte_stride == 0);
    }
  }

  /// Loads [start_k, end_k) x [start_n, end_n) scales from global memory to fragment
  /// [start_n, end_n) was specified in the constructor
  CUTLASS_DEVICE
  void load_to_smem(const int lane_idx, const int start_k, const int k_cnt, ElementT* smem, OffsetT* offset_smem) const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    {
      // Load scales to smem
      int lane_ptr_offset = mul_power2<16 / sizeof(ElementT)>(lane_idx);
      const uint8_t* scales_ptr = scales_byte_p + start_k * sizeof(ElementT);
      const int k_idx = lane_ptr_offset % WarpShape::kK;
      const int n_idx = lane_ptr_offset / WarpShape::kK;

      unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(&smem[lane_ptr_offset]);
      int src_in_bytes = ((k_idx < k_cnt && n_idx < n_cnt) ? 16 : 0);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
          "}\n"
          ::"r"((int)(kScaleLoadThreads > lane_idx)),
            "r"(smem_int_ptr),
            "l"(&scales_ptr[n_idx * scales_byte_stride + k_idx * sizeof(ElementT)]),
            "n"(16), "r"(src_in_bytes));
    }
    if constexpr(has_offsets) {
      // Load offsets to smem
      int lane_ptr_offset = mul_power2<16 / sizeof(OffsetT)>(lane_idx);
      const uint8_t* offsets_ptr = offsets_byte_p + start_k * sizeof(OffsetT);
      const int k_idx = lane_ptr_offset % WarpShape::kK;
      const int n_idx = lane_ptr_offset / WarpShape::kK;

      unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(&offset_smem[lane_ptr_offset]);
      int src_in_bytes = ((k_idx < k_cnt && n_idx < n_cnt) ? 16 : 0);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
          "}\n"
          ::"r"((int)(kOffsetLoadThreads > lane_idx)),
            "r"(smem_int_ptr),
            "l"(&offsets_ptr[n_idx * offsets_byte_stride + k_idx * sizeof(OffsetT)]),
            "n"(16), "r"(src_in_bytes));
    }
#else
      assert(false);
      (void)(lane_idx);
      (void)(start_k);
      (void)(k_cnt);
      (void)(smem);
      (void)(offset_smem);
#endif
  }

  CUTLASS_DEVICE
  static void load_fragment(const int lane_idx,
      FragmentScales &frag_scales, const ElementT* smem,
      [[maybe_unused]] FragmentOffsets &frag_offsets,
      [[maybe_unused]] const OffsetT* offset_smem) {
    // Row-wise quantization, every row has its own scale/offset, elements have been rearraged
    // such that we can load two tile at a time.
    // T0        T0
    // T1        T0
    // T2        T1
    // T3   =>   T1
    // T0        T2
    // T1        T2
    // T2        T3
    // T3        T3
    const int lane_offset = mod_power2<4>(lane_idx) << 2;
    const uint32_t* scales_ptr = reinterpret_cast<const uint32_t*>(smem + lane_offset);
    [[maybe_unused]] const uint32_t* offsets_ptr = nullptr;
    if constexpr(has_offsets) {
      offsets_ptr = reinterpret_cast<const uint32_t*>(offset_smem + lane_offset);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentScales::kElements; i += 4) {
      uint32_t* frag_ptr = reinterpret_cast<uint32_t *>(frag_scales.data() + i);
      frag_ptr[0] = scales_ptr[0];
      frag_ptr[1] = scales_ptr[1];
      scales_ptr += 8;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      if constexpr(has_offsets) {
        // offsets are always 4 a group, this give us an opportunity to use
        // a little trick to reduce the number of instructions.
        // So here 4 offset a, b, c, d, we convert them to fp16, but not quite,
        // the converted value is a + 1024.0, b + 1024.0, c + 64.0, d + 64.0.
        // This dovetail with the dequantization method below, where we have
        // a similar conversion of the weights.
        uint32_t* offset_pair = reinterpret_cast<uint32_t*>(frag_offsets.data() + i);
        {
          const uint32_t ab = offsets_ptr[0];
          const uint32_t cd = ab >> 4;
          asm volatile(
            "  lop3.b32      %0, %2, 0x000f000f, %4, 0xea;\n"
            "  lop3.b32      %1, %3, 0x00f000f0, %5, 0xea;\n"
            : "=r"(offset_pair[0]), "=r"(offset_pair[1])
            : "r"(ab), "r"(cd),
              "r"(0x64006400), "r"(0x54005400));
        }
        offsets_ptr += 4;
      }
#endif
    }
  }

  using FragmentB = cutlass::Array<ElementT, 2 * (WarpShape::kN / 8) * 2>;

  /// Dequantize a block of (16, WarpShape::kN) packed int4 weights to 16b float.
  /// This block has (WarpShape::kN / 8) * 2 tiles, each tile has 2 elements per thread,
  /// thus the FragmentB has (WarpShape::kN / 8) * 2 * 2 elements.
  template<int PackedBSize>
  CUTLASS_DEVICE
  static void dequant_k16(
      const int k_iter,
      cutlass::Array<unsigned, PackedBSize> const &frag_pack_b,
      FragmentScales const &frag_scales,
      FragmentOffsets const &frag_offsets,
      FragmentB &frag_b) {
    // Each 32b number in packed B represent a 16x16 tile
    constexpr int kPackedBNTiles = WarpShape::kN / 16;
    constexpr int kPackedBKStride = PackedBSize / kPackedBNTiles;
    static_assert(kPackedBKStride * kPackedBNTiles == PackedBSize);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // Row-wise quantization, every row has its own scale/offset
    CUTLASS_PRAGMA_UNROLL
    for (int nn = 0; nn < (WarpShape::kN / 16); ++nn) {
      const int b_idx = (k_iter % kPackedBKStride) * kPackedBNTiles + nn;
      half2* const fb_pair = reinterpret_cast<half2*>(frag_b.data() + nn * 8);
      const int meta_n = (nn * 16) / QuantBlocking::kColumn;
      const int idx = meta_n * kMetaFragSize + (k_iter * 4);
      half2 const* const scale_pair = reinterpret_cast<half2 const*>(frag_scales.data() + idx); // k_offset / 16 * 4
      [[maybe_unused]] half2 const* offsets = nullptr;
      if constexpr(has_offsets) {
        offsets = reinterpret_cast<half2 const*>(frag_offsets.data() + idx);
      }
      cutlass::Array<ElementT, 8> ws;
      half2* weight_pair = reinterpret_cast<half2*>(ws.data());
      if constexpr (has_offsets) {
        // a group of 4 offsets was converted to a + 1024.0, b + 1024.0, c + 64.0, d + 64.0
        // when loaded from shared memory.
        {
          uint32_t*      b32s   = reinterpret_cast<uint32_t*>(ws.data());
          const uint32_t low_8s = frag_pack_b[b_idx];
          const uint32_t high_8s = low_8s >> 8;

          asm volatile(
            "  lop3.b32      %0, %4, 0x000f000f, 0x64006400, 0xea;\n"
            "  lop3.b32      %1, %4, 0x00f000f0, 0x54005400, 0xea;\n"
            "  lop3.b32      %2, %5, 0x000f000f, 0x64006400, 0xea;\n"
            "  lop3.b32      %3, %5, 0x00f000f0, 0x54005400, 0xea;\n"
            : "=r"(b32s[0]), "=r"(b32s[1]), "=r"(b32s[2]), "=r"(b32s[3])
            : "r"(low_8s), "r"(high_8s));
        }

        weight_pair[0] = __hsub2(weight_pair[0], offsets[0]);
        weight_pair[1] = __hsub2(weight_pair[1], offsets[1]);
        weight_pair[2] = __hsub2(weight_pair[2], offsets[0]);
        weight_pair[3] = __hsub2(weight_pair[3], offsets[1]);
      } else {
        detail::weightsMinuEight2Half(frag_pack_b[b_idx], ws);
      }

      fb_pair[0] = __hmul2(scale_pair[0], weight_pair[0]);
      fb_pair[1] = __hmul2(scale_pair[1], weight_pair[1]);
      fb_pair[2] = __hmul2(scale_pair[0], weight_pair[2]);
      fb_pair[3] = __hmul2(scale_pair[1], weight_pair[3]);

      if constexpr (DebugPrint) {
        const int lane_id = threadIdx.x % 32;
        const char* const format = ((lane_id % 4) == 3) ? "%f=%fx%f, %f=%fx%f\n" : "%f=%fx%f, %f=%fx%f, ";
        printf(format, float(fb_pair[0].x), float(weight_pair[0].x), float(scale_pair[0].x),
               float(fb_pair[0].y), float(weight_pair[0].y), float(scale_pair[0].y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[1].x), float(weight_pair[1].x), float(scale_pair[1].x),
               float(fb_pair[1].y), float(weight_pair[1].y), float(scale_pair[1].y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[2].x), float(weight_pair[2].x), float(scale_pair[0].x),
               float(fb_pair[2].y), float(weight_pair[2].y), float(scale_pair[0].y));
        if (lane_id == 31) {
          printf("\n");
        }
        printf(format, float(fb_pair[3].x), float(weight_pair[3].x), float(scale_pair[1].x),
               float(fb_pair[3].y), float(weight_pair[3].y), float(scale_pair[1].y));
        if (lane_id == 31) {
          printf("\n");
        }
      }
    }
#else
    assert(false);
    (void)(k_iter);
    (void)(frag_pack_b);
    (void)(frag_scales);
    (void)(frag_offsets);
    (void)(frag_b);
#endif  // __CUDA_ARCH__
  }

};

}  // namespace warp
}  // namespace gemm
}  // namespace mickey
