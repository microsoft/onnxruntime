/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file warp/swizzle_tile_loader.h
 * @brief Load matrix tiles from global memory to shared memory, in a way the shared memory
 * tiles can be readily loaded using ldmatrix instruction while avoiding bank conflicts.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cute/layout.hpp"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

#include "int_util.h"

namespace mickey {
namespace gemm {
namespace warp {

/**
 * @brief Load a row major tile (SmemDimM, SmemDimK) from global memory to shared
 *        memory and then to fragment with ldmatrix instruction, with swizzling
 *        to avoid bank conflicts.
 */
template <int SmemDimM, int SmemDimK>
class SwizzleTileLoader;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Swizzle pattern really depend on the stride of K, with different pattern size too.
// Now we specialize for each case. Need to find a way to unify them.

template <int SmemDimM_>
class SwizzleTileLoader<SmemDimM_, 64> {
 public:
  static constexpr int SmemDimM = SmemDimM_;
  static constexpr int SmemDimK = 64;
  static constexpr int kLoadVectorSize = 16;  // one cp.async loads 16 bytes
  static constexpr int kBlockSize = SmemDimM * SmemDimK;
  static constexpr int kTiles = (SmemDimM / 8) * (SmemDimK / 16);

  // Swizzle pattern is 4x8
  static constexpr int kSwizzleK = SmemDimK / kLoadVectorSize;
  static_assert(kSwizzleK == cute::_4::value);
  static constexpr int kSwizzleM = cute::_8::value;
  static constexpr int kSwizzleTileSize = kSwizzleK * kSwizzleM;
  using Swizzled64 = decltype(cute::composition(cute::Swizzle<2, 0, 3>{},
                                                cute::Layout<cute::Shape<cute::_4, cute::_8>,
                                                             cute::Stride<cute::_1, cute::_4>>{}));

  static constexpr int kThreads = 32;
  static constexpr int kGmemLoadStrideM = kThreads / kSwizzleK;
  static_assert(kGmemLoadStrideM * kSwizzleK == kThreads);
  static_assert(SmemDimM % kGmemLoadStrideM == 0);

  // During pipelined MMA, each stage (processing a tile) is split
  // into multiple mma iterations. We need to somehow split the
  // the loading of global memory tile into multiple calls, doing
  // our best to help spread these actions across different iterations
  // in a stage.
  static constexpr int kGloadSplit = SmemDimM / kGmemLoadStrideM;

 private:
  /// Pointer to global memory to load data from
  uint8_t const* g_ptr_{nullptr};
  /// Iteration boundaries in the M or N dimension
  int mn_cnt_{0};
  /// Iteration boundaries in the K dimension, in strides of 16
  int k_cnt_{0};
  /// Stride in bytes to advance to next row in m or n dimension
  const int stride_;

 public:
  CUTLASS_DEVICE
  SwizzleTileLoader(
      void const* data_ptr,  ///< Pointer to the global memory tiles
      int byte_stride,       ///< Stride in bytes to advance to next row
      int mn_start,          ///< Starting position in the M or N dimension
      int mn_end,            ///< End position in the M or N dimension
      int k_start,           ///< Starting position in the K dimension
      int k_end,             ///< End position in the K dimension
      int lane_id)           ///< ID of each participating thread
      : stride_(byte_stride) {
#ifndef NDEBUG
    bool assertion_pass = true;
    if (reinterpret_cast<uintptr_t>(data_ptr) % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("data_ptr: %p is not aligned to 16B boundary!\n", data_ptr);
      }
    }
    if (byte_stride % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("byte_stride: %d is not aligned to 16B boundary!\n", byte_stride);
      }
    }
    if (k_start % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_start: %d is not aligned to 16B boundary!\n", k_start);
      }
    }
    if (k_end % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_end: %d is not aligned to 16B boundary!\n", k_end);
      }
    }
    if (mn_end <= mn_start) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("mn_end: %d is less than or equal to mn_start: %d!\n", mn_end, mn_start);
      }
    }
    if (k_end <= k_start) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_end: %d is less than or equal to k_start: %d!\n", k_end, k_start);
      }
    }
    if (lane_id < 0 || lane_id >= kThreads) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("Warp based loader, lane_id should be [0-32) but it is: %d!\n", lane_id);
      }
    }
    assert(assertion_pass);
#endif

    int lane_m = div_power2<kSwizzleK>(lane_id);
    int lane_k = mod_power2<kSwizzleK>(lane_id);
    mn_start += lane_m;
    k_start += mul_power2<kLoadVectorSize>(lane_k);

    mn_cnt_ = div_up(mn_end - mn_start, kGmemLoadStrideM);
    k_cnt_ = div_up(k_end - k_start, kSwizzleK * kLoadVectorSize);
    if (mn_cnt_ <= 0 || k_cnt_ <= 0) {
      mn_cnt_ = 0;
      k_cnt_ = 0;
      g_ptr_ = nullptr;
      return;
    }
    g_ptr_ = reinterpret_cast<uint8_t const*>(data_ptr) + mn_start * byte_stride + k_start;
    // if (lane_id == 0)
    //   printf("lane_id: %d, mn_start: %d, mn_end: %d, k_start: %d, k_end: %d, g_ptr: %p\n", lane_id, mn_start, mn_end, k_start, k_end, g_ptr_);
  }

  /**
   * @brief Load a row major tile (SmemDimM, 64) from global memory to shared memory
   */
  CUTLASS_DEVICE
  void load_to_smem(const int lane_id, void* smem) {
    // Here we rely on the fact that kThreads is 32, same as the swizzle pattern size
    static_assert(kGmemLoadStrideM == kSwizzleM);
    const uint8_t* data_ptr = g_ptr_;
    uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem) + mul_power2<kLoadVectorSize>(Swizzled64{}(lane_id));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < SmemDimM / kSwizzleM; ++i) {
      cutlass::arch::cp_async<kLoadVectorSize, cutlass::arch::CacheOperation::Global>(
          smem_ptr, data_ptr, g_ptr_ != nullptr && i < mn_cnt_);
      data_ptr += mul_power2<kGmemLoadStrideM>(stride_);
      smem_ptr += kSwizzleTileSize * kLoadVectorSize;
    }
  }

  CUTLASS_DEVICE
  void load_to_smem_split(const int lane_id, void* smem, const int split_idx) {
    // Here we rely on the fact that kThreads is 32, same as the swizzle pattern size
    static_assert(kGmemLoadStrideM == kSwizzleM);

    const uint8_t* split_ptr = g_ptr_ + mul_power2<kGmemLoadStrideM>(split_idx * stride_);
    uint8_t* split_smem_ptr = reinterpret_cast<uint8_t*>(smem) + mul_power2<kLoadVectorSize>(Swizzled64{}(lane_id)) + split_idx * kSwizzleTileSize * kLoadVectorSize;

    cutlass::arch::cp_async<kLoadVectorSize, cutlass::arch::CacheOperation::Global>(
        split_smem_ptr, split_ptr, g_ptr_ != nullptr && split_idx < mn_cnt_);
  }

  /**
   * @brief Advance global memory pointer to the next tile in the K dimension
   */
  CUTLASS_DEVICE
  SwizzleTileLoader& operator++() {
    --k_cnt_;
    if (k_cnt_ > 0) {
      g_ptr_ += kLoadVectorSize * kSwizzleK;
    } else {
      g_ptr_ = nullptr;
    }
    return *this;
  }

  /**
   * @brief Load a ribbin of (SmemDimM, 32) from shared memory to fragment,
   * fitting fp16 gemm sm80 tensor core shape, where k = 16 x sizeof(fp16)
   */
  CUTLASS_DEVICE
  void load_fragment_k32(const int lane_id, void const* smem, int offset_k, void* frag) {
#ifndef NDEBUG
    bool assert_fail = false;
    if (offset_k != 0 && offset_k != 32) {
      assert_fail = true;
      if (lane_id == 0) {
        printf("Invalid offset_k: %d!\n", offset_k);
      }
    }
    if (SmemDimM % 16 != 0) {
      // 2x2 tiles per load: 16 threads on the M dim and 2 on the K dim
      // and don't want to deal with left over M
      assert_fail = true;
      if (lane_id == 0) {
        printf("SmemDimM: %d two small, cannot use ldmatrix fully!\n", SmemDimM);
      }
    }
    assert(assert_fail == false);
#endif

    constexpr int kStrideM = 16 / kSwizzleM;  // Span 2 swizzle patterns on M dim
    int m_lane_id = mod_power2<16>(lane_id);
    int k_lane_id = (lane_id >> 4) + (offset_k >> 4);

    int m_tile_id = div_power2<kSwizzleM>(m_lane_id);
    int m_tile_offset = mod_power2<kSwizzleM>(m_lane_id);
    int swizzled_id = Swizzled64{}(k_lane_id, m_tile_offset) + mul_power2<kSwizzleTileSize>(m_tile_id);
    // printf("lane_id: %d, m_lane_id: %d, k_lane_id: %d, swizzled_id: %d\n", lane_id, m_lane_id, k_lane_id, swizzled_id);
    const uint8_t* smem_ptr = reinterpret_cast<const uint8_t*>(smem) + mul_power2<kLoadVectorSize>(swizzled_id);

    using FragType = cutlass::Array<unsigned, 4>;
    FragType* frag_ptr = reinterpret_cast<FragType*>(frag);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < (SmemDimM / 16); ++i) {
      // printf("lane_id: %d, load %d, val: %d, smem_ptr: %p\n", lane_id, i, smem_ptr[0], smem_ptr);
      cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(frag_ptr[i], smem_ptr);
      smem_ptr += kSwizzleTileSize * kStrideM * kLoadVectorSize;
    }
  }

  CUTLASS_DEVICE
  void load_fragment_k64(const int lane_id, void const* smem, int offset_k, void* frag) {
#ifndef NDEBUG
    // Here we use a single warp to load 4 tiles on the k dimension.
    // This is only useful in loading packed B tensor where a 2x2 int4
    // tile structure is disguised as a single fp16 tile. So 4 such
    // tiles, when dequantized, become 4 set of 2x2 fp16 tiles. Each
    // of the 2x2 fp16 tiles can participate in two 16x8x16 tensor core
    // operations.
    bool assert_fail = false;
    if (SmemDimM != 8) {
      assert_fail = true;
      if (lane_id == 0) {
        printf("Special case for SmemDimM = 8 but found %d!\n", SmemDimM);
      }
    }
    if (offset_k != 0) {
      assert_fail = true;
      if (lane_id == 0) {
        printf("Special case for offset_k = 0 but found %d!\n", offset_k);
      }
    }
    assert(assert_fail == false);
#endif

    // 1x4 tiles per load: 8 threads on the M dim and 4 on the K dim
    int m_lane_id = mod_power2<8>(lane_id);
    int k_lane_id = div_power2<8>(lane_id);

    int swizzled_id = Swizzled64{}(k_lane_id, m_lane_id);
    // printf("lane_id: %d, m_lane_id: %d, k_lane_id: %d, swizzled_id: %d\n", lane_id, m_lane_id, k_lane_id, swizzled_id);
    const uint8_t* smem_ptr = reinterpret_cast<const uint8_t*>(smem) + mul_power2<kLoadVectorSize>(swizzled_id);

    using FragType = cutlass::Array<unsigned, 4>;
    FragType* frag_ptr = reinterpret_cast<FragType*>(frag);
    cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(frag_ptr[0], smem_ptr);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int SmemDimM_>
class SwizzleTileLoader<SmemDimM_, 128> {
 public:
  static constexpr int SmemDimM = SmemDimM_;
  static constexpr int SmemDimK = 128;
  static constexpr int kLoadVectorSize = 16;  // one cp.async loads 16 bytes
  static constexpr int kBlockSize = SmemDimM * SmemDimK;
  static constexpr int kTiles = (SmemDimM / 8) * (SmemDimK / 16);

  // Swizzle pattern is 8x8
  static constexpr int kSwizzleK = SmemDimK / kLoadVectorSize;
  static_assert(kSwizzleK == cute::_8::value);
  static constexpr int kSwizzleM = cute::_8::value;
  static constexpr int kSwizzleTileSize = kSwizzleK * kSwizzleM;
  using Swizzled128 = decltype(cute::composition(cute::Swizzle<3, 0, 3>{},
                                                 cute::Layout<cute::Shape<cute::_8, cute::_8>,
                                                              cute::Stride<cute::_1, cute::_8>>{}));

  static constexpr int kThreads = 32;
  static constexpr int kGmemLoadStrideM = kThreads / kSwizzleK;
  static_assert(kGmemLoadStrideM * kSwizzleK == kThreads);

  // During pipelined MMA, each stage (processing a tile) is split
  // into multiple mma iterations. We need to somehow split the
  // the loading of global memory tile into multiple calls, doing
  // our best to help spread these actions across different iterations
  // in a stage.
  static constexpr int kGloadSplit = SmemDimM / kGmemLoadStrideM;

 private:
  /// Pointer to global memory to load data from
  uint8_t const* g_ptr_{nullptr};
  /// Iteration boundaries in the M or N dimension
  int mn_cnt_{0};
  /// Iteration boundaries in the K dimension, in strides of 16
  int k_cnt_{0};
  /// Stride in bytes to advance to next row in m or n dimension
  const int stride_;

 public:
  CUTLASS_DEVICE
  SwizzleTileLoader(
      void const* data_ptr,  ///< Pointer to the global memory tiles
      int byte_stride,       ///< Stride in bytes to advance to next row
      int mn_start,          ///< Starting position in the M or N dimension
      int mn_end,            ///< End position in the M or N dimension
      int k_start,           ///< Starting position in the K dimension
      int k_end,             ///< End position in the K dimension
      int lane_id)           ///< ID of each participating thread
      : stride_(byte_stride) {
#ifndef NDEBUG
    bool assertion_pass = true;
    if (reinterpret_cast<uintptr_t>(data_ptr) % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("data_ptr: %p is not aligned to 16B boundary!\n", data_ptr);
      }
    }
    if (byte_stride % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("byte_stride: %d is not aligned to 16B boundary!\n", byte_stride);
      }
    }
    if (k_start % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_start: %d is not aligned to 16B boundary!\n", k_start);
      }
    }
    if (k_end % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_end: %d is not aligned to 16B boundary!\n", k_end);
      }
    }
    if (mn_end <= mn_start) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("mn_end: %d is less than or equal to mn_start: %d!\n", mn_end, mn_start);
      }
    }
    if (k_end <= k_start) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_end: %d is less than or equal to k_start: %d!\n", k_end, k_start);
      }
    }
    if (lane_id < 0 || lane_id >= kThreads) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("Warp based loader, lane_id should be [0-32) but it is: %d!\n", lane_id);
      }
    }
    assert(assertion_pass);
#endif

    int lane_m = lane_id / kSwizzleK;
    int lane_k = lane_id % kSwizzleK;
    mn_start += lane_m;
    k_start += lane_k * kLoadVectorSize;

    mn_cnt_ = div_up(mn_end - mn_start, kGmemLoadStrideM);
    k_cnt_ = div_up(k_end - k_start, kSwizzleK * kLoadVectorSize);
    if (mn_cnt_ <= 0 || k_cnt_ <= 0) {
      mn_cnt_ = 0;
      k_cnt_ = 0;
      g_ptr_ = nullptr;
      return;
    }
    g_ptr_ = reinterpret_cast<uint8_t const*>(data_ptr) + mn_start * byte_stride + k_start;
    // if (lane_id == 0)
    //   printf("lane_id: %d, mn_start: %d, mn_end: %d, k_start: %d, k_end: %d, g_ptr: %p\n", lane_id, mn_start, mn_end, k_start, k_end, g_ptr_);
  }

  /**
   * @brief Load a row major tile (SmemDimM, 128) from global memory to shared memory
   */
  CUTLASS_DEVICE
  void load_to_smem(const int lane_id, void* smem) {
    const uint8_t* data_ptr = g_ptr_;

    // The swizzle pattern is 8x8, but we only have 32 threads,
    // covering half of the swizzle pattern
    static_assert(kGmemLoadStrideM * 2 == kSwizzleM);
    uint8_t* smem_ptr0 = reinterpret_cast<uint8_t*>(smem) + Swizzled128{}(lane_id)*kLoadVectorSize;
    uint8_t* smem_ptr1 = reinterpret_cast<uint8_t*>(smem) + Swizzled128{}(lane_id + kThreads) * kLoadVectorSize;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < SmemDimM / kGmemLoadStrideM;) {
      cutlass::arch::cp_async<kLoadVectorSize, cutlass::arch::CacheOperation::Global>(
          smem_ptr0, data_ptr, g_ptr_ != nullptr && i < mn_cnt_);
      data_ptr += stride_ * kGmemLoadStrideM;
      smem_ptr0 += kSwizzleTileSize * kLoadVectorSize;
      ++i;

      cutlass::arch::cp_async<kLoadVectorSize, cutlass::arch::CacheOperation::Global>(
          smem_ptr1, data_ptr, g_ptr_ != nullptr && i < mn_cnt_);
      data_ptr += stride_ * kGmemLoadStrideM;
      smem_ptr1 += kSwizzleTileSize * kLoadVectorSize;
      ++i;
    }
  }

  CUTLASS_DEVICE
  void load_to_smem_split(const int lane_id, void* smem, const int split_idx) {
    const uint8_t* split_ptr = g_ptr_ + split_idx * stride_ * kGmemLoadStrideM;
    const int offset = (split_idx >> 1) * kSwizzleTileSize * kLoadVectorSize;
    const int swizzled = Swizzled128{}(lane_id + (split_idx & 1) * kThreads) * kLoadVectorSize;
    uint8_t* split_smem_ptr = reinterpret_cast<uint8_t*>(smem) + swizzled + offset;

    cutlass::arch::cp_async<kLoadVectorSize, cutlass::arch::CacheOperation::Global>(
        split_smem_ptr, split_ptr, g_ptr_ != nullptr && split_idx < mn_cnt_);
  }

  /**
   * @brief Advance global memory pointer to the next tile in the K dimension
   */
  CUTLASS_DEVICE
  SwizzleTileLoader& operator++() {
    --k_cnt_;
    if (k_cnt_ > 0) {
      g_ptr_ += kLoadVectorSize * kSwizzleK;
    } else {
      g_ptr_ = nullptr;
    }
    return *this;
  }

  /**
   * @brief Load a ribbin of (SmemDimM, 32) from shared memory to fragment,
   * fitting fp16 gemm sm80 tensor core shape, where k = 16 x sizeof(fp16)
   */
  CUTLASS_DEVICE
  void load_fragment_k32(const int lane_id, void const* smem, int offset_k, void* frag) {
#ifndef NDEBUG
    bool assert_fail = false;
    if ((offset_k % 32) != 0) {
      assert_fail = true;
      if (lane_id == 0) {
        printf("Invalid offset_k: %d!\n", offset_k);
      }
    }
    if ((SmemDimM % 16) != 0) {
      // 2x2 tiles per load: 16 threads on the M dim and 2 on the K dim
      // and don't want to deal with left over M
      assert_fail = true;
      if (lane_id == 0) {
        printf("SmemDimM: %d two small, cannot use ldmatrix fully!\n", SmemDimM);
      }
    }
    assert(assert_fail == false);
#endif

    constexpr int kStrideM = 16 / kSwizzleM;  // Span 2 swizzle patterns on M dim
    int m_lane_id = lane_id % 16;
    int k_lane_id = lane_id / 16 + offset_k / 16;

    int m_tile_id = m_lane_id / kSwizzleM;
    int m_tile_offset = m_lane_id % kSwizzleM;
    int swizzled_id = Swizzled128{}(k_lane_id, m_tile_offset) + m_tile_id * kSwizzleTileSize;
    // printf("lane_id: %d, m_lane_id: %d, k_lane_id: %d, swizzled_id: %d\n", lane_id, m_lane_id, k_lane_id, swizzled_id);
    const uint8_t* smem_ptr = reinterpret_cast<const uint8_t*>(smem) + swizzled_id * kLoadVectorSize;

    using FragType = cutlass::Array<unsigned, 4>;
    FragType* frag_ptr = reinterpret_cast<FragType*>(frag);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < SmemDimM / 16; ++i) {
      // printf("lane_id: %d, load %d, val: %d, smem_ptr: %p\n", lane_id, i, smem_ptr[0], smem_ptr);
      cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(frag_ptr[i], smem_ptr);
      smem_ptr += kSwizzleTileSize * kStrideM * kLoadVectorSize;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int SmemDimM_>
class SwizzleTileLoader<SmemDimM_, 32> {
 public:
  static constexpr int SmemDimM = SmemDimM_;
  static constexpr int SmemDimK = 32;
  static constexpr int kLoadVectorSize = 16;  // one cp.async loads 16 bytes
  static constexpr int kBlockSize = SmemDimM * SmemDimK;
  static constexpr int kTiles = (SmemDimM / 8) * (SmemDimK / 16);

  // Swizzle pattern is 2x16
  static constexpr int kSwizzleK = SmemDimK / kLoadVectorSize;
  static_assert(kSwizzleK == cute::_2::value);
  static constexpr int kSwizzleM = cute::_16::value;
  static constexpr int kSwizzleTileSize = kSwizzleK * kSwizzleM;
  using Swizzled32 = decltype(cute::composition(cute::Swizzle<1, 0, 3>{},
                                                cute::Layout<cute::Shape<cute::_2, cute::_16>,
                                                             cute::Stride<cute::_1, cute::_2>>{}));

  static constexpr int kThreads = 32;
  static constexpr int kGmemLoadStrideM = kThreads / kSwizzleK;
  static_assert(kGmemLoadStrideM * kSwizzleK == kThreads);

  // During pipelined MMA, each stage (processing a tile) is split
  // into multiple mma iterations. We need to somehow split the
  // the loading of global memory tile into multiple calls, doing
  // our best to help spread these actions across different iterations
  // in a stage.
  static constexpr int kGloadSplit = SmemDimM / kGmemLoadStrideM;

 private:
  /// Pointer to global memory to load data from
  uint8_t const* g_ptr_{nullptr};
  /// Iteration boundaries in the M or N dimension
  int mn_cnt_{0};
  /// Iteration boundaries in the K dimension, in strides of 16
  int k_cnt_{0};
  /// Stride in bytes to advance to next row in m or n dimension
  const int stride_;

 public:
  CUTLASS_DEVICE
  SwizzleTileLoader(
      void const* data_ptr,  ///< Pointer to the global memory tiles
      int byte_stride,       ///< Stride in bytes to advance to next row
      int mn_start,          ///< Starting position in the M or N dimension
      int mn_end,            ///< End position in the M or N dimension
      int k_start,           ///< Starting position in the K dimension
      int k_end,             ///< End position in the K dimension
      int lane_id)           ///< ID of each participating thread
      : stride_(byte_stride) {
#ifndef NDEBUG
    bool assertion_pass = true;
    if (reinterpret_cast<uintptr_t>(data_ptr) % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("data_ptr: %p is not aligned to 16B boundary!\n", data_ptr);
      }
    }
    if (byte_stride % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("byte_stride: %d is not aligned to 16B boundary!\n", byte_stride);
      }
    }
    if (k_start % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_start: %d is not aligned to 16B boundary!\n", k_start);
      }
    }
    if (k_end % kLoadVectorSize != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_end: %d is not aligned to 16B boundary!\n", k_end);
      }
    }
    if (mn_end <= mn_start) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("mn_end: %d is less than or equal to mn_start: %d!\n", mn_end, mn_start);
      }
    }
    if (k_end <= k_start) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_end: %d is less than or equal to k_start: %d!\n", k_end, k_start);
      }
    }
    if (lane_id < 0 || lane_id >= kThreads) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("Warp based loader, lane_id should be [0-32) but it is: %d!\n", lane_id);
      }
    }
    assert(assertion_pass);
#endif

    int lane_m = lane_id / kSwizzleK;
    int lane_k = lane_id % kSwizzleK;
    mn_start += lane_m;
    k_start += lane_k * kLoadVectorSize;

    mn_cnt_ = div_up(mn_end - mn_start, kGmemLoadStrideM);
    k_cnt_ = div_up(k_end - k_start, kSwizzleK * kLoadVectorSize);
    if (mn_cnt_ <= 0 || k_cnt_ <= 0) {
      mn_cnt_ = 0;
      k_cnt_ = 0;
      g_ptr_ = nullptr;
      return;
    }
    g_ptr_ = reinterpret_cast<uint8_t const*>(data_ptr) + mn_start * byte_stride + k_start;
    // if (lane_id == 0)
    //   printf("lane_id: %d, mn_start: %d, mn_end: %d, k_start: %d, k_end: %d, g_ptr: %p\n", lane_id, mn_start, mn_end, k_start, k_end, g_ptr_);
  }

  /**
   * @brief Load a row major tile (SmemDimM, 32) from global memory to shared memory
   */
  CUTLASS_DEVICE
  void load_to_smem(const int lane_id, void* smem) {
    // The swizzle pattern is 2x16, same size as kThreads
    static_assert(kGmemLoadStrideM == kSwizzleM);
    const uint8_t* data_ptr = g_ptr_;
    uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem) + Swizzled32{}(lane_id)*kLoadVectorSize;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < SmemDimM / kSwizzleM; ++i) {
      cutlass::arch::cp_async<kLoadVectorSize, cutlass::arch::CacheOperation::Global>(
          smem_ptr, data_ptr, g_ptr_ != nullptr && i < mn_cnt_);
      data_ptr += stride_ * kGmemLoadStrideM;
      smem_ptr += kSwizzleTileSize * kLoadVectorSize;
    }
  }

  CUTLASS_DEVICE
  void load_to_smem_split(const int lane_id, void* smem, const int split_idx) {
    // Here we rely on the fact that kThreads is 32, same as the swizzle pattern size
    static_assert(kGmemLoadStrideM == kSwizzleM);

    const uint8_t* split_ptr = g_ptr_ + split_idx * stride_ * kGmemLoadStrideM;
    uint8_t* split_smem_ptr = reinterpret_cast<uint8_t*>(smem) + Swizzled32{}(lane_id)*kLoadVectorSize + split_idx * kSwizzleTileSize * kLoadVectorSize;

    cutlass::arch::cp_async<kLoadVectorSize, cutlass::arch::CacheOperation::Global>(
        split_smem_ptr, split_ptr, g_ptr_ != nullptr && split_idx < mn_cnt_);
  }

  /**
   * @brief Advance global memory pointer to the next tile in the K dimension
   */
  CUTLASS_DEVICE
  SwizzleTileLoader& operator++() {
    --k_cnt_;
    if (k_cnt_ > 0) {
      g_ptr_ += kLoadVectorSize * kSwizzleK;
    } else {
      g_ptr_ = nullptr;
    }
    return *this;
  }

  /**
   * @brief Load a ribbin of (SmemDimM, 32) from shared memory to fragment,
   * fitting fp16 gemm sm80 tensor core shape, where k = 16 x sizeof(fp16)
   */
  CUTLASS_DEVICE
  void load_fragment_k32(const int lane_id, void const* smem, int offset_k, void* frag) {
#ifndef NDEBUG
    bool assert_fail = false;
    if (offset_k != 0) {
      assert_fail = true;
      if (lane_id == 0) {
        printf("Invalid offset_k: %d!\n", offset_k);
      }
    }
    if ((SmemDimM % 16) != 0) {
      // 2x2 tiles per load: 16 threads on the M dim and 2 on the K dim
      // and don't want to deal with left over M
      assert_fail = true;
      if (lane_id == 0) {
        printf("SmemDimM: %d two small, cannot use ldmatrix fully!\n", SmemDimM);
      }
    }
    assert(assert_fail == false);
#endif

    constexpr int kStrideM = 16 / kSwizzleM;  // Span 1 swizzle patterns on M dim
    int m_lane_id = lane_id % 16;
    int k_lane_id = lane_id / 16;  // 0 or 1

    int swizzled_id = Swizzled32{}(k_lane_id, m_lane_id);
    // printf("lane_id: %d, m_lane_id: %d, k_lane_id: %d, swizzled_id: %d\n", lane_id, m_lane_id, k_lane_id, swizzled_id);
    const uint8_t* smem_ptr = reinterpret_cast<const uint8_t*>(smem) + swizzled_id * kLoadVectorSize;

    using FragType = cutlass::Array<unsigned, 4>;
    FragType* frag_ptr = reinterpret_cast<FragType*>(frag);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < SmemDimM / 16; ++i) {
      // printf("lane_id: %d, load %d, val: %d, smem_ptr: %p\n", lane_id, i, smem_ptr[0], smem_ptr);
      cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(frag_ptr[i], smem_ptr);
      smem_ptr += kSwizzleTileSize * kStrideM * kLoadVectorSize;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace mickey
