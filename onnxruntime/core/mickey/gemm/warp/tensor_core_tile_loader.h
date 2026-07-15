/***************************************************************************************************
 * Copyright (c) Microsoft.
 * Licensed under the MIT license.
 *
 * @file warp/tensor_core_tile_loader.h
 * @brief Load matrix tiles from global memory to shared memory, in a way the shared memory
 * tiles can be readily loaded using ldmatrix instruction while avoiding bank conflicts.
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

/////////////////////////////////////////////////////////////////////////////////////////////////

///  Tensor Core tile loader, load tiles from global memory to shared memory using ld.async,
///  so that the shared memory tiles can be loaded to registers using ldmatrix instruction while
///  minimizing bank conflicts, without using memory swizzling
///
template <
    /// Number of tiles in the M or N dimension
    int MNTiles,
    /// Number of tiles in the K dimension
    int KTiles>
class TensorCoreTileLoader {
 public:
  // Number of tiles must be loaded from global memory to shared memory with a single ld.async
  // instruction by each thread in the warp, and a single ldmatrix instruction by the warp.
  static constexpr int kMNTiles = MNTiles;
  static constexpr int kKTiles = KTiles;
  static constexpr int kTiles = kMNTiles * kKTiles;
  static_assert(kTiles == 1 || kTiles == 2 || kTiles == 4, "Number of tiles must be 1, 2 or 4");

  static constexpr int kMNThreads = kMNTiles * 8;
  static constexpr int kKThreads = kKTiles;
  static constexpr int kThreads = kMNThreads * kKThreads;

  /// Each tensor core tile is 16x8 in size
  static constexpr int kMNStride = kMNTiles * 8;
  static constexpr int kKStride = kKTiles * 16;
  static constexpr int kByteSize = kTiles * 16 * 8;

 private:
  /// Pointer to global memory to load data from
  uint8_t const* g_ptr_{nullptr};
  /// Iteration boundaries in the M or N dimension
  int mn_cnt_{0};
  /// Iteration boundaries in the K dimension, in strides of 16
  int k16_cnt_{0};
  /// Stride in bytes to advance to next row in m or n dimension
  const int stride_;
  /// thread id in a warp
  const int lane_id_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  TensorCoreTileLoader(
      void const* data_ptr,  ///< Pointer to the global memory tiles
      int byte_stride,       ///< Stride in bytes to advance to next row
      int mn_start,          ///< Starting position in the M or N dimension
      int mn_end,            ///< End position in the M or N dimension
      int k_start,           ///< Starting position in the K dimension
      int k_end,             ///< End position in the K dimension
      int lane_id)           ///< ID of each participating thread
      : stride_(byte_stride), lane_id_(lane_id) {
#ifndef NDEBUG
    bool assertion_pass = true;
    if (reinterpret_cast<uintptr_t>(data_ptr) % 16 != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("data_ptr: %p is not aligned to 16B boundary!\n", data_ptr);
      }
    }
    if (byte_stride % 16 != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("byte_stride: %d is not aligned to 16B boundary!\n", byte_stride);
      }
    }
    if (k_start % 16 != 0) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("k_start: %d is not aligned to 16B boundary!\n", k_start);
      }
    }
    if (k_end % 16 != 0) {
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
    if (lane_id < 0 || lane_id >= 32) {
      assertion_pass = false;
      if (lane_id == 0) {
        printf("Warp based loader, lane_id should be [0-32) but it is: %d!\n", lane_id);
      }
    }
    assert(assertion_pass);
#endif

    if constexpr (kThreads < 32) {
      if (lane_id >= kThreads) {
        g_ptr_ = nullptr;
        mn_cnt_ = 0;
        k16_cnt_ = 0;
        return;
      }
    }

    uint8_t const* byte_ptr = reinterpret_cast<uint8_t const*>(data_ptr);
    byte_ptr += mn_start * byte_stride + k_start;
    mn_cnt_ = mn_end - mn_start;
    k16_cnt_ = (k_end - k_start) / 16;

    /// Adjcent threads points to different rows, the same way as ldmatrix
    /// loads from shared memory to registers. The goal is to avoid bank conflicts
    /// in shared memory.
    ///
    ///  Global                        Shared
    ///   T0 , T16
    ///   T1 , T17
    ///   T2 , T18
    ///   T3 , T19
    ///   T4 , T20              T0,  T1,  T2,  T3,  T4,  T5,  T6,  T7
    ///   T5 , T21              T8,  T9,  T10, T11, T12, T13, T14, T15
    ///   T6 , T22      =>      T16, T17, T18, T19, T20, T21, T22, T23
    ///   T7 , T23              T24, T25, T26, T27, T28, T29, T30, T31
    ///   T8 , T24
    ///   T9 , T25
    ///   T10, T26
    ///   T11, T27
    ///   T12, T28
    ///   T13, T29
    ///   T14, T30
    ///   T15, T31

    int mn_lane_id = mod_power2<kMNThreads>(lane_id);
    int k_lane_id = lane_id / kMNThreads;
    if (mn_lane_id < mn_cnt_ && k_lane_id < k16_cnt_) {
      g_ptr_ = byte_ptr + mn_lane_id * byte_stride + k_lane_id * 16;
      mn_cnt_ -= mn_lane_id;
      k16_cnt_ -= k_lane_id;
    } else {
      g_ptr_ = nullptr;
      mn_cnt_ = 0;
      k16_cnt_ = 0;
    }
  }

  /**
   * @brief Get the pointer to the shared memory location for the current lane
   * @param smem_ptr pointer to the shared memory location for the warp.
   */
  template <typename T>
  CUTLASS_DEVICE T* get_smem_lane_ptr(T* smem_ptr) const {
    if constexpr (kThreads < 32) {
      static_assert(kThreads & (kThreads - 1) == 0, "kThreads must be power of 2");
      return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(smem_ptr) + ((lane_id_ & (kThreads - 1)) << 4));
    } else {
      return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(smem_ptr) + (lane_id_ << 4));
    }
  }

  template <typename T>
  CUTLASS_DEVICE T* get_smem_warp_base_ptr(T* smem_lane_ptr) const {
    if constexpr (kThreads < 32) {
      static_assert(kThreads & (kThreads - 1) == 0, "kThreads must be power of 2");
      return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(smem_lane_ptr) - ((lane_id_ & (kThreads - 1)) << 4));
    } else {
      return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(smem_lane_ptr) - (lane_id_ << 4));
    }
  }

  /// Loads a tile from global memory to shared memory
  CUTLASS_DEVICE
  void load_to(void* smem_lane_ptr) const {
    if constexpr (kThreads < 32) {
      if (lane_id_ >= kThreads) {
        return;
      }
    }
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
        smem_lane_ptr, g_ptr_, g_ptr_ != nullptr);
  }

  /// Load from next position in the M or N dimension
  CUTLASS_DEVICE
  void load_with_mn_offset(void* smem_lane_ptr, int mn_offset) const {
    if constexpr (kThreads < 32) {
      if (lane_id_ >= kThreads) {
        return;
      }
    }
    assert(mn_offset > 0 && mod_power2<kMNStride>(mn_offset) == 0);
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
        smem_lane_ptr, g_ptr_ + mn_offset * stride_, g_ptr_ != nullptr && mn_offset < mn_cnt_);
  }

  /// Advances to the next position in the K dimension
  CUTLASS_DEVICE
  TensorCoreTileLoader& operator++() {
    if (g_ptr_ == nullptr) {
      return *this;
    }

    k16_cnt_ -= kKTiles;
    if (k16_cnt_ > 0) {
      g_ptr_ += 16 * kKTiles;
    } else {
      g_ptr_ = nullptr;
    }
    return *this;
  }

  template <int MNLoads>
  CUTLASS_DEVICE void load_lateral_n(void* smem_lane_ptr) const {
    uint8_t* smem_bytes = reinterpret_cast<uint8_t*>(smem_lane_ptr);
    this->load_to(smem_bytes);
    smem_bytes += kByteSize;
    CUTLASS_PRAGMA_UNROLL
    for (int n_load = 1; n_load < MNLoads; ++n_load) {
      this->load_with_mn_offset(smem_bytes, n_load * kMNStride);
      smem_bytes += kByteSize;
    }
  }

  CUTLASS_DEVICE
  static void ldmatrix_sync(cutlass::Array<unsigned, kTiles>& frag, void const* smem_lane_ptr) {
    cutlass::arch::ldsm<cutlass::layout::RowMajor, kTiles>(frag, smem_lane_ptr);
  }

  template <typename T1, typename T2, int Loads, int Size = (kTiles * sizeof(unsigned) * Loads) / sizeof(T1)>
  CUTLASS_DEVICE static void multi_ldmatrix_sync(cutlass::Array<T1, Size>& fragment, T2 const*& smem_lane_ptr) {
    static_assert(sizeof(unsigned) * kTiles * Loads == sizeof(T1) * Size, "Fragment size mismatch");
    cutlass::Array<unsigned, kTiles>* ptr =
        reinterpret_cast<cutlass::Array<unsigned, kTiles>*>(fragment.data());

    CUTLASS_PRAGMA_UNROLL
    for (int m_load = 0; m_load < Loads; ++m_load, ++ptr) {
      ldmatrix_sync(*ptr, smem_lane_ptr);
      smem_lane_ptr += kByteSize / sizeof(T2);
    }
  }
};

}  // namespace warp
}  // namespace gemm
}  // namespace mickey
