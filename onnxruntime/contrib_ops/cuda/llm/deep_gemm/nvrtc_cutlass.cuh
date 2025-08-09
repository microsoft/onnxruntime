/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// SM90
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 0))
#define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1
#if (!defined(CUTLASS_ARCH_MMA_SM90_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900)
#define CUTLASS_ARCH_MMA_SM90_ENABLED 1

#if (!defined(CUTLASS_ARCH_MMA_SM90A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM90_ALL))
#define CUTLASS_ARCH_MMA_SM90A_ENABLED 1
#endif
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 2)
#define CUTLASS_ARCH_MMA_SPARSE_SM90_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM90 Modifiable
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
#define CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED 1
#if (!defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900)
#define CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED 1

#if (!defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM90_ALL))
#define CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90A_ENABLED 1
#endif
#endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM90 F64
#if (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8))
#define CUTLASS_ARCH_MMA_SM90_F64_MMA_SUPPORTED 1
#if (!defined(CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
#define CUTLASS_ARCH_MMA_SM90_F64_MMA_ENABLED 1
#endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// TMA instructions
#if defined(CUTLASS_ARCH_MMA_SM90_ENABLED)
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED)
#define CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED
#endif

// STSM
#if defined(CUTLASS_ARCH_MMA_SM90_ENABLED)
#define CUTE_ARCH_STSM_SM90_ENABLED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))))
#define CUTE_ARCH_CLUSTER_SM90_ENABLED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#define CUTE_ARCH_ELECT_ONE_SM90_ENABLED
#endif

#ifndef CUDA_CTA_RECONFIG_ACTIVATED
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL))
#define CUDA_CTA_RECONFIG_ACTIVATED 1
#endif
#endif

#ifndef CU_TENSOR_MAP_NUM_QWORDS
#define CU_TENSOR_MAP_NUM_QWORDS 16

struct CUtensorMap_st {
#if defined(__cplusplus) && (__cplusplus >= 201103L)
  alignas(64)
#elif __STDC_VERSION__ >= 201112L
  _Alignas(64)
#endif
      cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
};

using CUtensorMap = CUtensorMap_st;
#endif

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE __forceinline__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE inline
#define CUTLASS_DEVICE inline
#endif

#define CUTLASS_UNUSED(expr) \
  do {                       \
    ;                        \
  } while (&expr != &expr)
#define CUTLASS_ASSERT(x) assert(x)

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define CUTE_HOST_DEVICE __forceinline__ __host__ __device__
#define CUTE_DEVICE __forceinline__ __device__
#define CUTE_HOST __forceinline__ __host__
#else
#define CUTE_HOST_DEVICE inline
#define CUTE_DEVICE inline
#define CUTE_HOST inline
#endif  // CUTE_HOST_DEVICE, CUTE_DEVICE

#if defined(__CUDA_ARCH__)
#define CUTE_INVALID_CONTROL_PATH(x) \
  assert(0 && x);                    \
  printf(x);                         \
  __brkpt()
#else
#define CUTE_INVALID_CONTROL_PATH(x) \
  assert(0 && x);                    \
  printf(x)
#endif

#define CUTLASS_HOST __host__
#define CUTLASS_GLOBAL __global__ static

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && (__CUDACC_VER_MAJOR__ >= 12)
#define CUDA_BARRIER_ENABLED 1
#else
#define CUDA_BARRIER_ENABLED 0
#endif

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ENABLE_SYNCLOG)

constexpr uint32_t synclog_cap = 1 << 26;

inline std::mutex synclog_mutex;
inline std::vector<uint32_t*> synclog_buf_list;
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
inline __device__ uint32_t* synclog_buf;
#endif

CUTLASS_DEVICE
uint32_t* synclog_alloc(uint32_t n) {
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  uint32_t* buf = synclog_buf;
  if (buf == nullptr)
    return nullptr;
  uint32_t last = atomicAdd(&buf[0], n);
  if (last + n < synclog_cap)
    return buf + last + 1;
  if (last >= synclog_cap)
    atomicAdd(&buf[0], -n);
#endif
  return nullptr;
}

CUTLASS_DEVICE
void synclog_emit_prefix(uint32_t* to, uint32_t header, uint32_t line) {
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  uint64_t time64;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(time64) :);
  to[0] = header;
  to[1] = line;
  to[2] = time64;
  to[3] = time64 >> 32;
  to[4] = threadIdx.x;
  to[5] = threadIdx.y;
  to[6] = threadIdx.z;
  to[7] = blockIdx.x;
  to[8] = blockIdx.y;
  to[9] = blockIdx.z;
#endif
}

constexpr uint32_t synclog_header_none = 0;
constexpr uint32_t synclog_length_prefix = 1 + 1 + 2 + 3 + 3;

constexpr bool synclog_enable_syncthreads = true;
constexpr uint32_t synclog_header_syncthreads = 1;
constexpr uint32_t synclog_length_syncthreads = synclog_length_prefix + 0;

constexpr bool synclog_enable_syncwarp = true;
constexpr uint32_t synclog_header_syncwarp = 2;
constexpr uint32_t synclog_length_syncwarp = synclog_length_prefix + 0;

constexpr bool synclog_enable_named_barrier_arrive_and_wait = true;
constexpr uint32_t synclog_header_named_barrier_arrive_and_wait = 3;
constexpr uint32_t synclog_length_named_barrier_arrive_and_wait = synclog_length_prefix + 2;

constexpr bool synclog_enable_named_barrier_arrive = true;
constexpr uint32_t synclog_header_named_barrier_arrive = 4;
constexpr uint32_t synclog_length_named_barrier_arrive = synclog_length_prefix + 2;

constexpr bool synclog_enable_cluster_barrier_init = true;
constexpr uint32_t synclog_header_cluster_barrier_init = 5;
constexpr uint32_t synclog_length_cluster_barrier_init = synclog_length_prefix + 2;

constexpr bool synclog_enable_cluster_barrier_wait = true;
constexpr uint32_t synclog_header_cluster_barrier_wait = 6;
constexpr uint32_t synclog_length_cluster_barrier_wait = synclog_length_prefix + 4;

constexpr bool synclog_enable_cluster_barrier_test_wait = true;
constexpr uint32_t synclog_header_cluster_barrier_test_wait = 7;
constexpr uint32_t synclog_length_cluster_barrier_test_wait = synclog_length_prefix + 5;

constexpr bool synclog_enable_cluster_barrier_try_wait = true;
constexpr uint32_t synclog_header_cluster_barrier_try_wait = 8;
constexpr uint32_t synclog_length_cluster_barrier_try_wait = synclog_length_prefix + 4;

constexpr bool synclog_enable_cluster_barrier_arrive_cluster = true;
constexpr uint32_t synclog_header_cluster_barrier_arrive_cluster = 9;
constexpr uint32_t synclog_length_cluster_barrier_arrive_cluster = synclog_length_prefix + 5;

constexpr bool synclog_enable_cluster_barrier_arrive = true;
constexpr uint32_t synclog_header_cluster_barrier_arrive = 10;
constexpr uint32_t synclog_length_cluster_barrier_arrive = synclog_length_prefix + 3;

constexpr bool synclog_enable_cluster_barrier_invalidate = true;
constexpr uint32_t synclog_header_cluster_barrier_invalidate = 11;
constexpr uint32_t synclog_length_cluster_barrier_invalidate = synclog_length_prefix + 3;

constexpr bool synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_arrive_and_expect_tx = 12;
constexpr uint32_t synclog_length_cluster_transaction_barrier_arrive_and_expect_tx = synclog_length_prefix + 4;

constexpr bool synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx_cluster = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_arrive_and_expect_tx_cluster = 13;
constexpr uint32_t synclog_length_cluster_transaction_barrier_arrive_and_expect_tx_cluster = synclog_length_prefix + 6;

constexpr bool synclog_enable_cluster_transaction_barrier_expect_transaction = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_expect_transaction = 14;
constexpr uint32_t synclog_length_cluster_transaction_barrier_expect_transaction = synclog_length_prefix + 4;

constexpr bool synclog_enable_cluster_transaction_barrier_complete_transaction = true;
constexpr uint32_t synclog_header_cluster_transaction_barrier_complete_transaction = 15;
constexpr uint32_t synclog_length_cluster_transaction_barrier_complete_transaction = synclog_length_prefix + 6;

constexpr bool synclog_enable_fence_barrier_init = true;
constexpr uint32_t synclog_header_fence_barrier_init = 16;
constexpr uint32_t synclog_length_fence_barrier_init = synclog_length_prefix + 0;

constexpr bool synclog_enable_fence_view_async_shared = true;
constexpr uint32_t synclog_header_fence_view_async_shared = 17;
constexpr uint32_t synclog_length_fence_view_async_shared = synclog_length_prefix + 0;

constexpr bool synclog_enable_cp_async_wait = true;
constexpr uint32_t synclog_header_cp_async_wait = 18;
constexpr uint32_t synclog_length_cp_async_wait = synclog_length_prefix + 1;

constexpr bool synclog_enable_cp_async_wait_all = true;
constexpr uint32_t synclog_header_cp_async_wait_all = 19;
constexpr uint32_t synclog_length_cp_async_wait_all = synclog_length_prefix + 0;

constexpr bool synclog_enable_cp_async_fence = true;
constexpr uint32_t synclog_header_cp_async_fence = 20;
constexpr uint32_t synclog_length_cp_async_fence = synclog_length_prefix + 0;

constexpr bool synclog_enable_cp_async_nan = true;
constexpr uint32_t synclog_header_cp_async_nan = 21;
constexpr uint32_t synclog_length_cp_async_nan = synclog_length_prefix + 4;

constexpr bool synclog_enable_cp_async_zfill = true;
constexpr uint32_t synclog_header_cp_async_zfill = 22;
constexpr uint32_t synclog_length_cp_async_zfill = synclog_length_prefix + 5;

constexpr bool synclog_enable_cp_async = true;
constexpr uint32_t synclog_header_cp_async = 23;
constexpr uint32_t synclog_length_cp_async = synclog_length_prefix + 5;

constexpr bool synclog_enable_tma_load = true;
constexpr uint32_t synclog_header_tma_load = 24;
constexpr uint32_t synclog_length_tma_load = synclog_length_prefix + 4;

constexpr bool synclog_enable_tma_store = true;
constexpr uint32_t synclog_header_tma_store = 25;
constexpr uint32_t synclog_length_tma_store = synclog_length_prefix + 3;

constexpr bool synclog_enable_tma_store_arrive = true;
constexpr uint32_t synclog_header_tma_store_arrive = 26;
constexpr uint32_t synclog_length_tma_store_arrive = synclog_length_prefix + 0;

constexpr bool synclog_enable_tma_store_wait = true;
constexpr uint32_t synclog_header_tma_store_wait = 27;
constexpr uint32_t synclog_length_tma_store_wait = synclog_length_prefix + 1;

constexpr bool synclog_enable_warpgroup_arrive = true;
constexpr uint32_t synclog_header_warpgroup_arrive = 28;
constexpr uint32_t synclog_length_warpgroup_arrive = synclog_length_prefix + 0;

constexpr bool synclog_enable_warpgroup_wait = true;
constexpr uint32_t synclog_header_warpgroup_wait = 29;
constexpr uint32_t synclog_length_warpgroup_wait = synclog_length_prefix + 1;

constexpr bool synclog_enable_warpgroup_commit_batch = true;
constexpr uint32_t synclog_header_warpgroup_commit_batch = 30;
constexpr uint32_t synclog_length_warpgroup_commit_batch = synclog_length_prefix + 0;

constexpr bool synclog_enable_wgmma_reg_smem = true;
constexpr uint32_t synclog_header_wgmma_reg_smem = 31;
constexpr uint32_t synclog_length_wgmma_reg_smem = synclog_length_prefix + 2;

constexpr bool synclog_enable_wgmma_smem_smem = true;
constexpr uint32_t synclog_header_wgmma_smem_smem = 32;
constexpr uint32_t synclog_length_wgmma_smem_smem = synclog_length_prefix + 4;

constexpr bool synclog_enable_cpasync_barrier_arrive = true;
constexpr uint32_t synclog_header_cpasync_barrier_arrive = 33;
constexpr uint32_t synclog_length_cpasync_barrier_arrive = synclog_length_prefix + 3;

CUTLASS_DEVICE
bool synclog_condition_emit() {
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  return threadIdx.x % NumThreadsPerWarp == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
#else
  return 0;
#endif
}

CUTLASS_DEVICE
bool synclog_condition_print() {
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
#else
  return false;
#endif
}

CUTLASS_DEVICE
void synclog_print_prefix(char const* header, uint32_t at) {
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  uint32_t line = synclog_buf[at + 1];
  uint32_t timeLo = synclog_buf[at + 2];
  uint32_t timeHi = synclog_buf[at + 3];
  uint32_t threadIdxX = synclog_buf[at + 4];
  uint32_t threadIdxY = synclog_buf[at + 5];
  uint32_t threadIdxZ = synclog_buf[at + 6];
  uint32_t blockIdxX = synclog_buf[at + 7];
  uint32_t blockIdxY = synclog_buf[at + 8];
  uint32_t blockIdxZ = synclog_buf[at + 9];
  printf("%s line=%u time=%lu thread=%u,%u,%u block=%u,%u,%u ", header, line, (uint64_t)timeHi << 32 | timeLo,
         threadIdxX, threadIdxY, threadIdxZ, blockIdxX, blockIdxY, blockIdxZ);
#endif
}

CUTLASS_DEVICE
uint64_t synclog_mbarrier_bits(uint32_t smem_addr) {
  uint64_t bits = 0;
  asm volatile(
      "mbarrier.inval.shared::cta.b64 [%1];\n"
      "ld.shared::cta.b64 %0, [%1];\n"
      : "=l"(bits)
      : "r"(smem_addr));
  return bits;
}

CUTLASS_DEVICE
void synclog_print_wgmma_desc(char const* str, uint32_t lo, uint32_t hi, char const* sep) {
  CUTLASS_UNUSED(hi);
  uint32_t smem_int_ptr = (lo & ((1 << 14) - 1)) << 4;
  printf("%s_smem_int_ptr=%u%s", str, smem_int_ptr, sep);
}

#endif  // defined(CUTLASS_ENABLE_SYNCLOG)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline void synclog_setup() {
#if defined(CUTLASS_ENABLE_SYNCLOG)
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  std::scoped_lock lock(synclog_mutex);
  auto fail = []() {
    fprintf(stderr, "synclog_setup() failed\n");
    std::terminate();
  };
  int orig_device = 0;
  if (cudaGetDevice(&orig_device) != cudaSuccess) {
    fail();
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
    fail();
  }
  if (synclog_buf_list.size() == 0) {
    for (int device = 0; device < device_count; device++) {
      uint32_t* buf = 0;
      if (cudaSetDevice(device) != cudaSuccess || cudaMalloc(&buf, synclog_cap * sizeof(uint32_t)) != cudaSuccess) {
        fail();
      }
      synclog_buf_list.push_back(buf);
    }
  }
  for (int device = 0; device < device_count; device++) {
    uint32_t* buf = synclog_buf_list.at(device);
    if (cudaSetDevice(device) != cudaSuccess || cudaMemset(buf, 0, synclog_cap * sizeof(uint32_t)) != cudaSuccess || cudaMemcpyToSymbol(synclog_buf, &buf, sizeof(buf)) != cudaSuccess) {
      fail();
    }
  }
  if (cudaSetDevice(orig_device) != cudaSuccess) {
    fail();
  }
#endif
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_syncthreads(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_syncthreads)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_syncthreads);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_syncthreads, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_syncwarp(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_syncwarp)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_syncwarp);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_syncwarp, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_named_barrier_arrive_and_wait(uint32_t line, uint32_t num_threads, uint32_t barrier_id) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_named_barrier_arrive_and_wait)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_named_barrier_arrive_and_wait);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_named_barrier_arrive_and_wait, line);
  to[synclog_length_prefix + 0] = num_threads;
  to[synclog_length_prefix + 1] = barrier_id;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(num_threads);
  CUTLASS_UNUSED(barrier_id);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_named_barrier_arrive(uint32_t line, uint32_t num_threads, uint32_t barrier_id) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_named_barrier_arrive)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_named_barrier_arrive);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_named_barrier_arrive, line);
  to[synclog_length_prefix + 0] = num_threads;
  to[synclog_length_prefix + 1] = barrier_id;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(num_threads);
  CUTLASS_UNUSED(barrier_id);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_init(uint32_t line, uint32_t smem_addr, uint32_t arrive_count) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_init)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_init);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_init, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = arrive_count;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(arrive_count);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_wait(uint32_t line, uint32_t smem_addr, uint32_t phase) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_wait)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_wait);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_wait, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = phase;
  to[synclog_length_prefix + 2] = bits;
  to[synclog_length_prefix + 3] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(phase);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_test_wait(uint32_t line, uint32_t smem_addr, uint32_t phase, uint32_t pred) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_test_wait)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_test_wait);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_test_wait, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = phase;
  to[synclog_length_prefix + 2] = pred;
  to[synclog_length_prefix + 3] = bits;
  to[synclog_length_prefix + 4] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(phase);
  CUTLASS_UNUSED(pred);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_try_wait(uint32_t line, uint32_t smem_addr, uint32_t phase) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_try_wait)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_try_wait);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_try_wait, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = phase;
  to[synclog_length_prefix + 2] = bits;
  to[synclog_length_prefix + 3] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(phase);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_arrive_cluster(uint32_t line, uint32_t smem_addr, uint32_t cta_id, uint32_t pred) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_arrive_cluster)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_arrive_cluster);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_arrive_cluster, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = cta_id;
  to[synclog_length_prefix + 2] = pred;
  to[synclog_length_prefix + 3] = bits;
  to[synclog_length_prefix + 4] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(cta_id);
  CUTLASS_UNUSED(pred);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_arrive(uint32_t line, uint32_t smem_addr) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_arrive)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_arrive);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_arrive, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = bits;
  to[synclog_length_prefix + 2] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_barrier_invalidate(uint32_t line, uint32_t smem_addr) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_barrier_invalidate)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_barrier_invalidate);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_barrier_invalidate, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = bits;
  to[synclog_length_prefix + 2] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx(
    uint32_t line, uint32_t smem_addr, uint32_t transaction_bytes) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_arrive_and_expect_tx);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_arrive_and_expect_tx, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = transaction_bytes;
  to[synclog_length_prefix + 2] = bits;
  to[synclog_length_prefix + 3] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(transaction_bytes);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx_cluster(
    uint32_t line, uint32_t smem_addr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx_cluster)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_arrive_and_expect_tx_cluster);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_arrive_and_expect_tx_cluster, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = transaction_bytes;
  to[synclog_length_prefix + 2] = cta_id;
  to[synclog_length_prefix + 3] = pred;
  to[synclog_length_prefix + 4] = bits;
  to[synclog_length_prefix + 5] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(transaction_bytes);
  CUTLASS_UNUSED(cta_id);
  CUTLASS_UNUSED(pred);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_expect_transaction(
    uint32_t line, uint32_t smem_addr, uint32_t transaction_bytes) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_expect_transaction)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_expect_transaction);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_expect_transaction, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = transaction_bytes;
  to[synclog_length_prefix + 2] = bits;
  to[synclog_length_prefix + 2] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(transaction_bytes);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cluster_transaction_barrier_complete_transaction(
    uint32_t line, uint32_t smem_addr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cluster_transaction_barrier_complete_transaction)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cluster_transaction_barrier_complete_transaction);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cluster_transaction_barrier_complete_transaction, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = dst_cta_id;
  to[synclog_length_prefix + 2] = transaction_bytes;
  to[synclog_length_prefix + 3] = pred;
  to[synclog_length_prefix + 4] = bits;
  to[synclog_length_prefix + 5] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(dst_cta_id);
  CUTLASS_UNUSED(transaction_bytes);
  CUTLASS_UNUSED(pred);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_fence_barrier_init(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_fence_barrier_init)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_fence_barrier_init);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_fence_barrier_init, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_fence_view_async_shared(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_fence_view_async_shared)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_fence_view_async_shared);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_fence_view_async_shared, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_wait(uint32_t line, uint32_t n) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_wait)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_wait);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cp_async_wait, line);
  to[synclog_length_prefix + 0] = n;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(n);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_wait_all(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_wait_all)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_wait_all);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cp_async_wait_all, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_fence(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_fence)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_fence);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cp_async_fence, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_nan(uint32_t line, uint32_t smem_addr, void const* gmem_ptr, uint32_t pred) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_nan)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_nan);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cp_async_nan, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_ptr);
  to[synclog_length_prefix + 2] = (uint32_t)((uint64_t)gmem_ptr >> 32);
  to[synclog_length_prefix + 3] = pred;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(gmem_ptr);
  CUTLASS_UNUSED(pred);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async_zfill(uint32_t line, uint32_t smem_addr, void const* gmem_ptr, uint32_t pred, uint32_t size) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async_zfill)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async_zfill);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cp_async_zfill, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_ptr);
  to[synclog_length_prefix + 2] = (uint32_t)((uint64_t)gmem_ptr >> 32);
  to[synclog_length_prefix + 3] = pred;
  to[synclog_length_prefix + 4] = size;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(gmem_ptr);
  CUTLASS_UNUSED(pred);
  CUTLASS_UNUSED(size);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cp_async(uint32_t line, uint32_t smem_addr, void const* gmem_ptr, uint32_t pred, uint32_t size) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cp_async)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_cp_async);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cp_async, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_ptr);
  to[synclog_length_prefix + 2] = (uint32_t)((uint64_t)gmem_ptr >> 32);
  to[synclog_length_prefix + 3] = pred;
  to[synclog_length_prefix + 4] = size;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
  CUTLASS_UNUSED(gmem_ptr);
  CUTLASS_UNUSED(pred);
  CUTLASS_UNUSED(size);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_load(uint32_t line, uint64_t gmem_int_desc, uint32_t smem_int_mbar, uint32_t smem_int_ptr) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_load)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_tma_load);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_tma_load, line);
  to[synclog_length_prefix + 0] = (uint32_t)((uint64_t)gmem_int_desc);
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_int_desc >> 32);
  to[synclog_length_prefix + 2] = smem_int_mbar;
  to[synclog_length_prefix + 3] = smem_int_ptr;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(gmem_int_desc);
  CUTLASS_UNUSED(smem_int_mbar);
  CUTLASS_UNUSED(smem_int_ptr);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_store(uint32_t line, uint64_t gmem_int_desc, uint32_t smem_int_ptr) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_store)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_tma_store);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_tma_store, line);
  to[synclog_length_prefix + 0] = (uint32_t)((uint64_t)gmem_int_desc);
  to[synclog_length_prefix + 1] = (uint32_t)((uint64_t)gmem_int_desc >> 32);
  to[synclog_length_prefix + 2] = smem_int_ptr;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(gmem_int_desc);
  CUTLASS_UNUSED(smem_int_ptr);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_store_arrive(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_store_arrive)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_tma_store_arrive);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_tma_store_arrive, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_tma_store_wait(uint32_t line, uint32_t count) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_tma_store_wait)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_tma_store_wait);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_tma_store_wait, line);
  to[synclog_length_prefix + 0] = count;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(count);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_warpgroup_arrive(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_warpgroup_arrive)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_warpgroup_arrive);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_warpgroup_arrive, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_warpgroup_wait(uint32_t line, uint32_t n) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_warpgroup_wait)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_warpgroup_wait);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_warpgroup_wait, line);
  to[synclog_length_prefix + 0] = n;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(n);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_warpgroup_commit_batch(uint32_t line) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_warpgroup_commit_batch)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_warpgroup_commit_batch);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_warpgroup_commit_batch, line);
#else
  CUTLASS_UNUSED(line);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_wgmma_reg_smem(uint32_t line, uint64_t desc_b) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_wgmma_reg_smem)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_wgmma_reg_smem);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_wgmma_reg_smem, line);
  to[synclog_length_prefix + 0] = desc_b;
  to[synclog_length_prefix + 1] = desc_b >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(desc_b);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_wgmma_smem_smem(uint32_t line, uint64_t desc_a, uint64_t desc_b) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_wgmma_smem_smem)
    return;
  if (!synclog_condition_emit())
    return;
  uint32_t* to = synclog_alloc(synclog_length_wgmma_smem_smem);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_wgmma_smem_smem, line);
  to[synclog_length_prefix + 0] = desc_a;
  to[synclog_length_prefix + 1] = desc_a >> 32;
  to[synclog_length_prefix + 2] = desc_b;
  to[synclog_length_prefix + 3] = desc_b >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(desc_a);
  CUTLASS_UNUSED(desc_b);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

CUTLASS_DEVICE
void synclog_emit_cpasync_barrier_arrive(uint32_t line, uint32_t smem_addr) {
#if defined(CUTLASS_ENABLE_SYNCLOG)
  if constexpr (!synclog_enable_cpasync_barrier_arrive)
    return;
  if (!synclog_condition_emit())
    return;
  uint64_t bits = synclog_mbarrier_bits(smem_addr);
  uint32_t* to = synclog_alloc(synclog_length_cpasync_barrier_arrive);
  if (to == nullptr)
    return;
  synclog_emit_prefix(to, synclog_header_cpasync_barrier_arrive, line);
  to[synclog_length_prefix + 0] = smem_addr;
  to[synclog_length_prefix + 1] = bits;
  to[synclog_length_prefix + 2] = bits >> 32;
#else
  CUTLASS_UNUSED(line);
  CUTLASS_UNUSED(smem_addr);
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

#if !defined(CUTLASS_ENABLE_SYNCLOG)
CUTLASS_DEVICE
#elif defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
static __attribute__((__noinline__)) __device__
#else
static __attribute__((__noinline__))
#endif
void synclog_print() {
#if defined(CUTLASS_ENABLE_SYNCLOG)
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
  if (synclog_buf == nullptr || !synclog_condition_print()) {
    return;
  }
  printf("synclog start\n");
  for (uint32_t at = 1; at < synclog_cap;) {
    uint32_t header = synclog_buf[at];
    if (header == synclog_header_none) {
      break;
    }
    printf("synclog at %u: ", at);
    if constexpr (synclog_enable_syncthreads) {
      if (header == synclog_header_syncthreads) {
        synclog_print_prefix("syncthreads", at);
        at += synclog_length_syncthreads;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_syncwarp) {
      if (header == synclog_header_syncwarp) {
        synclog_print_prefix("syncwarp", at);
        at += synclog_length_syncwarp;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_named_barrier_arrive_and_wait) {
      if (header == synclog_header_named_barrier_arrive_and_wait) {
        synclog_print_prefix("named_barrier_arrive_and_wait", at);
        at += synclog_length_named_barrier_arrive_and_wait;
        printf("num_threads=%u barrier_id=%u\n", synclog_buf[at - 2], synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_named_barrier_arrive) {
      if (header == synclog_header_named_barrier_arrive) {
        synclog_print_prefix("named_barrier_arrive", at);
        at += synclog_length_named_barrier_arrive;
        printf("num_threads=%u barrier_id=%u\n", synclog_buf[at - 2], synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_init) {
      if (header == synclog_header_cluster_barrier_init) {
        synclog_print_prefix("cluster_barrier_init", at);
        at += synclog_length_cluster_barrier_init;
        printf("smem_addr=%u arrive_count=%u\n", synclog_buf[at - 2], synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_wait) {
      if (header == synclog_header_cluster_barrier_wait) {
        synclog_print_prefix("cluster_barrier_wait", at);
        at += synclog_length_cluster_barrier_wait;
        printf("smem_addr=%u phase=%u", synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_test_wait) {
      if (header == synclog_header_cluster_barrier_test_wait) {
        synclog_print_prefix("cluster_barrier_test_wait", at);
        at += synclog_length_cluster_barrier_test_wait;
        printf("smem_addr=%u phase=%u pred=%u", synclog_buf[at - 5], synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_try_wait) {
      if (header == synclog_header_cluster_barrier_try_wait) {
        synclog_print_prefix("cluster_barrier_try_wait", at);
        at += synclog_length_cluster_barrier_try_wait;
        printf("smem_addr=%u phase=%u", synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_arrive_cluster) {
      if (header == synclog_header_cluster_barrier_arrive_cluster) {
        synclog_print_prefix("cluster_barrier_arrive_cluster", at);
        at += synclog_length_cluster_barrier_arrive_cluster;
        printf("smem_addr=%u cta_id=%u pred=%u", synclog_buf[at - 5], synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_arrive) {
      if (header == synclog_header_cluster_barrier_arrive) {
        synclog_print_prefix("cluster_barrier_arrive", at);
        at += synclog_length_cluster_barrier_arrive;
        printf("smem_addr=%u", synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_barrier_invalidate) {
      if (header == synclog_header_cluster_barrier_invalidate) {
        synclog_print_prefix("cluster_barrier_invalidate", at);
        at += synclog_length_cluster_barrier_invalidate;
        printf("smem_addr=%u", synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx) {
      if (header == synclog_header_cluster_transaction_barrier_arrive_and_expect_tx) {
        synclog_print_prefix("cluster_transaction_barrier_arrive_and_expect_tx", at);
        at += synclog_length_cluster_transaction_barrier_arrive_and_expect_tx;
        printf("smem_addr=%u transaction_bytes=%u", synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_arrive_and_expect_tx_cluster) {
      if (header == synclog_header_cluster_transaction_barrier_arrive_and_expect_tx_cluster) {
        synclog_print_prefix("cluster_transaction_barrier_arrive_and_expect_tx_cluster", at);
        at += synclog_length_cluster_transaction_barrier_arrive_and_expect_tx_cluster;
        printf("smem_addr=%u transaction_bytes=%u cta_id=%u pred=%u", synclog_buf[at - 6], synclog_buf[at - 5],
               synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_expect_transaction) {
      if (header == synclog_header_cluster_transaction_barrier_expect_transaction) {
        synclog_print_prefix("cluster_transaction_barrier_expect_transaction", at);
        at += synclog_length_cluster_transaction_barrier_expect_transaction;
        printf("smem_addr=%u transaction_bytes=%u", synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_cluster_transaction_barrier_complete_transaction) {
      if (header == synclog_header_cluster_transaction_barrier_complete_transaction) {
        synclog_print_prefix("cluster_transaction_barrier_complete_transaction", at);
        at += synclog_length_cluster_transaction_barrier_complete_transaction;
        printf("smem_addr=%u dst_cta_id=%u transaction_bytes=%u pred=%u", synclog_buf[at - 6],
               synclog_buf[at - 5], synclog_buf[at - 4], synclog_buf[at - 3]);
        continue;
      }
    }
    if constexpr (synclog_enable_fence_barrier_init) {
      if (header == synclog_header_fence_barrier_init) {
        synclog_print_prefix("fence_barrier_init", at);
        at += synclog_length_fence_barrier_init;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_fence_view_async_shared) {
      if (header == synclog_header_fence_view_async_shared) {
        synclog_print_prefix("fence_view_async_shared", at);
        at += synclog_length_fence_view_async_shared;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_wait) {
      if (header == synclog_header_cp_async_wait) {
        synclog_print_prefix("cp_async_wait", at);
        at += synclog_length_cp_async_wait;
        printf("n=%u\n", synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_wait_all) {
      if (header == synclog_header_cp_async_wait_all) {
        synclog_print_prefix("cp_async_wait_all", at);
        at += synclog_length_cp_async_wait_all;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_fence) {
      if (header == synclog_header_cp_async_fence) {
        synclog_print_prefix("cp_async_fence", at);
        at += synclog_length_cp_async_fence;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_nan) {
      if (header == synclog_header_cp_async_nan) {
        synclog_print_prefix("cp_async_nan", at);
        at += synclog_length_cp_async_nan;
        uint64_t gmem_addr = synclog_buf[at - 3];
        gmem_addr += (uint64_t)synclog_buf[at - 2] << 32;
        printf("smem_addr=%u gmem_addr=%llu pred=%u\n", synclog_buf[at - 4], gmem_addr, synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async_zfill) {
      if (header == synclog_header_cp_async_zfill) {
        synclog_print_prefix("cp_async_zfill", at);
        at += synclog_length_cp_async_zfill;
        uint64_t gmem_addr = synclog_buf[at - 4];
        gmem_addr += (uint64_t)synclog_buf[at - 3] << 32;
        printf("smem_addr=%u gmem_addr=%llu pred=%u size=%u\n", synclog_buf[at - 5], gmem_addr,
               synclog_buf[at - 2], synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_cp_async) {
      if (header == synclog_header_cp_async) {
        synclog_print_prefix("cp_async", at);
        at += synclog_length_cp_async;
        uint64_t gmem_addr = synclog_buf[at - 4];
        gmem_addr += (uint64_t)synclog_buf[at - 3] << 32;
        printf("smem_addr=%u gmem_addr=%llu pred=%u size=%u\n", synclog_buf[at - 5], gmem_addr,
               synclog_buf[at - 2], synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_tma_load) {
      if (header == synclog_header_tma_load) {
        synclog_print_prefix("tma_load", at);
        at += synclog_length_tma_load;
        uint64_t gmem_int_desc = synclog_buf[at - 4];
        gmem_int_desc += (uint64_t)synclog_buf[at - 3] << 32;
        printf("gmem_int_desc=%llu smem_int_mbar=%u smem_int_ptr=%u\n", gmem_int_desc, synclog_buf[at - 2],
               synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_tma_store) {
      if (header == synclog_header_tma_store) {
        synclog_print_prefix("tma_store", at);
        at += synclog_length_tma_store;
        uint64_t gmem_int_desc = synclog_buf[at - 3];
        gmem_int_desc += (uint64_t)synclog_buf[at - 2] << 32;
        printf("gmem_int_desc=%llu smem_int_ptr=%u\n", gmem_int_desc, synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_tma_store_arrive) {
      if (header == synclog_header_tma_store_arrive) {
        synclog_print_prefix("tma_store_arrive", at);
        at += synclog_length_tma_store_arrive;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_tma_store_wait) {
      if (header == synclog_header_tma_store_wait) {
        synclog_print_prefix("tma_store_wait", at);
        at += synclog_length_tma_store_wait;
        printf("count=%u\n", synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_warpgroup_arrive) {
      if (header == synclog_header_warpgroup_arrive) {
        synclog_print_prefix("warpgroup_arrive", at);
        at += synclog_length_warpgroup_arrive;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_warpgroup_wait) {
      if (header == synclog_header_warpgroup_wait) {
        synclog_print_prefix("warpgroup_wait", at);
        at += synclog_length_warpgroup_wait;
        printf("n=%u\n", synclog_buf[at - 1]);
        continue;
      }
    }
    if constexpr (synclog_enable_warpgroup_commit_batch) {
      if (header == synclog_header_warpgroup_commit_batch) {
        synclog_print_prefix("warpgroup_commit_batch", at);
        at += synclog_length_warpgroup_commit_batch;
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_wgmma_reg_smem) {
      if (header == synclog_header_wgmma_reg_smem) {
        synclog_print_prefix("wgmma_reg_smem", at);
        at += synclog_length_wgmma_reg_smem;
        synclog_print_wgmma_desc("desc_b", synclog_buf[at - 2], synclog_buf[at - 1], "");
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_wgmma_smem_smem) {
      if (header == synclog_header_wgmma_smem_smem) {
        synclog_print_prefix("wgmma_smem_smem", at);
        at += synclog_length_wgmma_smem_smem;
        synclog_print_wgmma_desc("desc_a", synclog_buf[at - 4], synclog_buf[at - 3], " ");
        synclog_print_wgmma_desc("desc_b", synclog_buf[at - 2], synclog_buf[at - 1], "");
        printf("\n");
        continue;
      }
    }
    if constexpr (synclog_enable_cpasync_barrier_arrive) {
      if (header == synclog_header_cpasync_barrier_arrive) {
        synclog_print_prefix("cpasync_barrier_arrive", at);
        at += synclog_length_cpasync_barrier_arrive;
        printf("smem_addr=%u", synclog_buf[at - 3]);
        continue;
      }
    }
    asm volatile("brkpt;\n" ::);
  }
  if (synclog_buf[0] >= synclog_cap) {
    printf("synclog was truncated (exceeded capacity of %lu bytes)\n", (synclog_cap - 1) * sizeof(uint32_t));
  }
  printf("synclog end\n");
#endif
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ENABLE_SYNCLOG)
#undef __syncthreads
#define __syncthreads()                                \
  do {                                                 \
    cutlass::arch::synclog_emit_syncthreads(__LINE__); \
    __syncthreads();                                   \
  } while (0)
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)

#if defined(CUTLASS_ENABLE_SYNCLOG)
#undef __syncwarp
#define __syncwarp(...)                             \
  do {                                              \
    cutlass::arch::synclog_emit_syncwarp(__LINE__); \
    __syncwarp(__VA_ARGS__);                        \
  } while (0)
#endif  // defined(CUTLASS_ENABLE_SYNCLOG)

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass

#if defined(__clang__) && defined(__CUDA__)
   //  __cvta_generic_to_shared was added in Clang 14:
   //  https://reviews.llvm.org/D111665
#if __clang_major__ >= 14
#define CUTE_CLANG_SUPPORTS_CVTA_GENERIC_TO_SHARED 1
#endif

// __nvvm_get_smem_pointer added in Clang 14: https://reviews.llvm.org/D111665
// ... but will not work on Windows until Clang 15:
// https://reviews.llvm.org/D122897
#if (!defined(_WIN32) && __clang_major__ >= 14) || __clang_major__ >= 15
#define CUTE_CLANG_SUPPORTS_NVVM_GET_SMEM_POINTER 1
#endif
#endif

#if defined(__NVCC__) || defined(__CUDACC_RTC__)
   // __cvta_generic_to_shared added in CUDA 11+
#if __CUDACC_VER_MAJOR__ >= 11
#define CUTE_NVCC_SUPPORTS_CVTA_GENERIC_TO_SHARED 1
#endif

// __nvvm_get_smem_pointer added in CUDA 10.2
#if __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2
#define CUTE_NVCC_SUPPORTS_NVVM_GET_SMEM_POINTER 1
#endif
#endif

#if CUTE_NVCC_SUPPORTS_CVTA_GENERIC_TO_SHARED || CUTE_CLANG_SUPPORTS_CVTA_GENERIC_TO_SHARED
#define CUTE_CVTA_GENERIC_TO_SHARED_SUPPORTED 1
#endif

#if !defined(CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED) && CUTE_CVTA_GENERIC_TO_SHARED_SUPPORTED && defined(__CUDA_ARCH__)
#define CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED 1
#endif

#if CUTE_NVCC_SUPPORTS_NVVM_GET_SMEM_POINTER || CUTE_CLANG_SUPPORTS_NVVM_GET_SMEM_POINTER
#define CUTE_NVVM_GET_SMEM_POINTER_SUPPORTED 1
#endif

#if !defined(CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED) && CUTE_NVVM_GET_SMEM_POINTER_SUPPORTED && defined(__CUDA_ARCH__)
#define CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED 1
#endif

// Clang 14+ provides a declaration of __nvvm_get_smem_pointer, so we only need
// to provide one for NVCC
#if CUTE_NVCC_SUPPORTS_NVVM_GET_SMEM_POINTER
extern "C" {
// This NVVM intrinsic is subject to change in future versions of CUDA.
// Clients should not call it directly.
CUTE_DEVICE uint32_t __nvvm_get_smem_pointer(void*);
}
#endif

namespace cute {

/// CUTE helper to cast SMEM pointer to unsigned
CUTE_DEVICE
uint32_t cast_smem_ptr_to_uint(void const* const ptr) {
// We prefer to use the new CVTA intrinsics if they are available, otherwise we
// will fall back to the previous internal intrinsics if they are available.
#if CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only
  // available in 10.2].
  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);

  /// CUTE helper to get SMEM pointer
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

#elif CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED

  return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

  uint32_t smem_ptr;

  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
      "smem_ptr; }\n"
      : "=r"(smem_ptr)
      : "l"(ptr));

  return smem_ptr;

#else

  (void)ptr;
  printf("ERROR: cast_smem_ptr_to_uint not supported but used.\n");
  return 0;

#endif
}

}  // namespace cute

namespace cute {

CUTE_DEVICE void cluster_arrive_relaxed() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_arrive() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.arrive.aligned;\n" : :);
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_wait() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

CUTE_DEVICE void cluster_sync() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  cluster_arrive();
  cluster_wait();
#else
  CUTE_INVALID_CONTROL_PATH("CUTE_ARCH_CLUSTER_SM90_ENABLED is not defined");
#endif
}

// Returns the dim3 grid size in terms of number of clusters.
CUTE_DEVICE dim3 cluster_grid_dims() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%nclusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%nclusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of gridDim with __CUDA_ARCH__.
  return gridDim;
#elif defined(_MSC_VER)
  CUTE_INVALID_CONTROL_PATH("cluster_grid_dims() can only be called on device");
  return {0, 0, 0};
#else
  return {0, 0, 0};
#endif
}

// Returns the dim3 cluster rank in the grid.
CUTE_DEVICE dim3 cluster_id_in_grid() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%clusterid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%clusterid.z;\n" : "=r"(z) :);
  return {x, y, z};
#elif defined(__CUDA_ARCH__)
  // MSVC requires protecting use of blockIdx with __CUDA_ARCH__.
  return blockIdx;
#elif defined(_MSC_VER)
  CUTE_INVALID_CONTROL_PATH("cluster_id_in_grid() can only be called on device");
  return {0, 0, 0};
#else
  return {0, 0, 0};
#endif
}

// Returns the relative dim3 block rank local to the cluster.
CUTE_DEVICE dim3 block_id_in_cluster() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
#else
  return {0, 0, 0};
#endif
}

// Returns the dim3 cluster shape.
CUTE_DEVICE dim3 cluster_shape() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(x) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.y;\n" : "=r"(y) :);
  asm volatile("mov.u32 %0, %%cluster_nctaid.z;\n" : "=r"(z) :);
  return {x, y, z};
#else
  return {1, 1, 1};
#endif
}

// Get 1D ctaid in a cluster.
CUTE_DEVICE uint32_t block_rank_in_cluster() {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
#else
  return 0;
#endif
}

// Set the destination block-ID in cluster for a given SMEM Address
CUTE_DEVICE uint32_t set_block_rank(uint32_t smemAddr, uint32_t rank) {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t result;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n" : "=r"(result) : "r"(smemAddr), "r"(rank));
  return result;
#else
  return smemAddr;
#endif
}

// Elect one thread in the warp. The elected thread gets its predicate set to
// true, all others obtain false.
CUTE_HOST_DEVICE uint32_t elect_one_sync() {
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
#elif defined(__CUDA_ARCH__)
  return (threadIdx.x % 32) == 0;
#else
  return true;
#endif
}

struct ElectOneLaneIdReturnType {
  uint32_t is_leader;
  uint32_t leader_lane_id;
};

CUTE_HOST_DEVICE
ElectOneLaneIdReturnType elect_one_leader_sync() {
#if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return {pred, laneid};
#elif defined(__CUDA_ARCH__)
  return {(threadIdx.x % 32) == 0, 0};
#else
  return {true, 0};
#endif
}

// Store value to remote shared memory in the cluster
CUTE_DEVICE
void store_shared_remote(uint32_t value, uint32_t smem_addr, uint32_t mbarrier_addr, uint32_t dst_cta_rank) {
#if defined(CUTE_ARCH_CLUSTER_SM90_ENABLED)
  uint32_t dsmem_addr = set_block_rank(smem_addr, dst_cta_rank);
  uint32_t remote_barrier_addr = set_block_rank(mbarrier_addr, dst_cta_rank);
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 "
      "[%0], %1, [%2];"
      :
      : "r"(dsmem_addr), "r"(value), "r"(remote_barrier_addr));
#endif
}

// Fence for smem stores for subsequent TMA_STORE
CUTE_HOST_DEVICE static void tma_store_fence() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  cutlass::arch::synclog_emit_fence_view_async_shared(__LINE__);
  asm volatile("fence.proxy.async.shared::cta;");
#elif defined(__CUDA_ARCH__)
  CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

// Indicate arrival of warp issuing TMA_STORE
CUTE_HOST_DEVICE static void tma_store_arrive() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  cutlass::arch::synclog_emit_tma_store_arrive(__LINE__);
  asm volatile("cp.async.bulk.commit_group;");
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

// Wait until at most Count committed TMA_STOREs are pending and all prior
// commits are complete
template <int Count>
CUTE_HOST_DEVICE static void tma_store_wait() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(Count) : "memory");
  cutlass::arch::synclog_emit_tma_store_wait(__LINE__, Count);
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

}  // end namespace cute

namespace cutlass {
/// @brief
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts
// This enum class specifies the NamedBarriers reserved by CUTLASS.
enum class ReservedNamedBarriers {
  EpilogueBarrier = 1,
  TransposeBarrier = 2,
  TransformBarrier = 3,
  StreamkBarrier0 = 4,
  StreamkBarrier1 = 5,
  FirstUserBarrier = StreamkBarrier1 + 1
};

class NamedBarrier {
  // Data Members:

  // Range = [1 , NUM_THREADS_PER_CTA]
  // Range % warp-size (i.e 32) == 0
  uint32_t const num_threads_;

  // Range : [0, 15]
  // Note that should be set to the final barrier ID, including
  // ReserveNamedBarrierCount should be considered
  uint32_t const id_;

 public:
  // Constructor for CUTLASS developers:
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers)
      : num_threads_(num_threads), id_(static_cast<uint32_t>(reserved_named_barriers)) {
  }

  // Constructor for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  NamedBarrier(uint32_t num_threads, uint32_t id = 0)
      : num_threads_(num_threads), id_(id + ReservedNamedBarrierCount) {
    CUTLASS_ASSERT(id + ReservedNamedBarrierCount <= HardwareMaxNumNamedBarriers && "Effective barrier_id should not exceed 16.");
  }

  CUTLASS_DEVICE
  void arrive_and_wait() const {
    // Note: The value of id_ is already the final barrier id (set correctly in
    // the constructor).
    NamedBarrier::arrive_and_wait_internal(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive_and_wait_unaligned() const {
    // Note: The value of id_ is already the final barrier id (set correctly in
    // the constructor).
    NamedBarrier::arrive_and_wait_internal_unaligned(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive() const {
    // Note: The value of id_ is already the final barrier id (set correctly in
    // the constructor).
    NamedBarrier::arrive_internal(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void arrive_unaligned() const {
    // Note: The value of id_ is already the final barrier id (set correctly in
    // the constructor).
    NamedBarrier::arrive_internal_unaligned(num_threads_, id_);
  }

  CUTLASS_DEVICE
  void sync() const {
    NamedBarrier::arrive_and_wait();
  }

  //  Static variants

  // Calling interface for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
    arrive_and_wait_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers:
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void arrive_and_wait(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_and_wait_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, uint32_t barrier_id) {
    arrive_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers:
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void arrive(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    arrive_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

  // Calling interface for CUTLASS users:
  // effective barrier ID starts from ReservedNamedBarrierCount
  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, uint32_t barrier_id) {
    sync_internal(num_threads, barrier_id + ReservedNamedBarrierCount);
  }

  // Calling interface for CUTLASS developers:
  // effective barrier ID starts from 0
  CUTLASS_DEVICE
  static void sync(uint32_t num_threads, ReservedNamedBarriers reserved_named_barriers) {
    sync_internal(num_threads, static_cast<int>(reserved_named_barriers));
  }

 private:
  CUTLASS_DEVICE
  static void arrive_and_wait_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_and_wait_internal_unaligned(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void arrive_internal_unaligned(uint32_t num_threads, uint32_t barrier_id) {
#if CUDA_BARRIER_ENABLED
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
    asm volatile("barrier.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void sync_internal(uint32_t num_threads, uint32_t barrier_id) {
    NamedBarrier::arrive_and_wait_internal(num_threads, barrier_id);
  }

 public:
  // Currently we reserve 8 NamedBarriers for CUTLASS' own use cases,
  // while leaving the renaming for general users.
  static const uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(ReservedNamedBarriers::FirstUserBarrier);
  static const uint32_t HardwareMaxNumNamedBarriers = 16;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Hopper introduces a new cluster-wide barrier which handle with Cluster-wide
// arrive-wait behaviour. This is an extension to the Ampere arrive-wait
// barriers Note : Ampere arrive-wait Barriers have a larger max-arrive count
// (2^30) than Hopper arrive-wait Barriers (2^20).
struct ClusterBarrier {
  using ValueType = uint64_t;

 protected:
  // Can never be initialized - can only be aliased to smem
  ValueType barrier_;

 public:
  CUTLASS_DEVICE
  ClusterBarrier() = delete;

  CUTLASS_DEVICE
  void init(uint32_t arrive_count) const {
    ClusterBarrier::init(&this->barrier_, arrive_count);
  }

  CUTLASS_DEVICE
  bool test_wait(uint32_t phase, uint32_t pred = true) const {
    return ClusterBarrier::test_wait(&this->barrier_, phase, pred);
  }

  CUTLASS_DEVICE
  bool try_wait(uint32_t phase) const {
    return ClusterBarrier::try_wait(&this->barrier_, phase);
  }

  CUTLASS_DEVICE
  void wait(uint32_t phase) const {
    ClusterBarrier::wait(&this->barrier_, phase);
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  void arrive() const {
    ClusterBarrier::arrive(&this->barrier_);
  }

  // Remote SMEM arrive with a perdicate (usually done to pick the thread doing
  // the arrive)
  CUTLASS_DEVICE
  void arrive(uint32_t cta_id, uint32_t pred = true) const {
    ClusterBarrier::arrive(&this->barrier_, cta_id, pred);
  }

  //
  //  Static Versions
  //
  CUTLASS_DEVICE
  static void init(ValueType const* smem_ptr, uint32_t arrive_count) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared::cta.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_barrier_init(__LINE__, smem_addr, arrive_count);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  // Static version of wait - in case we don't want to burn a register
  CUTLASS_DEVICE
  static void wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_wait(__LINE__, smem_addr, phase);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));

#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static bool test_wait(ValueType const* smem_ptr, uint32_t phase, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_test_wait(__LINE__, smem_addr, phase, pred);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        ".reg .pred P2; \n\t"
        "setp.eq.u32 P2, %3, 1;\n\t"
        "@P2 mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase), "r"(pred));

    return static_cast<bool>(waitComplete);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
    return 0;
  }

  CUTLASS_DEVICE
  static bool try_wait(ValueType const* smem_ptr, uint32_t phase) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_cluster_barrier_try_wait(__LINE__, smem_addr, phase);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase));

    return static_cast<bool>(waitComplete);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
    return 0;
  }

  // Static Predicated version of the above - in case we know the address.
  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr, uint32_t cta_id, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    if (pred) {
      asm volatile(
          "{\n\t"
          ".reg .b32 remAddr32;\n\t"
          "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
          "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
          "}"
          :
          : "r"(smem_addr), "r"(cta_id));
    }

    cutlass::arch::synclog_emit_cluster_barrier_arrive_cluster(__LINE__, smem_addr, cta_id, pred);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  // Barrier arrive on local smem
  CUTLASS_DEVICE
  static void arrive(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_barrier_arrive(__LINE__, smem_addr);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  CUTLASS_DEVICE
  static void invalidate(ValueType const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.inval.shared::cta.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr));
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// SM90 also introduces a new type of cluster-barrier which supports sync.
// not just based on Arrive Count, but also transaction count (in bytes)
struct ClusterTransactionBarrier : public ClusterBarrier {
  CUTLASS_DEVICE
  ClusterTransactionBarrier() = delete;

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes);
  }

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred = 1u) const {
    ClusterTransactionBarrier::arrive_and_expect_tx(&this->barrier_, transaction_bytes, cta_id, pred);
  }

  // Performs an expected transaction bytes increment without doing an arrive
  // operation
  CUTLASS_DEVICE
  void expect_transaction(uint32_t transaction_bytes) const {
    ClusterTransactionBarrier::expect_transaction(&this->barrier_, transaction_bytes);
  }

  // Performs an expected transaction bytes decrement without doing an arrive
  // operation
  CUTLASS_DEVICE
  void complete_transaction(uint32_t transaction_bytes, uint32_t pred = 1) const {
    uint32_t cta_rank = cute::block_rank_in_cluster();
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, cta_rank, transaction_bytes, pred);
  }

  // Performs an expected transaction bytes decrement without doing an arrive
  // operation
  CUTLASS_DEVICE
  void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    ClusterTransactionBarrier::complete_transaction(&this->barrier_, dst_cta_id, transaction_bytes, pred);
  }

  //
  //  Static Versions
  //

  // Performs an arrive operation + expected transaction bytes increment
  CUTLASS_DEVICE
  static void arrive_and_expect_tx(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx(
        __LINE__, smem_addr, transaction_bytes);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  // Performs an arrive operation + expected transaction bytes increment for a
  // remote cta_id in a Cluster
  CUTLASS_DEVICE
  static void arrive_and_expect_tx(
      ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, "
        "[remAddr32], %3;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  // Performs an expected transaction bytes increment without doing an arrive
  // operation
  CUTLASS_DEVICE
  static void expect_transaction(ValueType const* smem_ptr, uint32_t transaction_bytes) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.expect_tx.shared::cta.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
    cutlass::arch::synclog_emit_cluster_transaction_barrier_expect_transaction(
        __LINE__, smem_addr, transaction_bytes);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  // Performs an expected transaction bytes decrement without doing an arrive
  // operation
  CUTLASS_DEVICE
  static void complete_transaction(
      ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
#if CUDA_BARRIER_ENABLED
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    smem_addr = cute::set_block_rank(smem_addr, dst_cta_id);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64  "
        " [%1], %0;"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr), "r"(pred));
    cutlass::arch::synclog_emit_cluster_transaction_barrier_complete_transaction(
        __LINE__, smem_addr, dst_cta_id, transaction_bytes, pred);
#elif defined(__CUDA_ARCH__)
    asm volatile("brkpt;\n" ::);
#endif
  }

  //
  // DEPRECATED APIs
  //
  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE void arrive_and_reset_bytes(
      uint32_t transaction_bytes) const {
    arrive_and_expect_tx(transaction_bytes);
  }

  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE void arrive_and_reset_bytes(
      uint32_t transaction_bytes, uint32_t cta_id) const {
    arrive_and_expect_tx(transaction_bytes, cta_id);
  }

  [[deprecated("Use expect_transaction instead")]] CUTLASS_DEVICE void reset_bytes(uint32_t transaction_bytes) const {
    expect_transaction(transaction_bytes);
  }

  [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE void commit(
      uint32_t transaction_bytes, uint32_t pred = 1) const {
    complete_transaction(transaction_bytes, pred);
  }

  [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE void commit(
      uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred) const {
    complete_transaction(dst_cta_id, transaction_bytes, pred);
  }

  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE static void arrive_and_reset_bytes(
      ValueType const* smem_ptr, uint32_t transaction_bytes) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes);
  }

  [[deprecated("Use arrive_and_expect_tx instead")]] CUTLASS_DEVICE static void arrive_and_reset_bytes(
      ValueType const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred) {
    arrive_and_expect_tx(smem_ptr, transaction_bytes, cta_id, pred);
  }

  [[deprecated("Use expect_transaction instead")]] CUTLASS_DEVICE static void reset_bytes(
      ValueType const* smem_ptr, uint32_t transaction_bytes) {
    expect_transaction(smem_ptr, transaction_bytes);
  }

  [[deprecated("Use complete_transaction instead")]] CUTLASS_DEVICE static void commit(
      ValueType const* smem_ptr, uint32_t dst_cta_id, uint32_t transaction_bytes, uint32_t pred = 1) {
    complete_transaction(smem_ptr, dst_cta_id, transaction_bytes, pred);
  }
};

// Helps with visibility of barrier init operations across warps / cta / cluster
// Available as a separate function so as to batch inits across barriers and
// fence once Note : It must be composed with an appropriate sync instruction
// with the right scope to ensure visibility eg. __syncthreads() or a
// cluster_arrive() + cluster_wait()
CUTLASS_DEVICE
void fence_barrier_init() {
#if CUDA_BARRIER_ENABLED
  cutlass::arch::synclog_emit_fence_barrier_init(__LINE__);
  asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}" ::);
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

// Issue a shared memory fence for async operations
CUTLASS_DEVICE
void fence_view_async_shared() {
#if CUDA_BARRIER_ENABLED
  cutlass::arch::synclog_emit_fence_view_async_shared(__LINE__);
  asm volatile(
      "{\n\t"
      "fence.proxy.async.shared::cta; \n"
      "}" ::);
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

// Arrive on completion of in-flight cp.async operations issued by the calling
// thread
CUTLASS_DEVICE
void cpasync_barrier_arrive(uint64_t const* smem_ptr) {
#if CUDA_BARRIER_ENABLED
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "{\n\t"
      "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
      "}"
      :
      : "r"(smem_addr));
  cutlass::arch::synclog_emit_cpasync_barrier_arrive(__LINE__, smem_addr);
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // end namespace arch
}  // end namespace cutlass

namespace cutlass {
namespace arch {

template <uint32_t RegCount>
CUTLASS_DEVICE void warpgroup_reg_alloc() {
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

template <uint32_t RegCount>
CUTLASS_DEVICE void warpgroup_reg_dealloc() {
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

}  // namespace arch
}  // namespace cutlass

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))))
#define CUTE_ARCH_CLUSTER_SM90_ENABLED
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDACC_VER_MAJOR__ >= 12))
#define CUTE_ARCH_ELECT_ONE_SM90_ENABLED
#endif

namespace cute {

#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
using TmaDescriptor = CUtensorMap;
using Im2ColTmaDescriptor = CUtensorMap;
#else
using TmaDescriptor = struct alignas(64) {
  char bytes[128];
};

using Im2ColTmaDescriptor = struct alignas(64) {
  char bytes[128];
};
#endif

CUTE_HOST_DEVICE
void prefetch_tma_descriptor(TmaDescriptor const* desc_ptr) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state
  // space: const or param)
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
#else
  CUTE_INVALID_CONTROL_PATH(
      "Trying to use TMA Descriptor Prefetch without "
      "CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

namespace TMA {
enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};
}

struct SM90_TMA_LOAD_2D {
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint, void* smem_ptr,
                                    int32_t const& crd0, int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0), "r"(crd1), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH {
    CUTE_HOST_DEVICE static void copy(void const* desc_ptr, int32_t const& crd0, int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile(
          "cp.async.bulk.prefetch.tensor.2d.L2.global"
          " [%0, {%1, %2}];"
          :
          : "l"(gmem_int_desc), "r"(crd0), "r"(crd1)
          : "memory");
#else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    }
  };
};

struct SM90_TMA_LOAD_MULTICAST_2D {
  CUTE_HOST_DEVICE static void copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
                                    uint64_t cache_hint, void* smem_ptr, int32_t const& crd0, int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
        "complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "h"(multicast_mask), "r"(crd0), "r"(crd1),
          "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_2D {
  CUTE_HOST_DEVICE static void copy(
      void const* desc_ptr, void const* smem_ptr, int32_t const& crd0, int32_t const& crd1) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, "
        "{%2, %3}], [%1];"
        :
        : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

}  // namespace cute
