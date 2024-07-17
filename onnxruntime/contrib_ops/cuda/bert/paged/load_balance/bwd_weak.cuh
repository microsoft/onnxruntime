// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/paged/cuda_common.h"
#include "contrib_ops/cuda/bert/paged/cuda_common.cuh"
#include "contrib_ops/cuda/bert/paged/mutex.cuh"

namespace onnxruntime::contrib::paged {

// TODO: move to atomics.cuh
namespace memory_order {
struct Acquire {};
struct Release {};
struct Relaxed {};
};  // namespace memory_order

namespace memory_scope {
struct Sys {};
struct Gpu {};
// struct Cluster{};
// struct Cta{};
}  // namespace memory_scope

template <typename Sem, typename Scope = memory_scope::Gpu>
__forceinline__ __device__ int32_t
atomic_load_global_i32(const int32_t* address) {
  int32_t ret;
  if constexpr (std::is_same_v<Sem, memory_order::Relaxed> && std::is_same_v<Scope, memory_scope::Gpu>) {
#if !defined(__HIPCC__)
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(address));
#else
    // ret = __atomic_load_n(address, __ATOMIC_RELAXED);
    ret = volatile_load(address);
#endif
  } else {
    static_assert(onnxruntime::contrib::paged::always_false<Sem, Scope>, "not implemented");
  }
  return ret;
}

template <typename Sem, typename Scope = memory_scope::Gpu>
__forceinline__ __device__ uint32_t
atomic_load_global_u32(uint32_t* address) {
  uint32_t ret;
  if constexpr (std::is_same_v<Sem, memory_order::Relaxed> && std::is_same_v<Scope, memory_scope::Gpu>) {
#if !defined(__HIPCC__)
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(ret) : "l"(address));
#else
    ret = __atomic_load_n(address, __ATOMIC_RELAXED);
#endif
  } else {
    static_assert(onnxruntime::contrib::paged::always_false<Sem, Scope>, "not implemented");
  }
  return ret;
}

template <typename Sem, typename Scope = memory_scope::Gpu>
__forceinline__ __device__ void
atomic_store_global_u32(uint32_t* address, uint32_t val) {
  if constexpr (std::is_same_v<Sem, memory_order::Relaxed> && std::is_same_v<Scope, memory_scope::Gpu>) {
#if !defined(__HIPCC__)
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(address), "r"(val));
#else
    // __atomic_store_n(address, val, __ATOMIC_RELAXED);
    volatile_store(address, val);
#endif
  } else if constexpr (std::is_same_v<Sem, memory_order::Release> && std::is_same_v<Scope, memory_scope::Gpu>) {
#if !defined(__HIPCC__)
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(address), "r"(val));
#else
    // __atomic_store_n(address, val, __ATOMIC_RELEASE);
    __threadfence();
    volatile_store(address, val);
#endif
  } else {
    static_assert(onnxruntime::contrib::paged::always_false<Sem, Scope>, "not implemented");
  }
}

template <typename Sem, typename Scope = memory_scope::Gpu>
__forceinline__ __device__ int32_t
atomic_add_global_i32(int32_t* address, int32_t val) {
  int32_t old;
  if constexpr (std::is_same_v<Sem, memory_order::Relaxed> && std::is_same_v<Scope, memory_scope::Gpu>) {
#if !defined(__HIPCC__)
    asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(address), "r"(val));
#else
    old = __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#endif
  } else {
    static_assert(onnxruntime::contrib::paged::always_false<Sem, Scope>, "not implemented");
  }
  return old;
}

template <typename Sem, typename Scope = memory_scope::Gpu>
__forceinline__ __device__ int32_t
atomic_sub_global_i32(int32_t* address, int32_t val) {
  return atomic_add_global_i32<Sem, Scope>(address, -val);
}

template <typename Sem, typename Scope = memory_scope::Gpu>
__forceinline__ __device__ uint32_t
atomic_add_global_u32(uint32_t* address, uint32_t val) {
  uint32_t old;
  if constexpr (std::is_same_v<Sem, memory_order::Relaxed> && std::is_same_v<Scope, memory_scope::Gpu>) {
#if !defined(__HIPCC__)
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(address), "r"(val));
#else
    old = __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#endif
  } else {
    static_assert(onnxruntime::contrib::paged::always_false<Sem, Scope>, "not implemented");
  }
  return old;
}

template <typename Sem, typename Scope = memory_scope::Gpu>
__forceinline__ __device__ void
memory_barrier() {
  if constexpr (std::is_same_v<Scope, memory_scope::Gpu>) {
#if !defined(__HIPCC__)
    asm volatile("fence.acq_rel.gpu;" ::);
#else
    // __atomic_thread_fence(__ATOMIC_ACQ_REL);
    __threadfence();
#endif
  } else {
    static_assert(onnxruntime::contrib::paged::always_false<Sem, Scope>, "not implemented");
  }
}

template <int N, typename T>
class BrokerWorkDistributor {
  static_assert(next_power_of_two(N) == N, "queue size must be power of two");

public:
  __device__ void init() {
    // Just manually initialize (memset) the BrokerWorkDistributor to 0 will be OK!
    const int lid = threadIdx.x + blockIdx.x * blockDim.x;
    if (lid == 0) {
      size_ = 0;
      head_ = 0;
      tail_ = 0;
    }

    for (int v = lid; v < N; v += blockDim.x * gridDim.x) {
      // ring_buffer_[v] = T{};
      tickets_[v] = 0;
    }
  }

  __forceinline__ __device__ bool
  enqueue(const T& data) {
    if (ensure_enqueue()) {
      put_data(data);
      return true;
    }
    return false;
  }

  __forceinline__ __device__ bool
  dequeue(T& data) {
    if (ensure_dequeue()) {
      read_data(data);
      return true;
    }
    return false;
  }

private:
  using ticket_t = uint32_t;
  using head_tail_t = uint32_t;

  T ring_buffer_[N];
  ticket_t tickets_[N];
  head_tail_t head_;
  head_tail_t tail_;
  int32_t size_;

  __forceinline__ __device__ void
  wait_for_ticket(const uint32_t i, const ticket_t number) {
    while (atomic_load_global_u32<memory_order::Relaxed>(&tickets_[i]) != number) {
      backoff();
    }
  }

  __forceinline__ __device__ bool
  ensure_enqueue() {
    int32_t num = atomic_load_global_i32<memory_order::Relaxed>(&size_);
    while (true) {
      if (num >= N) {
        return false;
      }
      if (atomic_add_global_i32<memory_order::Relaxed>(&size_, 1) < N) {
        break;
      }
      num = atomic_sub_global_i32<memory_order::Relaxed>(&size_, 1) - 1;
    }
    return true;
  }

  __forceinline__ __device__ bool
  ensure_dequeue() {
    int32_t num = atomic_load_global_i32<memory_order::Relaxed>(&size_);
    int ns = 1;
    while (true) {
      if (num <= 0) {
        return false;
      }
      if (atomic_sub_global_i32<memory_order::Relaxed>(&size_, 1) > 0) {
        break;
      }
      backoff(ns);
      ns = min(ns * 2, 256);
      num = atomic_add_global_i32<memory_order::Relaxed>(&size_, 1) + 1;
    }
    return true;
  }

  __forceinline__ __device__ void
  read_data(T& data) {
    uint32_t pos = atomic_add_global_u32<memory_order::Relaxed>(&head_, 1);
    uint32_t p = pos % N;
    wait_for_ticket(p, 2 * (pos / N) + 1);
    memory_barrier<memory_order::Acquire>();
    data = ring_buffer_[p];
    atomic_store_global_u32<memory_order::Release>(&tickets_[p], 2 * ((pos + N) / N));  // store_ticket
  }

  __forceinline__ __device__ void
  put_data(const T& data) {
    uint32_t pos = atomic_add_global_u32<memory_order::Relaxed>(&tail_, 1);
    uint32_t p = pos % N;
    uint32_t b = 2 * (pos / N);
    wait_for_ticket(p, b);
    ring_buffer_[p] = data;
    atomic_store_global_u32<memory_order::Release>(&tickets_[p], b + 1);  // store_ticket
  }
};

}  // namespace onnxruntime::contrib::paged
