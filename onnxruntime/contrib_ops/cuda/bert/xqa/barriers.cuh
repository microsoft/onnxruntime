/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cuda_hint.cuh"
#include "defines.h"
#if !USE_CUSTOM_BARRIER
#include <cuda/std/barrier>
using CtaBarrier = cuda::barrier<cuda::thread_scope_block>;
#else

#ifndef __CUDACC__
#include <cuda_runtime_api.h>
#endif

#if __CUDACC_VER_MAJOR__ < 12
#define STR_REL_CTA ""
#define STR_ACQ_CTA ""
#else
#define STR_REL_CTA ".release.cta"
#define STR_ACQ_CTA ".acquire.cta"
#endif

enum class Scope : uint32_t {
  CTA = 0,
  CGA = 1,
};

enum class ArriveOrder : uint32_t {
  RELEASE = 0,
  RELAXED = 1,
};

enum class ArrivalToken : uint64_t {
};

template <Scope defaultScope_ = Scope::CTA>
class MBarrier  // rename this to MBarrier
{
 public:
  using ArrivalToken = ::ArrivalToken;
  static constexpr Scope defaultScope = defaultScope_;
  using arrival_token = ArrivalToken;

  __device__ inline MBarrier(uint32_t count) {
    assert(count > 0);
    asm volatile("mbarrier.init.b64 [%0], %1;\n" ::"l"(addr()), "r"(count) : "memory");
  }

  __device__ ~MBarrier() {
    asm volatile("mbarrier.inval.b64 [%0];\n" ::"l"(addr()) : "memory");
  }

  template <Scope scope = defaultScope, ArriveOrder order = ArriveOrder::RELEASE>
  __device__ inline mha::conditional_t<scope == Scope::CTA, ArrivalToken, void> arrive(uint32_t update = 1) {
    ArrivalToken token{};
#if __CUDA_ARCH__ >= 900
    if constexpr (scope == Scope::CTA) {
      switch (order) {
        case ArriveOrder::RELEASE:
          asm volatile("mbarrier.arrive.release.cta.b64 %0, [%1], %2;\n"
                       : "=l"(token)
                       : "l"(addr()), "r"(update)
                       : "memory");
          break;
        case ArriveOrder::RELAXED:
          asm volatile("mbarrier.arrive.relaxed.cta.b64 %0, [%1], %2;\n"
                       : "=l"(token)
                       : "l"(addr()), "r"(update)
                       : "memory");
          break;
      }
      return token;
    } else {
      static_assert(scope == Scope::CGA);
      switch (order) {
        case ArriveOrder::RELEASE:
          asm volatile("mbarrier.arrive.release.cluster.b64 _, [%0], %1;\n" ::"l"(addr()), "r"(update)
                       : "memory");
          break;
        case ArriveOrder::RELAXED:
          asm volatile("mbarrier.arrive.relaxed.cluster.b64 _, [%0], %1;\n" ::"l"(addr()), "r"(update)
                       : "memory");
          break;
      }
      return;
    }
#else
    static_assert(scope == Scope::CTA && order == ArriveOrder::RELEASE);
    if (update > 1) {
      asm volatile("mbarrier.arrive.noComplete" STR_REL_CTA ".b64 %0, [%1], %2;\n"
                   : "=l"(token)
                   : "l"(addr()), "r"(update - 1U)
                   : "memory");
      [[maybe_unused]] ArrivalToken refToken;
      asm volatile("mbarrier.arrive" STR_REL_CTA ".b64 %0, [%1];\n" : "=l"(refToken) : "l"(addr()) : "memory");
      assert(token == refToken);
      return token;
    } else {
      asm volatile("mbarrier.arrive" STR_REL_CTA ".b64 %0, [%1];\n" : "=l"(token) : "l"(addr()) : "memory");
      return token;
    }
#endif
  }

  __device__ inline bool isLocal() const {
    uint32_t addrCtaRank{};
    asm("getctarank.u64 %0, %1;\n" : "=r"(addrCtaRank) : "l"(addr()));
    uint32_t ctaRank{};
    asm("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(ctaRank));
    return addrCtaRank == ctaRank;
  }

  __device__ inline void remoteArrive(uint32_t update = 1) {
#if __CUDA_ARCH__ >= 900
    assert(!isLocal());
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0], %1;\n"
                 :
                 : "l"(__cvta_generic_to_shared(&mBar)), "r"(update)
                 : "memory");
#else
    asm volatile("trap;\n");
#endif
  }

  template <Scope scope = defaultScope, ArriveOrder order = ArriveOrder::RELEASE>
  __device__ inline mha::conditional_t<scope == Scope::CTA, ArrivalToken, void> arrive_tx_relaxed(uint32_t txCount) {
#if __CUDA_ARCH__ >= 900
    if constexpr (scope == Scope::CTA) {
      ArrivalToken token{};
      asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.b64 %0, [%1], %2;\n"
                   : "=l"(token)
                   : "l"(addr()), "r"(txCount)
                   : "memory");
      return token;
    } else {
      asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster.b64 _, [%0], %1;\n" ::"l"(addr()), "r"(txCount)
                   : "memory");
      return;
    }
#else
    asm volatile("trap;\n");
#endif
  }

  template <Scope scope = defaultScope, ArriveOrder order = ArriveOrder::RELEASE>
  __device__ inline mha::conditional_t<scope == Scope::CTA, ArrivalToken, void> arrive_tx(
      uint32_t txCount, uint32_t arriveCount = 1) {
#if __CUDA_ARCH__ >= 900
    if (arriveCount == 1) {
      if constexpr (scope == Scope::CTA) {
        ArrivalToken token{};
        switch (order) {
          case ArriveOrder::RELEASE:
            asm volatile("mbarrier.arrive.expect_tx.release.cta.b64 %0, [%1], %2;\n"
                         : "=l"(token)
                         : "l"(addr()), "r"(txCount)
                         : "memory");
            break;
          case ArriveOrder::RELAXED:
            asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.b64 %0, [%1], %2;\n"
                         : "=l"(token)
                         : "l"(addr()), "r"(txCount)
                         : "memory");
            break;
        }
        return token;
      } else {
        static_assert(scope == Scope::CGA);
        switch (order) {
          case ArriveOrder::RELEASE:
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cluster.b64 _, [%0], %1;\n" ::"l"(addr()), "r"(txCount)
                : "memory");
            break;
          case ArriveOrder::RELAXED:
            asm volatile(
                "mbarrier.arrive.expect_tx.relaxed.cluster.b64 _, [%0], %1;\n" ::"l"(addr()), "r"(txCount)
                : "memory");
            break;
        }
        return;
      }
    } else {
      if constexpr (scope == Scope::CTA) {
        asm volatile("mbarrier.expect_tx.relaxed.cta.b64 [%0], %1;\n" ::"l"(addr()), "r"(txCount) : "memory");
      } else {
        asm volatile("mbarrier.expect_tx.relaxed.cluster.b64 [%0], %1;\n" ::"l"(addr()), "r"(txCount)
                     : "memory");
      }
      return arrive<scope, order>(arriveCount);
    }
#else
    asm volatile("trap;\n");
#endif
  }

  template <Scope scope = defaultScope>
  __device__ inline bool test_wait(ArrivalToken&& token) {
    uint32_t ready{};
    if constexpr (scope == Scope::CGA) {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.test_wait.acquire.cluster.b64 ready, [%1], %2;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "l"(token)
          : "memory");
    } else {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.test_wait" STR_ACQ_CTA
          ".b64 ready, [%1], %2;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "l"(token)
          : "memory");
    }
    return ready != 0;
  }

  template <Scope scope = defaultScope>
  __device__ inline bool test_wait_parity(bool parity) {
    uint32_t ready{};
    if constexpr (scope == Scope::CGA) {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.test_wait.parity.acquire.cluster.b64 ready, [%1], %2;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "r"(uint32_t{parity})
          : "memory");
    } else {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.test_wait.parity" STR_ACQ_CTA
          ".b64 ready, [%1], %2;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "r"(uint32_t{parity})
          : "memory");
    }
    return ready != 0;
  }
#if __CUDA_ARCH__ >= 900
  template <Scope scope = defaultScope>
  __device__ inline bool try_wait(ArrivalToken&& token) {
    uint32_t ready{};
    if constexpr (scope == Scope::CGA) {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.try_wait.acquire.cluster.b64 ready, [%1], %2, %3;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "l"(token), "n"(kSUSPEND_TIME_HINT)
          : "memory");
    } else {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.try_wait.acquire.cta.b64 ready, [%1], %2, %3;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "l"(token), "n"(kSUSPEND_TIME_HINT)
          : "memory");
    }
    return ready != 0;
  }

  template <Scope scope = defaultScope>
  __device__ inline bool try_wait_parity(bool parity) {
    uint32_t ready{};
    if constexpr (scope == Scope::CGA) {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.try_wait.parity.acquire.cluster.b64 ready, [%1], %2, %3;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "r"(uint32_t{parity}), "n"(kSUSPEND_TIME_HINT)
          : "memory");
    } else {
      asm volatile(
          "{\n"
          ".reg .pred ready;\n"
          "mbarrier.try_wait.parity.acquire.cta.b64 ready, [%1], %2, %3;\n"
          "selp.b32 %0, 1, 0, ready;\n"
          "}\n"
          : "=r"(ready)
          : "l"(addr()), "r"(uint32_t{parity}), "n"(kSUSPEND_TIME_HINT)
          : "memory");
    }
    return ready != 0;
  }
#endif
  template <Scope scope = defaultScope>
  __device__ inline void wait(ArrivalToken&& token) {
#if __CUDA_ARCH__ >= 900
    poll<true>([&]() { return try_wait<scope>(ArrivalToken{token}); });
#else
    poll<false>([&]() { return test_wait<scope>(ArrivalToken{token}); });
#endif
  }

  // starting from `parity = false`.
  template <Scope scope = defaultScope>
  __device__ inline void wait_parity(bool parity) {
#if __CUDA_ARCH__ >= 900
    poll<true>([&]() { return try_wait_parity<scope>(parity); });
#else
    poll<false>([&]() { return test_wait_parity<scope>(parity); });
#endif
  }

  template <Scope scope = defaultScope, ArriveOrder order = ArriveOrder::RELEASE>
  __device__ inline mha::enable_if_t<scope == Scope::CTA, void> arrive_and_wait(uint32_t update = 1) {
    wait<scope>(arrive<scope, order>(update));
  }

 private:
  __device__ inline uint64_t addr() const {
    return reinterpret_cast<uint64_t>(&mBar);
  }

  template <bool funcSupportsBlocking, typename F>
  __device__ inline static void poll(F&& func) {
    if constexpr (funcSupportsBlocking) {
      while (!func()) {
      }
    } else {
      float sleepDuration = 0.125F;
      while (!func()) {
        __nanosleep(uint32_t(sleepDuration));
        sleepDuration = sleepDuration * 1.25F + 0.F;
      }
    }
  }

 public:
  static constexpr uint32_t kSUSPEND_TIME_HINT = 0xFFFFFFFFU;

 private:
  uint64_t mBar;
};

template <Scope defaultScope>
__device__ inline void init(MBarrier<defaultScope>* bar, uint32_t count) {
  new (bar) MBarrier<defaultScope>{count};
}

using CtaBarrier = MBarrier<Scope::CTA>;
using CgaBarrier = MBarrier<Scope::CGA>;

template <uint32_t nbBars>
__device__ inline constexpr bool toParity(uint32_t i) {
  return i % (nbBars * 2) / nbBars;
}

class NamedBarrier {
 public:
  __device__ inline NamedBarrier(uint32_t idxBar, uint32_t arriveCount)
      : mName{idxBar}, mArriveCount{arriveCount} {
    assert(idxBar < 16 && arriveCount % 32 == 0);
  }

  __device__ inline void arrive() const {
    asm volatile("barrier.cta.arrive %0, %1;\n" ::"r"(mName), "r"(mArriveCount) : "memory");
  }

  __device__ inline void arrive_and_wait() const {
    asm volatile("barrier.cta.sync %0, %1;\n" ::"r"(mName), "r"(mArriveCount) : "memory");
  }

 private:
  uint32_t const mName;
  uint32_t const mArriveCount;
};

__device__ inline void namedBarSync(uint32_t idxBar, uint32_t arriveCount) {
  NamedBarrier bar{idxBar, arriveCount};
  bar.arrive_and_wait();
}
#endif
