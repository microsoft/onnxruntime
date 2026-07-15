/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef GENERATE_CUBIN
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#endif
#include "mha_stdheaders.cuh"

#ifdef __CUDA_ARCH__
#define XQA_UNROLL _Pragma("unroll")
#else
#define XQA_UNROLL
#endif

template <typename T>
HOST_DEVICE_FUNC constexpr inline void unused(T&& x) {
  static_cast<void>(x);
}

#ifndef GENERATE_CUBIN
inline void checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorName(err));
    throw std::runtime_error(cudaGetErrorName(err));
  }
}

inline void checkCu(CUresult err) {
  if (err != CUDA_SUCCESS) {
    char const* str = nullptr;
    if (cuGetErrorName(err, &str) != CUDA_SUCCESS) {
      str = "A cuda driver API error happened, but we failed to query the error name\n";
    }
    printf("%s\n", str);
    throw std::runtime_error(str);
  }
}
#endif

HOST_DEVICE_FUNC constexpr inline uint32_t greatestPowerOf2Divisor(uint32_t x) {
  return x & ~(x - 1);
}

template <typename T>
HOST_DEVICE_FUNC constexpr uint32_t maxArrayAlign(uint32_t size) {
  return sizeof(T) * greatestPowerOf2Divisor(size);
}

HOST_DEVICE_FUNC constexpr inline uint32_t exactDiv(uint32_t a, uint32_t b) {
  assert(a % b == 0);
  return a / b;
}

template <typename T>
HOST_DEVICE_FUNC constexpr inline T divUp(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
HOST_DEVICE_FUNC constexpr inline T roundUp(T a, T b) {
  return divUp(a, b) * b;
}

// upperBound is exclusive, i.e. range is [0, upperBound)
template <uint32_t upperBound>
struct BoundedVal {
  template <uint32_t divisor>
  HOST_DEVICE_FUNC inline BoundedVal<upperBound / divisor> divBy() const {
    assert(value < upperBound);
    return {upperBound <= divisor ? 0 : value / divisor};
  }

  template <uint32_t divisor>
  HOST_DEVICE_FUNC inline BoundedVal<mha::min(divisor, upperBound)> mod() const {
    assert(value < upperBound);
    return {upperBound <= divisor ? value : value % divisor};
  }

  HOST_DEVICE_FUNC inline bool operator<=(uint32_t rhs) const {
    assert(value < upperBound);
    return upperBound <= rhs || value <= rhs;
  }

  HOST_DEVICE_FUNC inline uint32_t get() const {
    assert(value < upperBound);
    return upperBound == 1 ? 0 : value;
  }

  uint32_t value;
};

template <typename T, uint32_t size_>
struct alignas(mha::max<uint32_t>(alignof(T), mha::min<uint32_t>(maxArrayAlign<T>(size_), 16))) Vec {
  using Elem = T;
  static constexpr uint32_t size = size_;
  Elem data[size];

  HOST_DEVICE_FUNC inline void fill(T const& val) {
    XQA_UNROLL
    for (uint32_t i = 0; i < size; i++) {
      data[i] = val;
    }
  }

  static HOST_DEVICE_FUNC inline Vec<T, size> filled(T const& val) {
    Vec<T, size> ret;
    ret.fill(val);
    return ret;
  }

  HOST_DEVICE_FUNC inline Elem const& operator[](uint32_t i) const {
    assert(i < size);
    return data[BoundedVal<size>{i}.get()];
  }

  HOST_DEVICE_FUNC inline Elem& operator[](uint32_t i) {
    assert(i < size);
    return data[BoundedVal<size>{i}.get()];
  }
};

template <uint32_t nbBuffers_>
struct CircIdx {
 public:
  static constexpr uint32_t nbBuffers = nbBuffers_;
  static_assert(nbBuffers >= 1);

  __device__ inline CircIdx(uint32_t init)
      : mIndex{init % nbBuffers} {
  }

  __device__ inline operator uint32_t() const {
    return mIndex;
  }

  __device__ inline CircIdx operator+(uint32_t i) const {
    return CircIdx{(mIndex + i) % nbBuffers};
  }

  __device__ inline CircIdx operator-(uint32_t i) const {
    return CircIdx{(mIndex + (nbBuffers - 1) * i) % nbBuffers};
  }

  __device__ inline CircIdx next() const {
    return *this + 1u;
  }

  __device__ inline CircIdx& operator++() {
    mIndex = next();
    return *this;
  }

  __device__ inline CircIdx operator++(int) {
    CircIdx old = *this;
    operator++();
    return old;
  }

  __device__ inline CircIdx prev() const {
    return *this - 1u;
  }

  __device__ inline CircIdx& operator--() {
    mIndex = prev();
    return *this;
  }

  __device__ inline CircIdx operator--(int) {
    CircIdx old = *this;
    operator--();
    return old;
  }

 private:
  uint32_t mIndex;
};

// base is usually in constant memory, so usually only require 1 register to store the offset.
template <typename T>
struct TinyPtr {
  T* base;          // typically in constant memory or uniform registers
  uint32_t offset;  // may be non-uniform

  template <typename D>
  __device__ __host__ inline TinyPtr<D> cast() const {
    D* const p = reinterpret_cast<D*>(base);
    assert(reinterpret_cast<uintptr_t>(p) % alignof(D) == 0);
    if constexpr (mha::is_void_v<T>) {
      assert(offset == 0);
      return TinyPtr<D>{p, 0};
    } else if constexpr (sizeof(T) < sizeof(D)) {
      return TinyPtr<D>{p, exactDiv(offset, exactDiv(sizeof(D), sizeof(T)))};
    } else {
      return TinyPtr<D>{p, offset * exactDiv(sizeof(T), sizeof(D))};
    }
  }

  __device__ __host__ inline T& operator*() const {
    return base[offset];
  }

  __device__ __host__ inline TinyPtr<T> operator+(uint32_t i) const {
    return TinyPtr<T>{base, offset + i};
  }

  __device__ __host__ inline T& operator[](uint32_t i) const {
    return *(*this + i);
  }

  __device__ __host__ inline operator T*() const {
    return base + offset;
  }
};

template <typename OffsetInt = uint32_t>
class Segmenter {
 public:
  HOST_DEVICE_FUNC Segmenter(uint32_t offset = 0)
      : mNextOffset{offset} {
  }

  // offset is in bytes
  template <typename T>
  HOST_DEVICE_FUNC OffsetInt newSeg(uint32_t count = 1, uint32_t alignment = alignof(T)) {
    mMaxAlignment = mha::max<uint32_t>(mMaxAlignment, alignment);
    OffsetInt const offset = roundUp<OffsetInt>(mNextOffset, alignment);
    mNextOffset = offset + sizeof(T) * count;
    return offset;
  }

  HOST_DEVICE_FUNC OffsetInt getEndOffset() const {
    return mNextOffset;
  }

  HOST_DEVICE_FUNC uint32_t getMaxAlignment() const {
    return mMaxAlignment;
  }

 private:
  OffsetInt mNextOffset;
  uint32_t mMaxAlignment = 1;
};

template <typename T, bool addConst>
using AddConst = mha::conditional_t<addConst, T const, T>;

template <bool isConst, typename OffsetInt = uint32_t>
class MemSegmenter {
 public:
  HOST_DEVICE_FUNC MemSegmenter(AddConst<void, isConst>* base, uint32_t offset = 0)
      : mBase{static_cast<AddConst<mha::byte, isConst>*>(base)}, mSegmenter{offset} {
  }

  // to use TinyPtr, alignment must be sizeof(T)
  template <typename T>
  HOST_DEVICE_FUNC TinyPtr<AddConst<T, isConst>> newSeg(uint32_t count = 1, uint32_t alignment = sizeof(T)) {
    assert(reinterpret_cast<uintptr_t>(mBase) % alignof(T) == 0);
    OffsetInt const offset = mSegmenter.template newSeg<T>(count, alignment);
    return TinyPtr<AddConst<mha::byte, isConst>>{mBase, offset}.template cast<AddConst<T, isConst>>();
  }

  HOST_DEVICE_FUNC OffsetInt getEndOffset() const {
    return mSegmenter.getEndOffset();
  }

  HOST_DEVICE_FUNC uint32_t getMaxAlignment() const {
    return mSegmenter.getMaxAlignment();
  }

 private:
  AddConst<mha::byte, isConst>* mBase;
  Segmenter<OffsetInt> mSegmenter;
};

// dims in little endian
template <uint32_t nbDims_>
struct DimsLE {
  static constexpr uint32_t nbDims = nbDims_;

  __device__ __host__ inline uint32_t& operator[](uint32_t i) {
    return d[i];
  }

  __device__ __host__ inline uint32_t const& operator[](uint32_t i) const {
    return d[i];
  }

  uint32_t d[nbDims];
};

// check if val is in range [lb, ub)
template <typename T>
constexpr bool inRange(T val, T lb, T ub) {
  return val >= lb && val < ub;
}

// val is an optimized / pre-computed value, ref is the original value
template <typename T>
HOST_DEVICE_FUNC constexpr inline T checkedVal(T val, T ref) {
  assert(val == ref);
  return val;
}
