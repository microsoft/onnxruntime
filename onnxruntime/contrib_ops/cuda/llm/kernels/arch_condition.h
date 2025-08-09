/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace onnxruntime::llm::kernels {

namespace detail {

#ifdef __CUDA_ARCH__

// __CUDA_ARCH_SPECIFIC__ is only available starting from CUDA 12.9
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
#define HAS_CUDA_SPECIFIC_MACRO 1

#if __CUDA_ARCH__ >= 900
#if !defined(__CUDA_ARCH_SPECIFIC__) && !defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#error "Compiling for SM90 or newer architectures must use Arch specific or Arch Family specific target"
#endif
#endif

#else
#define HAS_CUDA_SPECIFIC_MACRO 0
#endif

// For CUDA < 12.9, we assume that sm90 or newer architectures are always built with arch specific.
#if defined(__CUDA_ARCH_SPECIFIC__) || (!HAS_CUDA_SPECIFIC_MACRO && __CUDA_ARCH__ >= 900)
static constexpr bool isArchSpecific = true;
#else
static constexpr bool isArchSpecific = false;
#endif

struct arch_info {
  static constexpr bool mIsDevice = true;
  static constexpr bool mArchSpecific = isArchSpecific;
  static constexpr int mMajor = __CUDA_ARCH__ / 100;
  static constexpr int mMinor = __CUDA_ARCH__ / 10 % 10;
  static constexpr int mArch = __CUDA_ARCH__ / 10;
};

#else

struct arch_info {
  static constexpr bool mIsDevice = false;
  static constexpr bool mArchSpecific = false;
  static constexpr int mMajor = 0;
  static constexpr int mMinor = 0;
  static constexpr int mArch = 0;
};

#endif

}  // namespace detail

namespace arch {

struct is_device : std::bool_constant<detail::arch_info::mIsDevice> {
};

struct is_arch_specific : std::bool_constant<detail::arch_info::mArchSpecific> {
};

template <int Arch>
struct is_match : std::bool_constant<is_device::value && detail::arch_info::mArch == Arch> {
};

template <int Major>
struct is_major : std::bool_constant<is_device::value && detail::arch_info::mMajor == Major> {
};

template <int Arch>
struct is_compatible : std::bool_constant<is_major<Arch>::value && detail::arch_info::mArch >= Arch> {
};

inline constexpr bool is_device_v = is_device::value;

inline constexpr bool is_arch_specific_v = is_arch_specific::value;

template <int Arch>
inline constexpr bool is_match_v = is_match<Arch>::value;

template <int Major>
inline constexpr bool is_major_v = is_major<Major>::value;

template <int Arch>
inline constexpr bool is_compatible_v = is_compatible<Arch>::value;

}  // namespace arch

}  // namespace onnxruntime::llm::kernels
