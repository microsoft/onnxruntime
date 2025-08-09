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

#ifdef NVRTC_JIT_COMPILATION

using int8_t = signed char;
using uint8_t = unsigned char;
using int16_t = signed short;
using uint16_t = unsigned short;
using int32_t = signed int;
using uint32_t = unsigned int;
using int64_t = signed long long;
using uint64_t = unsigned long long;
using cuuint64_t = unsigned long long;

namespace std {
template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant;  // using injected-class-name

  __device__ constexpr operator value_type() const noexcept {
    return value;
  }

  __device__ constexpr value_type operator()() const noexcept {
    return value;
  }  // since c++14
};

using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;

template <class T, class U>
struct is_same : false_type {
};

template <class T>
struct is_same<T, T> : true_type {
};

template <class T, class U>
inline constexpr bool is_same_v = is_same<T, U>::value;
}  // namespace std

#endif
