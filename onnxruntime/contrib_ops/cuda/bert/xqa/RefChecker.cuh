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
#include "cuda_hint.cuh"
#include "utils.cuh"
#include <cassert>
#include <cuda_fp16.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <type_traits>

struct RefChecker {
  half q[8][32][32];
  half k[8][4][64][32];
  float qk[4][32][64];
  float tileRowMax[4][32];
  half x[4][32][64];
  half v[8][4][32][64];
  float tileRowSum[4][32];
  float acc1PerStep[4][32][256];
  half out[32][256];

  void init() {
#define INIT_MEMBER(member) initMember(member, #member)
    INIT_MEMBER(q);
    INIT_MEMBER(k);
    INIT_MEMBER(qk);
    INIT_MEMBER(tileRowMax);
    INIT_MEMBER(x);
    INIT_MEMBER(v);
    INIT_MEMBER(tileRowSum);
    INIT_MEMBER(acc1PerStep);
    INIT_MEMBER(out);
#undef INIT_MEMBER
  }

 private:
  template <typename T>
  void initMember(T& dst, char const* varName);
};

template <typename T, size_t d0, size_t d1, size_t d2, size_t d3>
std::enable_if_t<std::is_same_v<std::decay_t<T>, float> || std::is_same_v<std::decay_t<T>, half>, std::string>
makeFileName(T (&dst)[d0][d1][d2][d3], char const* varName) {
  std::stringstream ss;
  ss << varName << '_' << d0 << 'x' << d1 << 'x' << d2 << 'x' << d3 << '_'
     << (std::is_same_v<std::decay_t<T>, float> ? "f32" : "f16") << ".bin";
  return ss.str();
}

template <typename T, size_t d0, size_t d1, size_t d2>
std::enable_if_t<std::is_same_v<std::decay_t<T>, float> || std::is_same_v<std::decay_t<T>, half>, std::string>
makeFileName(T (&dst)[d0][d1][d2], char const* varName) {
  std::stringstream ss;
  ss << varName << '_' << d0 << 'x' << d1 << 'x' << d2 << '_'
     << (std::is_same_v<std::decay_t<T>, float> ? "f32" : "f16") << ".bin";
  return ss.str();
}

template <typename T, size_t d0, size_t d1>
std::enable_if_t<std::is_same_v<std::decay_t<T>, float> || std::is_same_v<std::decay_t<T>, half>, std::string>
makeFileName(T (&dst)[d0][d1], char const* varName) {
  std::stringstream ss;
  ss << varName << '_' << d0 << 'x' << d1 << '_' << (std::is_same_v<std::decay_t<T>, float> ? "f32" : "f16")
     << ".bin";
  return ss.str();
}

template <typename T>
void RefChecker::initMember(T& dst, char const* varName) {
  std::string const filename = makeFileName(dst, varName);
  printf("loading %s\n", filename.c_str());
  namespace fs = std::filesystem;
  assert(fs::exists(filename));
  assert(fs::file_size(filename) == sizeof(dst));
  std::ifstream fin(filename, std::ios::binary);
  fin.read(reinterpret_cast<char*>(&dst), sizeof(dst));
  assert(fin);
}
