// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
struct MurmurHash3 {
  // generate 32-bit hash from input and write to 'out'
  static void x86_32(const void* key, int len, uint32_t seed, void* out);

  // generate 128-bit hash from input and write to 'out'.
  static void x86_128(const void* key, int len, uint32_t seed, void* out);
};
}  // namespace onnxruntime
