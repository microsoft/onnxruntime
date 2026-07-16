// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

namespace onnxruntime {

// Generate a random RFC 4122 version-4 UUID formatted as
// "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx" (36 lowercase-hex characters, hyphen-separated).
//
// The 128 bits are drawn directly from a CSPRNG-backed std::random_device rather than seeding a
// PRNG, so the full entropy is preserved and the value is non-predictable. Seeding a std::mt19937
// from a single random_device value would cap the entropy at 32 bits and make ids collide across a
// large fleet (birthday-bound at ~77k values), so each field is sourced straight from the device.
inline std::string GenerateGuidV4() {
  std::random_device rd;
  uint64_t hi = (static_cast<uint64_t>(rd()) << 32) | rd();
  uint64_t lo = (static_cast<uint64_t>(rd()) << 32) | rd();
  // Set version (4) and variant (10xx) bits.
  hi = (hi & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  lo = (lo & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

  char buf[37];
  std::snprintf(buf, sizeof(buf),
                "%08x-%04x-%04x-%04x-%012llx",
                static_cast<uint32_t>(hi >> 32),
                static_cast<uint32_t>((hi >> 16) & 0xFFFF),
                static_cast<uint32_t>(hi & 0xFFFF),
                static_cast<uint32_t>(lo >> 48),
                static_cast<unsigned long long>(lo & 0xFFFFFFFFFFFFULL));
  return std::string(buf);
}

}  // namespace onnxruntime
