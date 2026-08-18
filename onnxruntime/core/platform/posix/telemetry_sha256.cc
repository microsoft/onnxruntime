// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/telemetry_sha256.h"

#include <algorithm>
#include <cstring>

namespace onnxruntime::telemetry_internal {

namespace {

constexpr uint32_t kInitialState[8] = {
    0x6a09e667u,
    0xbb67ae85u,
    0x3c6ef372u,
    0xa54ff53au,
    0x510e527fu,
    0x9b05688cu,
    0x1f83d9abu,
    0x5be0cd19u,
};

constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98u,
    0x71374491u,
    0xb5c0fbcfu,
    0xe9b5dba5u,
    0x3956c25bu,
    0x59f111f1u,
    0x923f82a4u,
    0xab1c5ed5u,
    0xd807aa98u,
    0x12835b01u,
    0x243185beu,
    0x550c7dc3u,
    0x72be5d74u,
    0x80deb1feu,
    0x9bdc06a7u,
    0xc19bf174u,
    0xe49b69c1u,
    0xefbe4786u,
    0x0fc19dc6u,
    0x240ca1ccu,
    0x2de92c6fu,
    0x4a7484aau,
    0x5cb0a9dcu,
    0x76f988dau,
    0x983e5152u,
    0xa831c66du,
    0xb00327c8u,
    0xbf597fc7u,
    0xc6e00bf3u,
    0xd5a79147u,
    0x06ca6351u,
    0x14292967u,
    0x27b70a85u,
    0x2e1b2138u,
    0x4d2c6dfcu,
    0x53380d13u,
    0x650a7354u,
    0x766a0abbu,
    0x81c2c92eu,
    0x92722c85u,
    0xa2bfe8a1u,
    0xa81a664bu,
    0xc24b8b70u,
    0xc76c51a3u,
    0xd192e819u,
    0xd6990624u,
    0xf40e3585u,
    0x106aa070u,
    0x19a4c116u,
    0x1e376c08u,
    0x2748774cu,
    0x34b0bcb5u,
    0x391c0cb3u,
    0x4ed8aa4au,
    0x5b9cca4fu,
    0x682e6ff3u,
    0x748f82eeu,
    0x78a5636fu,
    0x84c87814u,
    0x8cc70208u,
    0x90befffau,
    0xa4506cebu,
    0xbef9a3f7u,
    0xc67178f2u,
};

uint32_t RotateRight(uint32_t value, int bits) {
  return (value >> bits) | (value << (32 - bits));
}

uint32_t Choose(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}

uint32_t Majority(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

uint32_t BigSigma0(uint32_t value) {
  return RotateRight(value, 2) ^ RotateRight(value, 13) ^ RotateRight(value, 22);
}

uint32_t BigSigma1(uint32_t value) {
  return RotateRight(value, 6) ^ RotateRight(value, 11) ^ RotateRight(value, 25);
}

uint32_t SmallSigma0(uint32_t value) {
  return RotateRight(value, 7) ^ RotateRight(value, 18) ^ (value >> 3);
}

uint32_t SmallSigma1(uint32_t value) {
  return RotateRight(value, 17) ^ RotateRight(value, 19) ^ (value >> 10);
}

}  // namespace

Sha256::Sha256() {
  std::memcpy(state_, kInitialState, sizeof(state_));
}

void Sha256::Transform(const uint8_t block[64]) {
  uint32_t words[64];
  for (int i = 0; i < 16; ++i) {
    words[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    words[i] = SmallSigma1(words[i - 2]) + words[i - 7] + SmallSigma0(words[i - 15]) + words[i - 16];
  }

  uint32_t a = state_[0];
  uint32_t b = state_[1];
  uint32_t c = state_[2];
  uint32_t d = state_[3];
  uint32_t e = state_[4];
  uint32_t f = state_[5];
  uint32_t g = state_[6];
  uint32_t h = state_[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t first = h + BigSigma1(e) + Choose(e, f, g) + kRoundConstants[i] + words[i];
    const uint32_t second = BigSigma0(a) + Majority(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + first;
    d = c;
    c = b;
    b = a;
    a = first + second;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256::Update(const void* data, size_t length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  bit_count_ += static_cast<uint64_t>(length) * 8;
  while (length > 0) {
    const size_t count = std::min<size_t>(64 - buffer_length_, length);
    std::memcpy(buffer_ + buffer_length_, bytes, count);
    buffer_length_ += count;
    bytes += count;
    length -= count;
    if (buffer_length_ == 64) {
      Transform(buffer_);
      buffer_length_ = 0;
    }
  }
}

void Sha256::Final(uint8_t output[kDigestSize]) {
  buffer_[buffer_length_++] = 0x80;
  if (buffer_length_ > 56) {
    std::memset(buffer_ + buffer_length_, 0, 64 - buffer_length_);
    Transform(buffer_);
    buffer_length_ = 0;
  }
  std::memset(buffer_ + buffer_length_, 0, 56 - buffer_length_);
  uint64_t bit_count = bit_count_;
  for (int i = 7; i >= 0; --i) {
    buffer_[56 + i] = static_cast<uint8_t>(bit_count & 0xff);
    bit_count >>= 8;
  }
  Transform(buffer_);

  for (int i = 0; i < 8; ++i) {
    output[i * 4] = static_cast<uint8_t>((state_[i] >> 24) & 0xff);
    output[i * 4 + 1] = static_cast<uint8_t>((state_[i] >> 16) & 0xff);
    output[i * 4 + 2] = static_cast<uint8_t>((state_[i] >> 8) & 0xff);
    output[i * 4 + 3] = static_cast<uint8_t>(state_[i] & 0xff);
  }
}

std::string Sha256::FinalHex() {
  static constexpr char kHex[] = "0123456789ABCDEF";
  uint8_t output[kDigestSize];
  Final(output);

  std::string result(kDigestSize * 2, '0');
  for (size_t i = 0; i < kDigestSize; ++i) {
    result[i * 2] = kHex[(output[i] >> 4) & 0x0f];
    result[i * 2 + 1] = kHex[output[i] & 0x0f];
  }
  return result;
}

std::string Sha256::HashStringHex(std::string_view value) {
  Sha256 hash;
  hash.Update(value.data(), value.size());
  return hash.FinalHex();
}

}  // namespace onnxruntime::telemetry_internal
