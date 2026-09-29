// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Clean-room SHA-256 (FIPS 180-4) implementation. No external crypto deps.
// Intended for content-addressed asset hashing, not for cryptographic
// authentication.

#include "sha256.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace model_package {

namespace {

constexpr uint32_t kInitState[8] = {
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

inline uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t Bsig0(uint32_t x) { return Rotr(x, 2) ^ Rotr(x, 13) ^ Rotr(x, 22); }
inline uint32_t Bsig1(uint32_t x) { return Rotr(x, 6) ^ Rotr(x, 11) ^ Rotr(x, 25); }
inline uint32_t Ssig0(uint32_t x) { return Rotr(x, 7) ^ Rotr(x, 18) ^ (x >> 3); }
inline uint32_t Ssig1(uint32_t x) { return Rotr(x, 17) ^ Rotr(x, 19) ^ (x >> 10); }

}  // namespace

Sha256::Sha256() {
  std::memcpy(state_, kInitState, sizeof(state_));
  bit_count_ = 0;
  buffer_len_ = 0;
}

void Sha256::Transform(const uint8_t block[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
           (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
           (static_cast<uint32_t>(block[i * 4 + 3]));
  }
  for (int i = 16; i < 64; ++i) {
    w[i] = Ssig1(w[i - 2]) + w[i - 7] + Ssig0(w[i - 15]) + w[i - 16];
  }

  uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = h + Bsig1(e) + Ch(e, f, g) + kRoundConstants[i] + w[i];
    uint32_t t2 = Bsig0(a) + Maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
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

void Sha256::Update(const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  bit_count_ += static_cast<uint64_t>(len) * 8;
  while (len > 0) {
    size_t take = std::min<size_t>(64 - buffer_len_, len);
    std::memcpy(buffer_ + buffer_len_, p, take);
    buffer_len_ += take;
    p += take;
    len -= take;
    if (buffer_len_ == 64) {
      Transform(buffer_);
      buffer_len_ = 0;
    }
  }
}

void Sha256::Final(uint8_t out[kDigestSize]) {
  // Append 0x80, pad with zeros, append 64-bit big-endian length.
  buffer_[buffer_len_++] = 0x80;
  if (buffer_len_ > 56) {
    std::memset(buffer_ + buffer_len_, 0, 64 - buffer_len_);
    Transform(buffer_);
    buffer_len_ = 0;
  }
  std::memset(buffer_ + buffer_len_, 0, 56 - buffer_len_);
  uint64_t bc = bit_count_;
  for (int i = 7; i >= 0; --i) {
    buffer_[56 + i] = static_cast<uint8_t>(bc & 0xff);
    bc >>= 8;
  }
  Transform(buffer_);
  for (int i = 0; i < 8; ++i) {
    out[i * 4] = static_cast<uint8_t>((state_[i] >> 24) & 0xff);
    out[i * 4 + 1] = static_cast<uint8_t>((state_[i] >> 16) & 0xff);
    out[i * 4 + 2] = static_cast<uint8_t>((state_[i] >> 8) & 0xff);
    out[i * 4 + 3] = static_cast<uint8_t>(state_[i] & 0xff);
  }
}

namespace {
constexpr char kHex[] = "0123456789abcdef";
std::string ToHex(const uint8_t* bytes, size_t len) {
  std::string s(len * 2, '0');
  for (size_t i = 0; i < len; ++i) {
    s[i * 2] = kHex[(bytes[i] >> 4) & 0x0f];
    s[i * 2 + 1] = kHex[bytes[i] & 0x0f];
  }
  return s;
}
}  // namespace

std::string Sha256::FinalHex() {
  uint8_t out[kDigestSize];
  Final(out);
  return ToHex(out, kDigestSize);
}

std::string Sha256::HashBytesHex(const void* data, size_t len) {
  Sha256 h;
  h.Update(data, len);
  return h.FinalHex();
}

std::string Sha256::HashStringHex(const std::string& s) {
  return HashBytesHex(s.data(), s.size());
}

std::string Sha256::HashFileHex(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::string();
  Sha256 h;
  char buf[8192];
  while (f) {
    f.read(buf, sizeof(buf));
    std::streamsize n = f.gcount();
    if (n > 0) h.Update(buf, static_cast<size_t>(n));
  }
  return h.FinalHex();
}

}  // namespace model_package
