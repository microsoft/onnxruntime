// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file sha256.h
/// \brief Minimal SHA-256 implementation used for content-addressed assets.
///        No external crypto dependency.

#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>

namespace model_package {

class Sha256 {
 public:
  static constexpr size_t kDigestSize = 32;

  Sha256();
  void Update(const void* data, size_t len);
  void Update(const std::string& s) { Update(s.data(), s.size()); }
  void Final(uint8_t out[kDigestSize]);

  /// Hex-encoded (lowercase) digest, 64 chars.
  std::string FinalHex();

  static std::string HashBytesHex(const void* data, size_t len);
  static std::string HashStringHex(const std::string& s);

  /// Stream-hash a file by path. Returns the hex digest, or empty string on
  /// IO error (caller should pre-check existence).
  static std::string HashFileHex(const std::string& path);

 private:
  void Transform(const uint8_t block[64]);
  uint32_t state_[8];
  uint64_t bit_count_;
  uint8_t buffer_[64];
  size_t buffer_len_;
};

}  // namespace model_package
