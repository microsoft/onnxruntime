// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace onnxruntime::telemetry_internal {

class Sha256 {
 public:
  static constexpr size_t kDigestSize = 32;

  Sha256();
  void Update(const void* data, size_t length);
  void Final(uint8_t output[kDigestSize]);
  std::string FinalHex();

  static std::string HashStringHex(std::string_view value);

 private:
  void Transform(const uint8_t block[64]);

  uint32_t state_[8];
  uint64_t bit_count_{};
  uint8_t buffer_[64]{};
  size_t buffer_length_{};
};

}  // namespace onnxruntime::telemetry_internal
