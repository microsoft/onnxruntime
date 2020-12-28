// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>

namespace onnxruntime {

enum class ArenaExtendStrategy : int32_t {
  kNextPowerOfTwo = 0,
  kSameAsRequested,
};

inline std::istream& operator>>(std::istream& is, ArenaExtendStrategy& value) {
  std::string value_str;
  if (is >> value_str) {
    if (value_str == "kNextPowerOfTwo") {
      value = ArenaExtendStrategy::kNextPowerOfTwo;
    } else if (value_str == "kSameAsRequested") {
      value = ArenaExtendStrategy::kSameAsRequested;
    } else {
      is.setstate(std::ios_base::failbit);
    }
  }
  return is;
}

inline std::ostream& operator<<(std::ostream& os, ArenaExtendStrategy value) {
  switch (value) {
    case ArenaExtendStrategy::kNextPowerOfTwo:
      os << "kNextPowerOfTwo";
      break;
    case ArenaExtendStrategy::kSameAsRequested:
      os << "kSameAsRequested";
      break;
    default:
      os << "unknown";
      break;
  }
  return os;
}

}  // namespace onnxruntime
