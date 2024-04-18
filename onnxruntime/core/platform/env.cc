/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include "core/platform/env.h"

namespace onnxruntime {

std::ostream& operator<<(std::ostream& os, const LogicalProcessors& aff) {
  os << "{";
  std::copy(aff.cbegin(), aff.cend(), std::ostream_iterator<int>(os, ", "));
  return os << "}";
}

std::ostream& operator<<(std::ostream& os, gsl::span<const LogicalProcessors> affs) {
  os << "{";
  for (const auto& aff : affs) {
    os << aff;
  }
  return os << "}";
}

Env::Env() = default;

std::pair<int, std::string> GetErrnoInfo() {
  auto err = errno;
  std::string msg;

  if (err != 0) {
    char buf[512];

#if defined(_WIN32)
    auto ret = strerror_s(buf, sizeof(buf), err);
    msg = ret == 0 ? buf : "Failed to get error message";  // buf is guaranteed to be null terminated by strerror_s
#else
    // strerror_r return type differs by platform.
    auto ret = strerror_r(err, buf, sizeof(buf));
    if constexpr (std::is_same_v<decltype(ret), int>) {  // POSIX returns int
      msg = ret == 0 ? buf : "Failed to get error message";
    } else {
      // GNU returns char*
      msg = ret;
    }
#endif
  }

  return {err, msg};
}

}  // namespace onnxruntime
