// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

// the semantics of this enum should match std::endian from C++20
enum class endian {
  // little-endian byte order
  little = 0,
  // big-endian byte order
  big = 1,
#if defined(ORT_IS_LITTLE_ENDIAN)
  // native byte order
  native = little,
#elif defined(ORT_IS_BIG_ENDIAN)
  // native byte order
  native = big,
#else
  // native byte order
  native = 2,
#endif
};

static_assert(
    endian::native == endian::little || endian::native == endian::big,
    "Only little-endian or big-endian native byte orders are supported.");

}  // namespace onnxruntime
