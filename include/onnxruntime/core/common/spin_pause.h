// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_M_AMD64) || defined(__x86_64__)
#include <cstdint>
#endif

namespace onnxruntime {
namespace concurrency {

// Intrinsic to use in spin-loops
void SpinPause();

}  // namespace concurrency
}  // namespace onnxruntime
