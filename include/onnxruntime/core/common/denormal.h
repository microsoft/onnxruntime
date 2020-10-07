// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

// Return true if gcc has SSE support or msvc has x86-64 or x86 with SSE support.
// Return false if flush-to-zero and denormal-as-zero are not supported.
bool SetDenormalAsZero(bool on);

#ifdef _OPENMP
// Set flush-to-zero and denormal-as-zero on OpenMP threads when on is true.
void InitializeWithDenormalAsZero(bool on);
#endif

}  // namespace onnxruntime
