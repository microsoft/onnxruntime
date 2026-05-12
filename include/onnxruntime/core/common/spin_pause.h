// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace concurrency {

// Intrinsic to use in spin-loops
void SpinPause();

// Measure the average duration of a single SpinPause() call in nanoseconds.
// Runs exactly once per process (thread-safe via function-local static init).
// Used to convert a user-specified spin duration in microseconds into an
// iteration count, avoiding clock reads in the hot spin loop.
int CalibrateSpinPauseNs();

}  // namespace concurrency
}  // namespace onnxruntime
