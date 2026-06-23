// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

class OpTester;

// How two sessions are configured to share the pre-packed weights of a MatMulNBits node.
enum class PrepackSharingMode {
  // session.share_matmulnbits_prepacked_weights = "1": MatMulNBits pre-packed weights participate in the
  // shared container, content-addressed by hash(packed_bytes). No OrtApi::AddInitializer call is required.
  // This is the path used by the DQ + MatMul -> MatMulNBits fusion, whose weights are synthesized at
  // session-creation time with auto-generated names.
  kShareMatMulNBitsPrepackedWeights,
  // Legacy path: the weight is explicitly registered as a shared initializer via
  // SessionOptions::AddInitializer.
  kAddInitializer,
  // Negative control: the shared container exists but neither opt-in mechanism is used, so no
  // cross-session sharing must happen.
  kNoSharing,
};

// Runs the already-configured MatMulNBits OpTester in two CPU sessions that share the same
// pre-packed weights container and asserts that the pre-packed weights are shared as expected.
// This logic is independent of the weight bit width, so it is shared by the 4-bit and 8-bit tests.
// `b_dims`/`b_data` describe the quantized B initializer and are only needed for the
// PrepackSharingMode::kAddInitializer path (to register B as a shared initializer).
void CheckSharedPrepackedWeights(OpTester& test, PrepackSharingMode mode,
                                 const std::vector<int64_t>& b_dims,
                                 std::vector<uint8_t>& b_data);

}  // namespace test
}  // namespace onnxruntime
