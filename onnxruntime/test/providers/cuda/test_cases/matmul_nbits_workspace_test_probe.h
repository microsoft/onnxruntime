// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Narrow bridge between the two translation units of the MatMulNBits workspace pilot's end-to-end
// test (Phase-A memory roadmap, issue microsoft/onnxruntime#29775):
//   - matmul_nbits_workspace_test.cc       : compiled with the CUDA-provider (shared-provider bridge)
//                                            headers; it knows the concrete MatMulNBits<T> type and
//                                            defines the probe below.
//   - matmul_nbits_e2e_workspace_test.cc   : compiled with the core framework headers (InferenceSession,
//                                            SessionState); it runs a real session and calls the probe.
// The two header worlds redefine the same logging macros and therefore cannot be mixed in a single
// translation unit, so this header intentionally pulls in NEITHER of them: it only forward-declares
// onnxruntime::OpKernel and uses size_t.

#include <cstddef>

namespace onnxruntime {
class OpKernel;

namespace test {

// Returns the workspace size (bytes) the fpA_intB CUTLASS runner requested on the most recent
// ComputeInternal() call of the given kernel, via MatMulNBits<MLFloat16>::LastComputeWorkspaceBytes().
// The caller MUST guarantee that `kernel` is a fp16 (MLFloat16) MatMulNBits kernel instance; the
// implementation performs a fixed-offset static_cast (MatMulNBits is single, non-virtual inheritance
// from OpKernel). See the definition in matmul_nbits_workspace_test.cc.
size_t GetMatMulNBitsLastComputeWorkspaceBytes(const OpKernel* kernel);

}  // namespace test
}  // namespace onnxruntime
