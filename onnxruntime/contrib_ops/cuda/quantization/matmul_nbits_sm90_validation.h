// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Pure, host-only (no GPU required) helpers for validating a MatMulNBits node that requests
// weight_prepacked=2 (the native SM90/Hopper mixed-GEMM weight layout). These are declared here,
// separate from matmul_nbits.h, so that unit tests can exercise the validation logic with synthetic
// (sm, block_size) values -- e.g. a Hopper `sm` on a machine/build that has no GPU at all -- which is
// not otherwise possible since MatMulNBits<T> normally reads `sm` from the real device properties.
//
// Both functions are DEFINED (non-inline) in matmul_nbits.cc, NOT in this header. matmul_nbits.cc is
// compiled as part of the CUDA execution provider target (onnxruntime_providers_cuda /
// onnxruntime_providers_cuda_plugin), which is the only place where the COMPILE_HOPPER_TMA_GEMMS
// macro -- recording whether this build actually compiles the native SM90 (Hopper TMA/WGMMA)
// fpA_intB kernel; it is left undefined on Windows/MSVC, see cmake/onnxruntime_providers_cuda.cmake
// -- is defined consistently. If IsNativeSm90FpAIntBGemmCompiled() were instead defined inline in
// this header, each translation unit that includes it (including one that does not receive that
// target-scoped compile definition) could observe a different answer for whether the native SM90
// kernel is available, silently producing an ODR violation / macro-skew bug. Keeping a single
// non-inline definition inside matmul_nbits.cc avoids that hazard.
//
#pragma once

#include <cstdint>

#if USE_FPA_INTB_GEMM

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Returns true iff this build compiled the native SM90 (Hopper TMA/WGMMA) fpA_intB mixed-GEMM
// kernel, i.e. iff COMPILE_HOPPER_TMA_GEMMS was defined when matmul_nbits.cc was compiled.
// Pure/host-only: does not touch the GPU and can be called without a CUDA device present.
bool IsNativeSm90FpAIntBGemmCompiled();

// Validates that a MatMulNBits node requesting weight_prepacked=2 (the native SM90 weight layout)
// can actually be served, given the device compute capability `sm` (e.g. as computed from
// GetDeviceProp().major*10+minor) and the node's `block_size` attribute. Throws (ORT_ENFORCE /
// ORT_THROW) with a diagnostic message if the request cannot be served; returns normally otherwise.
// Pure/host-only: `sm` and `block_size` are plain parameters (not read from real hardware), so this
// can be unit-tested with synthetic values on any machine, without a GPU.
void ValidateSm90PrepackedWeightSupport(int sm, int64_t block_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // USE_FPA_INTB_GEMM
