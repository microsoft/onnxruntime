// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Flushes all process-global MatMulNBits fpA_intB tactic caches to disk (best-effort, dirty-guarded).
//
// During a session, MatMulNBits kernel destructors only STAGE lazily-discovered tactics into the
// in-memory caches (no disk I/O); this function performs the single deterministic disk write. It is
// invoked at CUDA EP teardown (CUDAExecutionProvider destructor), which covers both the built-in EP
// and the CUDA plugin EP (CudaEp wraps a CUDAExecutionProvider), so tuned tactics are persisted at
// session end. In builds without onnxruntime_USE_FPA_INTB_GEMM there is no tactic cache and this is
// a no-op.
#if USE_FPA_INTB_GEMM
void FlushMatMulNBitsTacticCaches();
#else
inline void FlushMatMulNBitsTacticCaches() {}
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
