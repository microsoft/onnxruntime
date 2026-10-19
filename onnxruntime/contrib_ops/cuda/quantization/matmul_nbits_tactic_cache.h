// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <string>

#if USE_FPA_INTB_GEMM && !defined(DISABLE_CONTRIB_OPS) && !defined(USE_CUDA_MINIMAL)
#include "contrib_ops/cuda/llm/gemm_tactic_cache.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Flushes all process-global MatMulNBits fpA_intB tactic caches to disk (best-effort, dirty-guarded).
//
// During a session, lazily profiled tactics are only staged into the in-memory caches (no disk I/O);
// this function performs the disk write. It is invoked at CUDA EP teardown: ~CUDAExecutionProvider for
// the built-in EP and ~CudaEp for the CUDA plugin EP, as well as provider shutdown before worker
// threads are stopped. In builds without onnxruntime_USE_FPA_INTB_GEMM there is no tactic cache and
// this is a no-op.
#if USE_FPA_INTB_GEMM && !defined(DISABLE_CONTRIB_OPS) && !defined(USE_CUDA_MINIMAL)
void FlushMatMulNBitsTacticCaches();
void FlushMatMulNBitsTacticCachesForShutdown();
std::shared_ptr<onnxruntime::llm::gemm_cache::MatMulNBitsTacticCache> GetGlobalMatMulNBitsTacticCacheForTesting(
    std::string file_path, onnxruntime::llm::gemm_cache::HardwareSignature signature, bool load);
std::unique_lock<std::mutex> LockMatMulNBitsTacticCacheRegistryForTesting();
#else
inline void FlushMatMulNBitsTacticCaches() {}
inline void FlushMatMulNBitsTacticCachesForShutdown() {}
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
