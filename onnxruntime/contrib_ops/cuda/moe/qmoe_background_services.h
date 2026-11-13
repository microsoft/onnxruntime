// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

#if !defined(DISABLE_CONTRIB_OPS) && !defined(USE_CUDA_MINIMAL)
// Stops and joins the process-wide QMoE and MoEPacked background threads: the CPU expert doorbell services and the
// shared CPU spin pools.
//
// Provider teardown must call this before the provider library is unloaded. Joining a thread from a static
// destructor while FreeLibrary holds the Windows loader lock deadlocks, because the exiting thread needs that lock.
// Idempotent. The services start again on demand if a QMoE or MoEPacked kernel is created afterwards, and nothing is
// stopped while a corresponding offload backend is still alive.
void ShutdownQMoEBackgroundServices() noexcept;
void ShutdownMoEPackedBackgroundServices() noexcept;
#else
inline void ShutdownQMoEBackgroundServices() noexcept {}
inline void ShutdownMoEPackedBackgroundServices() noexcept {}
#endif

}  // namespace onnxruntime::contrib::cuda
