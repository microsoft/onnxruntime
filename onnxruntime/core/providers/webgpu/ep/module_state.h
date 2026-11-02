// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::webgpu::ep {

// Release module-owned registries, contexts and logging after all EP users have
// been destroyed. Shared builds also shut down their private protobuf state;
// static builds leave the host's protobuf state intact.
void CleanupModuleState();

}  // namespace onnxruntime::webgpu::ep
