// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

/**
 * Gets whether a CPU allocator should use an arena or not.
 */
bool ShouldCpuAllocatorUseArena(bool is_arena_requested);

}  // namespace onnxruntime
