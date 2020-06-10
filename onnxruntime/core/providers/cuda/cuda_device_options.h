// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#pragma once

namespace onnxruntime {

/**
  * Configuration information for a provider.
  */
struct CudaDeviceOptions {

    OrtDevice::DeviceId device_id = 0;
    size_t cuda_mem_limit = std::numeric_limits<size_t>::max();
    onnxruntime::ArenaExtendStrategy arena_extend_strategy = onnxruntime::ArenaExtendStrategy::kNextPowerOfTwo;
};
}  // namespace onnxruntime
