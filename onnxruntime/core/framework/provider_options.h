// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#pragma once

namespace onnxruntime {

/**
  * Configuration information for a provider.
  */
struct DeviceOptions {

    OrtDevice::DeviceId device_id = 0;
};
}  // namespace onnxruntime
