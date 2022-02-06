// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {
namespace deep_speed {

Status RegisterDeepSpeedKernels(KernelRegistry& kernel_registry);

}  // namespace deep_speed
}  // namespace cuda
}  // namespace onnxruntime
