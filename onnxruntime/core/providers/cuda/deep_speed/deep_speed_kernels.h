// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace deep_speed {
namespace cuda {

Status RegisterDeepSpeedKernels(KernelRegistry& kernel_registry);

}  // namespace cuda
}  // namespace deep_speed
}  // namespace onnxruntime
