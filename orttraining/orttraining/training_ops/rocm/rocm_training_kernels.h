// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace rocm {

Status RegisterRocmTrainingKernels(KernelRegistry& kernel_registry);

}  // namespace rocm
}  // namespace onnxruntime
