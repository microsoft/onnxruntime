// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

Status RegisterCudaTrainingKernels(KernelRegistry& kernel_registry);

}  // namespace cuda
}  // namespace onnxruntime
