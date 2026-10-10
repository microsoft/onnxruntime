// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

Status RegisterCudaContribKernels(KernelRegistry& kernel_registry,
                                  bool enable_host_pageable_gather = false);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
