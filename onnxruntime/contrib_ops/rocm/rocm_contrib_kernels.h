// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace rocm {

Status RegisterRocmContribKernels(KernelRegistry& kernel_registry);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
