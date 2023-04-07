// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    Snpe,
    kMSDomain,
    1,
    kSnpeExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<float, uint8_t, uint16_t>()),
    SnpeKernel);

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
