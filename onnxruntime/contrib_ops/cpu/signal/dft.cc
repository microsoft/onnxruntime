// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

#include "core/providers/cpu/signal/dft.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
// Some of these were added to the standard op set. Register them under the MS domain
// for backwards compatibility.
ONNX_OPERATOR_KERNEL_EX(
    DFT,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()                                                     //
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())  //
        .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
    DFT);

// TODO: add function body
ONNX_OPERATOR_KERNEL_EX(
    IDFT,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()                                                     //
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())  //
        .TypeConstraint("T2", BuildKernelDefConstraints<int64_t>()),
    IDFT);

ONNX_OPERATOR_KERNEL_EX(
    STFT,
    kMSExperimentalDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().MayInplace(0, 0)                                    //
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double>())  //
        .TypeConstraint("T2", BuildKernelDefConstraints<int32_t, int64_t>()),
    STFT);
}  // namespace contrib
}  // namespace onnxruntime

#endif
