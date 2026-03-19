// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/controlflow/scan_var_len.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Scan,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
    ScanVarLen);

}  // namespace contrib
}  // namespace onnxruntime
