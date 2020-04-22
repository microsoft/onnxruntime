// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/slice.h"

using namespace onnxruntime::common;
using namespace std;

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    DynamicSlice,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", {DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Slice10);

}  // namespace contrib
}  // namespace onnxruntime
