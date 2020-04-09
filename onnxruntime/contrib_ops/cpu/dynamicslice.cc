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

ADD_TYPED_DYNAMIC_SLICE_OP(uint8_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint16_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(uint64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int8_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int16_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int32_t);
ADD_TYPED_DYNAMIC_SLICE_OP(int64_t);
ADD_TYPED_DYNAMIC_SLICE_OP(float);
ADD_TYPED_DYNAMIC_SLICE_OP(double);
ADD_TYPED_DYNAMIC_SLICE_OP(MLFloat16);
ADD_TYPED_DYNAMIC_SLICE_OP(bool);
ADD_TYPED_DYNAMIC_SLICE_OP(string);

}  // namespace contrib
}  // namespace onnxruntime
