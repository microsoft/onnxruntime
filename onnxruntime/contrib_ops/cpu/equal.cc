// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "equal.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types.h"
#include <cmath>
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace contrib {

#define REG_ELEMENTWISE_LOGICALOP_TYPED_MS_KERNEL(OP_TYPE, VERSION, TYPE, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                 \
      OP_TYPE,                                                                       \
      VERSION,                                                                       \
      TYPE,                                                                          \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())    \
                        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),  \
      KERNEL_CLASS<TYPE>);

REG_ELEMENTWISE_LOGICALOP_TYPED_MS_KERNEL(Equal, 1, bool, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_MS_KERNEL(Equal, 1, int32_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_MS_KERNEL(Equal, 1, int64_t, Equal);
REG_ELEMENTWISE_LOGICALOP_TYPED_MS_KERNEL(Equal, 1, float, Equal);

template <typename T>
Status Equal<T>::Compute(OpKernelContext* context) const {
  return BroadcastTwo<T, bool>(
      *context,
      [](EigenVectorMap<bool> output, T input0, ConstEigenVectorMap<T> input1) { output = input1.array() == input0; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, T input1) { output = input0.array() == input1; },
      [](EigenVectorMap<bool> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1) { output = input0.array() == input1.array(); });
}

}  // namespace contrib
};  // namespace onnxruntime
