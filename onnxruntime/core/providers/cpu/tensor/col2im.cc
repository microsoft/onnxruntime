// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/col2im.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/common.h"
#include "core/framework/copy.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

#define REGISTER_KERNEL_TYPED(T)                                                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                           \
      Col2Im,                                                                               \
      1,                                                                                    \
      T,                                                                                    \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()), \
      Col2Im<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
Status Col2Im<T>::Compute(OpKernelContext* context) const {
  (void) context;
  return Status::OK();
}

}  // namespace onnxruntime
