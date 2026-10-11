// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/gated_add.h"

#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GatedAdd,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GatedAdd<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

#undef REGISTER_KERNEL_TYPED

namespace {

// output = x + round_to_T(y * gate). For MLFloat16 the product is rounded to half before the
// add, matching separate ONNX Mul and Add operators.
template <typename T>
inline T GatedAddValue(T x, T y, T gate) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    const T product(y.ToFloat() * gate.ToFloat());
    return T(x.ToFloat() + product.ToFloat());
  } else {
    return x + y * gate;
  }
}

}  // namespace

template <typename T>
Status GatedAdd<T>::Compute(OpKernelContext* context) const {
  const Tensor* x = context->Input<Tensor>(0);
  const Tensor* y = context->Input<Tensor>(1);
  const Tensor* gate = context->Input<Tensor>(2);
  const TensorShape& shape = x->Shape();

  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "X must have rank >= 1");
  ORT_RETURN_IF_NOT(y->Shape() == shape, "Y must have the same shape as X");
  ORT_RETURN_IF_NOT(gate->Shape().NumDimensions() == shape.NumDimensions(),
                    "gate must have the same rank as X");

  const size_t last_axis = shape.NumDimensions() - 1;
  const int64_t hidden_size = shape[last_axis];
  ORT_RETURN_IF_NOT(hidden_size > 0, "X last dimension must be positive");
  ORT_RETURN_IF_NOT(gate->Shape()[last_axis] == 1, "gate last dimension must be 1");
  for (size_t axis = 0; axis < last_axis; ++axis) {
    ORT_RETURN_IF_NOT(gate->Shape()[axis] == shape[axis],
                      "gate dimension ", axis, " must match X");
  }

  Tensor* output = context->Output(0, shape);
  const int64_t count = shape.Size();
  if (count == 0) {
    return Status::OK();
  }

  const T* x_data = x->Data<T>();
  const T* y_data = y->Data<T>();
  const T* gate_data = gate->Data<T>();
  T* output_data = output->MutableData<T>();
  const int64_t num_rows = count / hidden_size;

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(num_rows),
      [&](ptrdiff_t row) {
        const int64_t offset = row * hidden_size;
        const T gate_value = gate_data[row];
        for (int64_t i = 0; i < hidden_size; ++i) {
          output_data[offset + i] = GatedAddValue<T>(x_data[offset + i], y_data[offset + i], gate_value);
        }
      },
      0);

  return Status::OK();
}

template class GatedAdd<float>;
template class GatedAdd<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
