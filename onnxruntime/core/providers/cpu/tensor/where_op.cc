// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/where_op.h"

#include "core/providers/cpu/math/element_wise_ops.h"  // for broadcast utilities

namespace onnxruntime {

// kernel builder functions
#define WHERE_TYPED_KERNEL_WITH_TYPE_NAME(type, type_name)                         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      Where,                                                                       \
      9,                                                                           \
      type_name,                                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      Where<type>)

#define WHERE_TYPED_KERNEL(type) \
  WHERE_TYPED_KERNEL_WITH_TYPE_NAME(type, type)

// start with a subset of types, enable more as needed...
//WHERE_TYPED_KERNEL(uint8_t)
//WHERE_TYPED_KERNEL(uint16_t)
//WHERE_TYPED_KERNEL(uint32_t)
//WHERE_TYPED_KERNEL(uint64_t)
//WHERE_TYPED_KERNEL(int8_t)
//WHERE_TYPED_KERNEL(int16_t)
//WHERE_TYPED_KERNEL(int32_t)
//WHERE_TYPED_KERNEL(int64_t)
//WHERE_TYPED_KERNEL(MLFloat16)
//WHERE_TYPED_KERNEL(BFloat16)
WHERE_TYPED_KERNEL(float)
//WHERE_TYPED_KERNEL(double)
WHERE_TYPED_KERNEL_WITH_TYPE_NAME(std::string, string)
//WHERE_TYPED_KERNEL(bool)

#undef WHERE_TYPED_KERNEL_WITH_TYPE_NAME
#undef WHERE_TYPED_KERNEL

namespace {
template <typename T>
std::unique_ptr<Tensor> WhereSelection(bool target, const Tensor& condition_tensor, const Tensor& value_tensor,
                                       TensorAllocator<T>& tensor_allocator) {
  TBroadcaster<bool, T> selection_broadcaster{condition_tensor, value_tensor};
  std::unique_ptr<Tensor> selection_tensor{
      tensor_allocator.Allocate(selection_broadcaster.GetOutputShape())};
  TBroadcastOutput<T> selection_broadcast_output{
      selection_broadcaster.GetSpanSize(), *selection_tensor};

  BroadcastLoop(
      selection_broadcaster, selection_broadcast_output,
      [target](EigenVectorMap<T> output, bool condition, ConstEigenVectorMap<T> value) {
        if (condition == target) {
          output = value;
        } else {
          output = EigenVectorMap<T>::PlainObject::Constant(value.size(), T{});
        }
      },
      [target](EigenVectorMap<T> output, ConstEigenVectorMap<bool> condition, const T& value) {
        output = (condition.array() == target)
                     .select(value, EigenVectorMap<T>::PlainObject::Constant(condition.size(), T{}));
      },
      [target](EigenVectorMap<T> output, ConstEigenVectorMap<bool> condition, ConstEigenVectorMap<T> value) {
        output = (condition.array() == target)
                     .select(value, EigenVectorMap<T>::PlainObject::Constant(condition.size(), T{}));
      });

  return selection_tensor;
}
}  // namespace

template <typename T>
Status Where<T>::Compute(OpKernelContext* context) const {
  const Tensor* const condition = context->Input<Tensor>(0);
  const Tensor* const X = context->Input<Tensor>(1);
  const Tensor* const Y = context->Input<Tensor>(2);
  ORT_ENFORCE(condition && X && Y, "condition, X, and Y inputs are required!");

  // The current implementation is limited to broadcasting over two tensors at once.
  // So, we first broadcast over condition and X to select the values from X:
  //   X_selection = condition ? X : default value
  // Similarly, we broadcast over condition and Y to select the values from Y:
  //   Y_selection = !condition ? Y : default value
  // Finally, we merge X_selection and Y_selection:
  //   output = (X_selection != default value) ? X_selection : Y_selection
  TensorAllocator<T> tensor_allocator{*context};
  auto X_selection_tensor = WhereSelection<T>(true, *condition, *X, tensor_allocator);
  auto Y_selection_tensor = WhereSelection<T>(false, *condition, *Y, tensor_allocator);

  TBroadcaster<T, T> merge_broadcaster{*X_selection_tensor, *Y_selection_tensor};
  Tensor* const output = context->Output(0, merge_broadcaster.GetOutputShape());
  ORT_ENFORCE(output, "failed to get first output!");
  TBroadcastOutput<T> merge_broadcast_output{
      merge_broadcaster.GetSpanSize(), *output};

  BroadcastLoop(
      merge_broadcaster, merge_broadcast_output,
      [](EigenVectorMap<T> output, const T& X_selection, ConstEigenVectorMap<T> Y_selection) {
        if (X_selection != T{}) {
          output = EigenVectorMap<T>::PlainObject::Constant(Y_selection.size(), X_selection);
        } else {
          output = Y_selection;
        }
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> X_selection, const T& Y_selection) {
        if (Y_selection != T{}) {
          output = EigenVectorMap<T>::PlainObject::Constant(X_selection.size(), Y_selection);
        } else {
          output = X_selection;
        }
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> X_selection, ConstEigenVectorMap<T> Y_selection) {
        output = X_selection.binaryExpr(Y_selection, [](T x, T y) -> T {
          return x != T{} ? x : y;
        });
      });

  return Status::OK();
}
}  // namespace onnxruntime
