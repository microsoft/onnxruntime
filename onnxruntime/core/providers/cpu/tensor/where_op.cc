// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/where_op.h"

#include <algorithm>
#include <type_traits>

#include "core/providers/cpu/math/element_wise_ops.h"  // for broadcast utilities

namespace onnxruntime {
// kernel builder functions
#define WHERE_VERSIONED_TYPED_KERNEL_WITH_TYPE_NAME(type, type_name)               \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                        \
      Where,                                                                       \
      9,                                                                           \
      15,                                                                          \
      type_name,                                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      Where<type>)

#define WHERE_VERSIONED_TYPED_KERNEL(type) \
  WHERE_VERSIONED_TYPED_KERNEL_WITH_TYPE_NAME(type, type)

WHERE_VERSIONED_TYPED_KERNEL(uint8_t)
WHERE_VERSIONED_TYPED_KERNEL(int32_t)
WHERE_VERSIONED_TYPED_KERNEL(int64_t)
WHERE_VERSIONED_TYPED_KERNEL(float)
WHERE_VERSIONED_TYPED_KERNEL(double)
WHERE_VERSIONED_TYPED_KERNEL_WITH_TYPE_NAME(std::string, string)

#define WHERE_TYPED_KERNEL_WITH_TYPE_NAME(type, type_name)                         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      Where,                                                                       \
      16,                                                                          \
      type_name,                                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      Where<type>)

#define WHERE_TYPED_KERNEL(type) \
  WHERE_TYPED_KERNEL_WITH_TYPE_NAME(type, type)

// start with a subset of types, enable more as needed...
WHERE_TYPED_KERNEL(uint8_t)
// WHERE_TYPED_KERNEL(uint16_t)
// WHERE_TYPED_KERNEL(uint32_t)
// WHERE_TYPED_KERNEL(uint64_t)
// WHERE_TYPED_KERNEL(int8_t)
// WHERE_TYPED_KERNEL(int16_t)
WHERE_TYPED_KERNEL(int32_t)
WHERE_TYPED_KERNEL(int64_t)
// WHERE_TYPED_KERNEL(MLFloat16)
// WHERE_TYPED_KERNEL(BFloat16)
WHERE_TYPED_KERNEL(float)
WHERE_TYPED_KERNEL(double)
WHERE_TYPED_KERNEL_WITH_TYPE_NAME(std::string, string)
// WHERE_TYPED_KERNEL(bool)

#undef WHERE_TYPED_KERNEL_WITH_TYPE_NAME
#undef WHERE_TYPED_KERNEL

namespace {

template <typename T, typename R>
using EnableIfEigenScalar = typename std::enable_if<std::is_arithmetic<T>::value, R>::type;

template <typename T, typename R>
using EnableIfEigenNotScalar = typename std::enable_if<!std::is_arithmetic<T>::value, R>::type;

template <typename T>
ProcessBroadcastSpanFuncs CreateScalarBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      [](BroadcastHelper& per_iter_bh) {
        bool target = (per_iter_bh.GetUserData() != nullptr);
        bool condition = per_iter_bh.ScalarInput0<bool>();
        auto value = per_iter_bh.EigenInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();
        if (condition == target) {
          output = value;
        } else {
          output = EigenVectorMap<T>::PlainObject::Constant(value.size(), T{});
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        bool target = (per_iter_bh.GetUserData() != nullptr);
        auto condition = per_iter_bh.EigenInput0<bool>();
        const T& value = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();
        output = (condition.array() == target)
                     .select(value, EigenVectorMap<T>::PlainObject::Constant(condition.size(), T{}));
      },
      [](BroadcastHelper& per_iter_bh) {
        bool target = (per_iter_bh.GetUserData() != nullptr);
        auto condition = per_iter_bh.EigenInput0<bool>();
        auto value = per_iter_bh.EigenInput1<T>();
        auto output = per_iter_bh.OutputEigen<T>();
        output = (condition.array() == target)
                     .select(value, EigenVectorMap<T>::PlainObject::Constant(condition.size(), T{}));
      }};
}

template <typename T>
ProcessBroadcastSpanFuncs CreateNonScalarBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      [](BroadcastHelper& per_iter_bh) {
        bool target = (per_iter_bh.GetUserData() != nullptr);
        bool condition = per_iter_bh.ScalarInput0<bool>();
        auto value = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        if (condition == target) {
          std::copy(value.begin(), value.end(), output.begin());
        } else {
          std::fill(output.begin(), output.end(), T{});
        }
      },
      [](BroadcastHelper& per_iter_bh) {
        bool target = (per_iter_bh.GetUserData() != nullptr);
        auto condition = per_iter_bh.SpanInput0<bool>();
        const T& value = per_iter_bh.ScalarInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        std::transform(condition.begin(), condition.end(), output.begin(),
                       [target, &value](bool condition_element) {
                         return condition_element == target ? value : T{};
                       });
      },
      [](BroadcastHelper& per_iter_bh) {
        bool target = (per_iter_bh.GetUserData() != nullptr);
        auto condition = per_iter_bh.SpanInput0<bool>();
        auto value = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        std::transform(condition.begin(), condition.end(), value.begin(), output.begin(),
                       [target](bool condition_element, const T& value_element) {
                         return condition_element == target ? value_element : T{};
                       });
      }};
}

template <typename T>
EnableIfEigenScalar<T, ProcessBroadcastSpanFuncs> SelectBroadcastFuncs() {
  // NOTE: Workaround a VS2017 bug by calling a separate function to create the broadcast funcs.
  // If we create them directly here it doesn't bring in the definitions of the Eigen classes leading to
  // a 'class has no constructors' error
  return CreateScalarBroadcastFuncs<T>();
}

template <typename T>
EnableIfEigenNotScalar<T, ProcessBroadcastSpanFuncs> SelectBroadcastFuncs() {
  return CreateNonScalarBroadcastFuncs<T>();
}

template <typename T>
void MergeScalarAndVector(EigenVectorMap<T> output, const T& scalar_value, ConstEigenVectorMap<T> vector_value) {
  if (scalar_value != T{}) {
    output = EigenVectorMap<T>::PlainObject::Constant(vector_value.size(), scalar_value);
  } else {
    output = vector_value;
  }
};

template <typename T>
EnableIfEigenScalar<T, ProcessBroadcastSpanFuncs> MergeBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputEigen<T>(),
                             per_iter_bh.ScalarInput0<T>(),  // X selection
                             per_iter_bh.EigenInput1<T>());  // Y selection
      },
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputEigen<T>(),
                             per_iter_bh.ScalarInput1<T>(),  // Y selection
                             per_iter_bh.EigenInput0<T>());  // X selection
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X_selection = per_iter_bh.EigenInput0<T>();
        auto Y_selection = per_iter_bh.EigenInput1<T>();
        per_iter_bh.OutputEigen<T>() = X_selection.binaryExpr(Y_selection,
                                                              [](T x, T y) -> T {
                                                                return x != T{} ? x : y;
                                                              });
      }};
}

template <typename T>
void MergeScalarAndVector(gsl::span<T> output, const T& scalar_value, gsl::span<const T> vector_value) {
  if (!scalar_value.empty()) {
    std::fill(output.begin(), output.end(), scalar_value);
  } else {
    std::copy(vector_value.begin(), vector_value.end(), output.begin());
  }
};

template <typename T>
EnableIfEigenNotScalar<T, ProcessBroadcastSpanFuncs> MergeBroadcastFuncs() {
  return ProcessBroadcastSpanFuncs{
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputSpan<T>(),
                             per_iter_bh.ScalarInput0<T>(),  // X selection
                             per_iter_bh.SpanInput1<T>());   // Y selection
      },
      [](BroadcastHelper& per_iter_bh) {
        MergeScalarAndVector(per_iter_bh.OutputSpan<T>(),
                             per_iter_bh.ScalarInput1<T>(),  // Y selection
                             per_iter_bh.SpanInput0<T>());   // X selection
      },
      [](BroadcastHelper& per_iter_bh) {
        auto X_selection = per_iter_bh.SpanInput0<T>();
        auto Y_selection = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();
        std::transform(X_selection.begin(), X_selection.end(), Y_selection.begin(), output.begin(),
                       [](const T& x, const T& y) { return !x.empty() ? x : y; });
      }};
}

// function pointer to create typed tensor from type agnostic code whilst avoiding the overhead of std::function
using AllocTensorFunc = std::unique_ptr<Tensor> (*)(const TensorAllocator& allocator, const TensorShape& shape);

static std::unique_ptr<Tensor> UntypedSelect(OpKernelContext& context, bool target,
                                             const TensorAllocator& allocator, AllocTensorFunc allocate_tensor,
                                             const ProcessBroadcastSpanFuncs& functors) {
  const auto& condition = *context.Input<Tensor>(0);
  // select the X input (input 1) for 'true', and Y input (input 2) for 'false'
  const auto& values = *context.Input<Tensor>(target ? 1 : 2);

  InputBroadcaster input_broadcaster(condition, values);

  std::unique_ptr<Tensor> selection_tensor = allocate_tensor(allocator, input_broadcaster.GetOutputShape());
  OutputBroadcaster output_broadcaster(input_broadcaster.GetSpanSize(), *selection_tensor);

  // store value of 'target' directly in void* for user_data so it's accessible in the state-less functors
  BroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster, reinterpret_cast<void*>(target));

  BroadcastLooper(broadcast_helper, functors);

  return selection_tensor;
}

static void UntypedMerge(OpKernelContext& context,
                         const Tensor& X_selection_tensor, const Tensor& Y_selection_tensor,
                         const ProcessBroadcastSpanFuncs& functors) {
  InputBroadcaster merge_broadcaster{X_selection_tensor, Y_selection_tensor};
  Tensor& output = *context.Output(0, merge_broadcaster.GetOutputShape());

  OutputBroadcaster output_broadcaster{merge_broadcaster.GetSpanSize(), output};
  BroadcastHelper broadcast_helper(merge_broadcaster, output_broadcaster);

  BroadcastLooper(broadcast_helper, functors);
}
}  // namespace

template <typename T>
Status Where<T>::Compute(OpKernelContext* context) const {
  // we use a func pointer to save the overhead of std::function, so we can't capture tensor_allocator here
  const auto typed_tensor_allocation = [](const TensorAllocator& allocator,
                                          const TensorShape& shape) {
    return allocator.Allocate<T>(shape);
  };

  TensorAllocator tensor_allocator{*context};
  ProcessBroadcastSpanFuncs funcs = SelectBroadcastFuncs<T>();

  // The current implementation is limited to broadcasting over two tensors at once.
  // So, we first broadcast over condition and X to select the values from X:
  //   X_selection = condition ? X : default value
  // Similarly, we broadcast over condition and Y to select the values from Y:
  //   Y_selection = !condition ? Y : default value
  //
  // These selections are handled within UntypedSelect.
  //
  // Finally, we broadcast over and merge X_selection and Y_selection:
  //   output = (X_selection != default value) ? X_selection : Y_selection
  //
  // The merging is handled within UntypedMerge.
  auto X_selection_tensor = UntypedSelect(*context, true, tensor_allocator, typed_tensor_allocation, funcs);
  auto Y_selection_tensor = UntypedSelect(*context, false, tensor_allocator, typed_tensor_allocation, funcs);

  UntypedMerge(*context, *X_selection_tensor, *Y_selection_tensor, MergeBroadcastFuncs<T>());

  return Status::OK();
}

}  // namespace onnxruntime
