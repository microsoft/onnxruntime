// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/variadic_elementwise_ops.h"

#include <cassert>

#include "core/framework/data_types_internal.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/variadic_elementwise_ops_impl.h"
#include "core/providers/cuda/math/variadic_elementwise_ops_tags.h"

namespace onnxruntime {
namespace cuda {

template <typename VariadicElementwiseOpTag, typename... SupportedElementTypes>
template <typename T>
Status VariadicElementwiseOp<VariadicElementwiseOpTag, SupportedElementTypes...>::
    NoBroadcastBatchImplDispatchTarget<T>::operator()(const InputTensorVector& inputs, Tensor& output) const {
  assert(inputs.size() > 1);

  using CudaT = typename ToCudaType<T>::MappedType;

  InputBatchArray<CudaT> input_data_batch{static_cast<int32_t>(inputs.size())};
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_data_batch[static_cast<int32_t>(i)] = reinterpret_cast<const CudaT*>(inputs[i].get().template Data<T>());
  }

  CudaT* output_data = reinterpret_cast<CudaT*>(output.template MutableData<T>());

  Impl_NoBroadcastInputBatch<CudaT, VariadicElementwiseOpTag>(
      input_data_batch, output_data, output.Shape().Size());

  return Status::OK();
}

// special case for 2 tensors to avoid memset zero
template <typename VariadicElementwiseOpTag, typename... SupportedElementTypes>
template <typename T>
Status VariadicElementwiseOp<VariadicElementwiseOpTag, SupportedElementTypes...>::
    BinaryImplDispatchTarget<T>::operator()(const Tensor& lhs, const Tensor& rhs, Tensor& output) const {
  using CudaT = typename ToCudaType<T>::MappedType;

  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(&lhs, &rhs, &output, &prepare));

  Impl_General<CudaT, VariadicElementwiseOpTag>(
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.lhs_tensor->template Data<T>()),
      &prepare.rhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.rhs_tensor->template Data<T>()),
      &prepare.fdm_output_strides,
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<CudaT*>(prepare.output_tensor->template MutableData<T>()),
      prepare.output_tensor->Shape().Size());

  return Status::OK();
}

// for more than 2 inputs, we need to accumulate into output tensor, as the shape from input0 + input1 might be different from output shape
template <typename VariadicElementwiseOpTag, typename... SupportedElementTypes>
template <typename T>
Status VariadicElementwiseOp<VariadicElementwiseOpTag, SupportedElementTypes...>::
    GeneralImplDispatchTarget<T>::operator()(const InputTensorVector& inputs, Tensor& output) const {
  assert(inputs.size() > 1);

  using CudaT = typename ToCudaType<T>::MappedType;

  CUDA_RETURN_IF_ERROR(cudaMemset(output.MutableDataRaw(), 0, output.SizeInBytes()));

  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(&output, &inputs[0].get(), &output, &prepare));

  Impl_Add(
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.lhs_tensor->template Data<T>()),
      &prepare.rhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.rhs_tensor->template Data<T>()),
      &prepare.fdm_output_strides,
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<CudaT*>(prepare.output_tensor->template MutableData<T>()),
      prepare.output_tensor->Shape().Size());

  for (size_t index = 1; index < inputs.size(); index++) {
    ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(&output, &inputs[index].get(), &output, &prepare));

    Impl_General<CudaT, VariadicElementwiseOpTag>(
        prepare.output_rank_or_simple_broadcast,
        &prepare.lhs_padded_strides,
        reinterpret_cast<const CudaT*>(prepare.lhs_tensor->template Data<T>()),
        &prepare.rhs_padded_strides,
        reinterpret_cast<const CudaT*>(prepare.rhs_tensor->template Data<T>()),
        &prepare.fdm_output_strides,
        prepare.fdm_H,
        prepare.fdm_C,
        reinterpret_cast<CudaT*>(prepare.output_tensor->template MutableData<T>()),
        prepare.output_tensor->Shape().Size());
  }

  return Status::OK();
}

template <typename VariadicElementwiseOpTag, typename... SupportedElementTypes>
Status VariadicElementwiseOp<VariadicElementwiseOpTag, SupportedElementTypes...>::ComputeInternal(
    OpKernelContext* context) const {
  const auto& node = Node();
  const auto& node_name = node.Name();
  auto input_count = node.InputArgCount().front();
  ORT_RETURN_IF_NOT(input_count >= 1, "Must have 1 or more inputs");

  const InputTensorVector input_tensors =
      [&context, input_count]() {
        InputTensorVector result{};
        result.reserve(input_count);
        for (int i = 0; i < input_count; ++i) {
          const auto& tensor = context->RequiredInput<Tensor>(i);
          result.push_back(std::cref(tensor));
        }
        return result;
      }();

  const auto& first_input_tensor = input_tensors[0].get();

  // special case for 1 input
  if (input_count == 1) {
    auto& output_tensor = context->RequiredOutput(0, first_input_tensor.Shape());
    if (first_input_tensor.DataRaw() != output_tensor.DataRaw()) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          output_tensor.MutableDataRaw(), first_input_tensor.DataRaw(), first_input_tensor.SizeInBytes(),
          cudaMemcpyDeviceToDevice));
    }

    return Status::OK();
  }

  const auto element_type = first_input_tensor.GetElementType();

  // special case for no broadcasting and few enough inputs
  if (input_count <= k_max_input_batch_size &&
      std::all_of(
          input_tensors.begin() + 1, input_tensors.end(),
          [&first_input_tensor](InputTensorVector::value_type t) {
            return first_input_tensor.Shape() == t.get().Shape();
          })) {
    auto& output_tensor = context->RequiredOutput(0, first_input_tensor.Shape());

    // special case for no broadcasting and 2 inputs
    if (input_count == 2) {
      utils::MLTypeCallDispatcherRet<Status, BinaryImplDispatchTarget, SupportedElementTypes...> dispatcher(element_type);
      ORT_RETURN_IF_ERROR(dispatcher.Invoke(input_tensors[0], input_tensors[1], output_tensor));

      return Status::OK();
    }

    utils::MLTypeCallDispatcherRet<Status, NoBroadcastBatchImplDispatchTarget, SupportedElementTypes...> dispatcher(
        element_type);
    ORT_RETURN_IF_ERROR(dispatcher.Invoke(input_tensors, output_tensor));

    return Status::OK();
  }

  // compute output shape first, using broadcast rule
  TensorShape output_shape;
  TensorShape previous_output_shape = first_input_tensor.Shape();
  for (int index = 1; index < input_count; index++) {
    ORT_RETURN_IF_ERROR(ComputeOutputShape(
        node_name, previous_output_shape, input_tensors[index].get().Shape(), output_shape));
    previous_output_shape = output_shape;
  }
  Tensor& output_tensor = context->RequiredOutput(0, output_shape);

  // special case for 2 inputs
  if (input_count == 2) {
    utils::MLTypeCallDispatcherRet<Status, BinaryImplDispatchTarget, SupportedElementTypes...> dispatcher(element_type);
    ORT_RETURN_IF_ERROR(dispatcher.Invoke(input_tensors[0], input_tensors[1], output_tensor));

    return Status::OK();
  }

  // general case for more than 2 inputs
  {
    utils::MLTypeCallDispatcherRet<Status, GeneralImplDispatchTarget, SupportedElementTypes...> dispatcher(
        element_type);
    ORT_RETURN_IF_ERROR(dispatcher.Invoke(input_tensors, output_tensor));
  }

  return Status::OK();
}

namespace {

using SumOp = VariadicElementwiseOp<
    variadic_elementwise_ops::Sum,
    MLFloat16, float, double>;

using MinOp = VariadicElementwiseOp<
    variadic_elementwise_ops::Min,
    uint32_t, uint64_t, int32_t, int64_t, MLFloat16, float, double>;

using MaxOp = VariadicElementwiseOp<
    variadic_elementwise_ops::Max,
    uint32_t, uint64_t, int32_t, int64_t, MLFloat16, float, double>;

const auto k_uzilhfd_datatypes =
    BuildKernelDefConstraints<uint32_t, uint64_t, int32_t, int64_t, MLFloat16, float, double>();
const auto k_hfd_datatypes =
    BuildKernelDefConstraints<MLFloat16, float, double>();

}  // namespace

// kernel registration

#define REGISTER_KERNEL(name, impl_class, version, datatypes) \
  ONNX_OPERATOR_KERNEL_EX(                                    \
      name,                                                   \
      kOnnxDomain,                                            \
      version,                                                \
      kCudaExecutionProvider,                                 \
      KernelDefBuilder().TypeConstraint("T", datatypes),      \
      impl_class)

#define REGISTER_VERSIONED_KERNEL(name, impl_class, start_version, end_version, datatypes) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                       \
      name,                                                                                \
      kOnnxDomain,                                                                         \
      start_version, end_version,                                                          \
      kCudaExecutionProvider,                                                              \
      KernelDefBuilder().TypeConstraint("T", datatypes),                                   \
      impl_class)

REGISTER_KERNEL(Sum, SumOp, 8, k_hfd_datatypes)
REGISTER_VERSIONED_KERNEL(Sum, SumOp, 6, 7, k_hfd_datatypes)

REGISTER_KERNEL(Min, MinOp, 12, k_uzilhfd_datatypes)
REGISTER_VERSIONED_KERNEL(Min, MinOp, 6, 11, k_hfd_datatypes)

REGISTER_KERNEL(Max, MaxOp, 12, k_uzilhfd_datatypes)
REGISTER_VERSIONED_KERNEL(Max, MaxOp, 6, 11, k_hfd_datatypes)

#undef REGISTER_VERSIONED_KERNEL
#undef REGISTER_KERNEL

}  // namespace cuda
}  // namespace onnxruntime
