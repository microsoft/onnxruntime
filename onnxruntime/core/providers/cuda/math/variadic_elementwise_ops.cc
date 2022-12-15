// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/math/variadic_elementwise_ops.h"

#include <cassert>
#include <algorithm>

#include "core/framework/data_types_internal.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/variadic_elementwise_ops_impl.h"
#include "core/providers/cuda/math/variadic_elementwise_ops_tags.h"

namespace onnxruntime {
namespace cuda {

template <typename VariadicElementwiseOpTag, typename... SupportedElementTypes>
template <typename T>
Status VariadicElementwiseOp<VariadicElementwiseOpTag, SupportedElementTypes...>::NoBroadcastBatchImplDispatchTarget<
    T>::operator()(cudaStream_t stream, const InputTensorVector& inputs, Tensor& output) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  size_t input_count = inputs.size();
  assert(input_count > 1);
  size_t index = std::min(input_count, static_cast<size_t>(k_max_input_batch_size));
  InputBatchArray<CudaT> input_data_batch{static_cast<int32_t>(index)};
  for (size_t i = 0; i < index; ++i) {
    input_data_batch[static_cast<int32_t>(i)] = reinterpret_cast<const CudaT*>(inputs[i].get().Data<T>());
  }

  CudaT* output_data = reinterpret_cast<CudaT*>(output.MutableData<T>());
  Impl_NoBroadcastInputBatch<CudaT, VariadicElementwiseOpTag>(stream, input_data_batch, output_data,
                                                              output.Shape().Size());

  while (index < input_count) {
    size_t left_count = input_count - index + 1;
    size_t batch = std::min(left_count, static_cast<size_t>(k_max_input_batch_size));
    // Special case for 2 inputs left.
    if (batch == 2) {
      BinaryElementwisePreparation prepare;
      ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(&output, &inputs[input_count - 1].get(), &output, &prepare));
      Impl_General<CudaT, VariadicElementwiseOpTag>(
          stream, prepare.output_rank_or_simple_broadcast, &prepare.lhs_padded_strides,
          reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()), &prepare.rhs_padded_strides,
          reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()), &prepare.fdm_output_strides,
          prepare.fdm_H, prepare.fdm_C, reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
          prepare.output_tensor->Shape().Size());

      // Must be the last.
      break;
    }

    InputBatchArray<CudaT> left_input_data_batch{static_cast<int32_t>(batch)};
    left_input_data_batch[0] = reinterpret_cast<const CudaT*>(output.Data<T>());
    for (size_t i = 1; i < batch; ++i) {
      left_input_data_batch[static_cast<int32_t>(i)] =
          reinterpret_cast<const CudaT*>(inputs[index].get().Data<T>());
      index++;
    }

    Impl_NoBroadcastInputBatch<CudaT, VariadicElementwiseOpTag>(stream, left_input_data_batch, output_data,
                                                                output.Shape().Size());
  }

  return Status::OK();
}

// special case for 2 tensors to avoid memset zero
template <typename VariadicElementwiseOpTag, typename... SupportedElementTypes>
template <typename T>
Status VariadicElementwiseOp<VariadicElementwiseOpTag, SupportedElementTypes...>::
    BinaryImplDispatchTarget<T>::operator()(cudaStream_t stream, const Tensor& lhs, const Tensor& rhs, Tensor& output) const {
  using CudaT = typename ToCudaType<T>::MappedType;

  BinaryElementwisePreparation prepare;
  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(&lhs, &rhs, &output, &prepare));

  Impl_General<CudaT, VariadicElementwiseOpTag>(
      stream,
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()),
      &prepare.rhs_padded_strides,
      reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()),
      &prepare.fdm_output_strides,
      prepare.fdm_H,
      prepare.fdm_C,
      reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
      prepare.output_tensor->Shape().Size());

  return Status::OK();
}

// for more than 2 inputs, we need to accumulate into output tensor, as the shape from input0 + input1 might be different from output shape
template <typename VariadicElementwiseOpTag, typename... SupportedElementTypes>
template <typename T>
Status
VariadicElementwiseOp<VariadicElementwiseOpTag, SupportedElementTypes...>::GeneralImplDispatchTarget<T>::operator()(
    cudaStream_t stream, const InputTensorVector& inputs, Tensor& output) const {
  assert(inputs.size() > 1);

  using CudaT = typename ToCudaType<T>::MappedType;

  // If there is any input having the same shape with output, we don't need the memset.
  size_t index_of_same_shape = 0;
  for (; index_of_same_shape < inputs.size(); index_of_same_shape++) {
    if (inputs[index_of_same_shape].get().Shape() == output.Shape()) {
      break;
    }
  }

  BinaryElementwisePreparation prepare;

  // No input has same shape of output, memset the output, and add the 1st input as initialization.
  if (index_of_same_shape == inputs.size()) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output.MutableDataRaw(), 0, output.SizeInBytes(), stream));
    ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(&output, &inputs[0].get(), &output, &prepare));
    Impl_Add(stream, prepare.output_rank_or_simple_broadcast, &prepare.lhs_padded_strides,
             reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()), &prepare.rhs_padded_strides,
             reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()), &prepare.fdm_output_strides,
             prepare.fdm_H, prepare.fdm_C, reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
             prepare.output_tensor->Shape().Size());
  } else {
    // First operation is between input[0] and input[index_of_same_shape] if index_of_same_shape is not 0.
    size_t index = index_of_same_shape == 0 ? 1 : 0;
    ORT_RETURN_IF_ERROR(
        BinaryElementwiseBroadcastPrepare(&inputs[index_of_same_shape].get(), &inputs[index].get(), &output, &prepare));
    Impl_General<CudaT, VariadicElementwiseOpTag>(
        stream, prepare.output_rank_or_simple_broadcast, &prepare.lhs_padded_strides,
        reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()), &prepare.rhs_padded_strides,
        reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()), &prepare.fdm_output_strides,
        prepare.fdm_H, prepare.fdm_C, reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
        prepare.output_tensor->Shape().Size());
  }

  for (size_t index = 1; index < inputs.size(); index++) {
    // If index_of_same_shape is 0, we already handle the 1st and 2nd inputs.
    if (index == index_of_same_shape || (index_of_same_shape == 0 && index == 1)) {
      continue;
    }

    ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(&output, &inputs[index].get(), &output, &prepare));
    Impl_General<CudaT, VariadicElementwiseOpTag>(
        stream, prepare.output_rank_or_simple_broadcast, &prepare.lhs_padded_strides,
        reinterpret_cast<const CudaT*>(prepare.lhs_tensor->Data<T>()), &prepare.rhs_padded_strides,
        reinterpret_cast<const CudaT*>(prepare.rhs_tensor->Data<T>()), &prepare.fdm_output_strides,
        prepare.fdm_H, prepare.fdm_C, reinterpret_cast<CudaT*>(prepare.output_tensor->MutableData<T>()),
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
          cudaMemcpyDeviceToDevice, Stream(context)));
    }

    return Status::OK();
  }

  const auto element_type = first_input_tensor.GetElementType();
  utils::MLTypeCallDispatcher<SupportedElementTypes...> dispatcher(element_type);

  // Special case for no broadcasting.
  if (std::all_of(input_tensors.begin() + 1, input_tensors.end(),
                  [&first_input_tensor](InputTensorVector::value_type t) {
                    return first_input_tensor.Shape() == t.get().Shape();
                  })) {
    auto& output_tensor = context->RequiredOutput(0, first_input_tensor.Shape());

    // special case for no broadcasting and 2 inputs
    if (input_count == 2) {
      return dispatcher.template InvokeRet<Status, BinaryImplDispatchTarget>(Stream(context), input_tensors[0],
                                                                             input_tensors[1], output_tensor);
    }

    return dispatcher.template InvokeRet<Status, NoBroadcastBatchImplDispatchTarget>(Stream(context), input_tensors,
                                                                                     output_tensor);
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
    return dispatcher.template InvokeRet<Status, BinaryImplDispatchTarget>(
        Stream(context), input_tensors[0], input_tensors[1], output_tensor);
  }

  // general case for more than 2 inputs
  return dispatcher.template InvokeRet<Status, GeneralImplDispatchTarget>(
      Stream(context), input_tensors, output_tensor);
}

namespace {

using SumOp = VariadicElementwiseOp<variadic_elementwise_ops::Sum, MLFloat16, float, double, BFloat16>;

using MinOp = VariadicElementwiseOp<variadic_elementwise_ops::Min, uint32_t, uint64_t, int32_t, int64_t, MLFloat16,
                                    float, double, BFloat16>;

using MaxOp = VariadicElementwiseOp<variadic_elementwise_ops::Max, uint32_t, uint64_t, int32_t, int64_t, MLFloat16,
                                    float, double, BFloat16>;
}  // namespace

// kernel registration

#define REGISTER_KERNEL(name, impl_class, version, datatypes)                                                        \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, version, kCudaExecutionProvider,                                        \
                          (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<datatypes>()), \
                          impl_class)

#define REGISTER_VERSIONED_KERNEL(name, impl_class, start_version, end_version, datatypes) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                       \
      name, kOnnxDomain, start_version, end_version, kCudaExecutionProvider,               \
      (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<datatypes>()), impl_class)

#define UZILHFD_TYPES uint32_t, uint64_t, int32_t, int64_t, MLFloat16, float, double, BFloat16
#define HFD_TYPES MLFloat16, float, double, BFloat16

REGISTER_KERNEL(Sum, SumOp, 13, HFD_TYPES)
REGISTER_VERSIONED_KERNEL(Sum, SumOp, 8, 12, HFD_TYPES)
REGISTER_VERSIONED_KERNEL(Sum, SumOp, 6, 7, HFD_TYPES)

REGISTER_KERNEL(Min, MinOp, 13, UZILHFD_TYPES)
REGISTER_VERSIONED_KERNEL(Min, MinOp, 12, 12, UZILHFD_TYPES)
REGISTER_VERSIONED_KERNEL(Min, MinOp, 6, 11, HFD_TYPES)

REGISTER_KERNEL(Max, MaxOp, 13, UZILHFD_TYPES)
REGISTER_VERSIONED_KERNEL(Max, MaxOp, 12, 12, UZILHFD_TYPES)
REGISTER_VERSIONED_KERNEL(Max, MaxOp, 6, 11, HFD_TYPES)

#undef HFD_TYPES
#undef UZILHFD_TYPES
#undef REGISTER_VERSIONED_KERNEL
#undef REGISTER_KERNEL

}  // namespace cuda
}  // namespace onnxruntime
