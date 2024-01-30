// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/scatter_elements.h"

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/tensor/gather_elements.h"
#include "core/providers/cuda/tensor/gather_elements_impl.h"
#include "core/providers/cuda/tensor/scatter_elements_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Scatter, kOnnxDomain, 9, 10, kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .TypeConstraint("Tind",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                              DataTypeImpl::GetTensorType<int64_t>()}),
                                  ScatterElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterElements, kOnnxDomain, 11, 12, kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .TypeConstraint("Tind",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                              DataTypeImpl::GetTensorType<int64_t>()}),
                                  ScatterElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterElements, kOnnxDomain, 13, 15, kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .TypeConstraint("Tind",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                              DataTypeImpl::GetTensorType<int64_t>()}),
                                  ScatterElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterElements, kOnnxDomain, 16, 17, kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .TypeConstraint("Tind",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                              DataTypeImpl::GetTensorType<int64_t>()}),
                                  ScatterElements);

ONNX_OPERATOR_KERNEL_EX(ScatterElements, kOnnxDomain, 18, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                            DataTypeImpl::GetTensorType<int64_t>()}),
                        ScatterElements);

#define CASE_SCATTER_ELEMENTS_IMPL(type)                                                                         \
  case sizeof(type): {                                                                                           \
    const type* indices_data = reinterpret_cast<const type*>(indices_data_raw);                                  \
    ORT_RETURN_IF_ERROR(ScatterElementsImpl(stream, input_data, indices_data, updates_data, output_data, args)); \
  } break

template <typename T>
struct ScatterElements::ComputeImpl {
  Status operator()(cudaStream_t stream, const void* input_data_raw, const void* updates_data_raw,
                    const void* indices_data_raw, void* output_data_raw, const size_t index_element_size,
                    const GatherScatterElementsArgs& args) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input_data_raw);
    const CudaT* updates_data = reinterpret_cast<const CudaT*>(updates_data_raw);
    CudaT* output_data = reinterpret_cast<CudaT*>(output_data_raw);
    switch (index_element_size) {
      CASE_SCATTER_ELEMENTS_IMPL(int32_t);
      CASE_SCATTER_ELEMENTS_IMPL(int64_t);
      // should not reach here as we validate if the all relevant types are supported in the Compute method
      default:
        ORT_THROW("Unsupported indices element size by the ScatterElements CUDA kernel");
    }

    return Status::OK();
  }
};

#undef CASE_SCATTER_ELEMENTS_IMPL

Status ScatterElements::ComputeInternal(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape();
  const int64_t input_size = input_shape.Size();
  const int64_t input_rank = static_cast<int64_t>(input_shape.NumDimensions());
  const int64_t axis = HandleNegativeAxis(axis_, input_rank);

  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* updates_tensor = context->Input<Tensor>(2);

  if (input_tensor->DataType() != updates_tensor->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  const auto& indices_shape = indices_tensor->Shape();

  auto indices_dims = indices_shape.GetDims();
  auto updates_dims = updates_tensor->Shape().GetDims();
  if (indices_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices and updates must have the same rank");
  }

  for (size_t i = 0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices vs updates dimensions differs at position=", i,
                             " ", indices_dims[i], " vs ", updates_dims[i]);
    }
  }

  // Validate input shapes and ranks (invoke the static method in the CPU GatherElements kernel that hosts the shared
  // checks)
  ORT_RETURN_IF_ERROR(onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis));

  auto* output_tensor = context->Output(0, input_shape);
  if (input_size == 0) return Status::OK();

  GatherScatterElementsArgs args;
  args.input_size = input_size;
  args.indices_size = indices_shape.Size();
  TensorShapeVector input_shape_vec = input_shape.AsShapeVector();
  TensorShapeVector indices_shape_vec = indices_shape.AsShapeVector();
  CoalesceDimensions(input_shape_vec, indices_shape_vec, nullptr, axis, args);

  if (reduction_ == "none") {
    args.operation = GatherScatterElementsArgs::Operation::NONE;
  } else if (reduction_ == "add") {
    args.operation = GatherScatterElementsArgs::Operation::ADD;
  } else if (reduction_ == "mul") {
    args.operation = GatherScatterElementsArgs::Operation::MUL;
  } else if (reduction_ == "min") {
    args.operation = GatherScatterElementsArgs::Operation::MIN;
  } else if (reduction_ == "max") {
    args.operation = GatherScatterElementsArgs::Operation::MAX;
  } else {
    ORT_THROW("Unsupported reduction type");
  }

  // Use element size instead of concrete types so we can specialize less template functions to reduce binary size.
  int dtype = GetElementType(input_tensor->DataType()->Size());
  if (dtype == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
    ORT_THROW("Unsupported element size by the ScatterElements CUDA kernel");
  }

  utils::MLTypeCallDispatcher<int8_t, MLFloat16, float, double> t_disp(dtype);
  return t_disp.InvokeRet<Status, ComputeImpl>(Stream(context), input_tensor->DataRaw(), updates_tensor->DataRaw(),
                                               indices_tensor->DataRaw(), output_tensor->MutableDataRaw(),
                                               indices_tensor->DataType()->Size(), args);
}

}  // namespace cuda
}  // namespace onnxruntime
