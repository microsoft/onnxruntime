// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/gather_elements_grad.h"

#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cuda/tensor/gather_elements.h"
#include "core/providers/cuda/tensor/gather_elements_impl.h"
#include "orttraining/training_ops/cuda/tensor/gather_elements_grad_impl.h"

namespace onnxruntime {
namespace cuda {

#ifdef ENABLE_STRIDED_TENSORS
#define CREATE_GATHER_ELEMENTS_GRAD_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(2)
#else
#define CREATE_GATHER_ELEMENTS_GRAD_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

ONNX_OPERATOR_KERNEL_EX(GatherElementsGrad, kMSDomain, 1, kCudaExecutionProvider,
                        CREATE_GATHER_ELEMENTS_GRAD_KERNEL_DEF
                            .InputMemoryType(OrtMemTypeCPUInput, 1)  // 'GatherElements' data shape needs to be on CPU
                            .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
                            .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                            .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                            DataTypeImpl::GetTensorType<int64_t>()}),
                        GatherElementsGrad);

#undef CREATE_GATHER_ELEMENTS_GRAD_KERNEL_DEF

#define CASE_GATHER_ELEMENTS_GRAD_IMPL(type)                                                       \
  case sizeof(type): {                                                                             \
    const type* indices_data = reinterpret_cast<const type*>(indices_data_raw);                    \
    ORT_RETURN_IF_ERROR(GatherElementsGradNonDeterministicImpl(stream, indices_data, updates_data, \
                                                               output_data, args));                \
  } break

template <typename T>
struct GatherElementsGrad::ComputeImpl {
  Status operator()(cudaStream_t stream, const void* dY_data_raw, const void* indices_data_raw, void* dX_data_raw,
                    const size_t index_element_size, const GatherScatterElementsArgs& args) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* updates_data = reinterpret_cast<const CudaT*>(dY_data_raw);
    CudaT* output_data = reinterpret_cast<CudaT*>(dX_data_raw);
    switch (index_element_size) {
      CASE_GATHER_ELEMENTS_GRAD_IMPL(int32_t);
      CASE_GATHER_ELEMENTS_GRAD_IMPL(int64_t);
      // should not reach here as we validate if the all relevant types are supported in the Compute method
      default:
        ORT_THROW("Unsupported indices element size by the GatherElementsGrad CUDA kernel");
    }

    return Status::OK();
  }
};

#undef CASE_GATHER_ELEMENTS_GRAD_IMPL

Status GatherElementsGrad::ComputeInternal(OpKernelContext* context) const {
  const auto* dY = context->Input<Tensor>(0);
  const Tensor* shape = context->Input<Tensor>(1);
  const TensorShape data_shape(shape->template Data<int64_t>(), shape->Shape().Size());
  const int64_t data_rank = static_cast<int64_t>(data_shape.NumDimensions());

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, data_rank));

  const auto* indices_tensor = context->Input<Tensor>(2);
  const auto& indices_shape = indices_tensor->Shape();
  if (indices_shape != dY->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices and dY must have the same shape.");
  }

  // Validate data shapes and ranks (invoke the static method in the CPU GatherElements kernel that hosts the shared
  // checks)
  ORT_RETURN_IF_ERROR(onnxruntime::GatherElements::ValidateInputShapes(data_shape, indices_shape, axis));

  Tensor* dX = context->Output(0, data_shape);
  if (data_shape.Size() == 0) return Status::OK();

  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(dX->MutableDataRaw(), 0, dX->SizeInBytes(), Stream(context)));

  GatherScatterElementsArgs args;
  args.indices_size = indices_shape.Size();
  TensorShapeVector data_shape_vec = data_shape.AsShapeVector();
  TensorShapeVector indices_shape_vec = indices_shape.AsShapeVector();
  TensorShapeVector* p_indices_strides_vec = nullptr;
  TensorShapeVector indices_strids_vec;
#ifdef ENABLE_STRIDED_TENSORS
  if (!indices_tensor->IsContiguous()) {
    indices_strids_vec = ToShapeVector(indices_tensor->Strides());
    p_indices_strides_vec = &indices_strids_vec;
  }
#endif
  CoalesceDimensions(data_shape_vec, indices_shape_vec, p_indices_strides_vec, axis, args);

  // Use element size instead of concrete types so we can specialize less template functions to reduce binary size.
  int dtype = GetElementType(dY->DataType()->Size());
  // GatherElementsGrad supports half, float and double only for now, it's element size will not but INT8.
  if (dtype == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED || dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    ORT_THROW("Unsupported element size by the GatherElementsGrad CUDA kernel");
  }

  if (context->GetUseDeterministicCompute()) {
    static std::once_flag log_warning;
    std::call_once(log_warning, []() {
      LOGS_DEFAULT(WARNING) << "GatherElementsGrad has no deterministic GPU kernel, its outputs may still be nondeterministic.";
    });
  }

  utils::MLTypeCallDispatcher<MLFloat16, float, double> t_disp(dtype);
  return t_disp.InvokeRet<Status, ComputeImpl>(Stream(context), dY->DataRaw(), indices_tensor->DataRaw(),
                                               dX->MutableDataRaw(),
                                               indices_tensor->DataType()->Size(), args);
}

}  // namespace cuda
}  // namespace onnxruntime
