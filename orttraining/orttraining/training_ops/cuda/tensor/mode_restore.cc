// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/mode_restore.h"
#include "orttraining/training_ops/cuda/tensor/mode_restore_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ModeRestore,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
        .TypeConstraint("T_MASK", DataTypeImpl::GetTensorType<BitmaskElementType>())
        .TypeConstraint("T_INT", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T_CFW", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    ModeRestore);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct RestoreFromMaskFunctor {
  void operator()(const cudaDeviceProp& prop,
                  cudaStream_t stream,
                  const int64_t total_element_count,
                  const float zero_point_value,
                  const Tensor& input_tensor,
                  const int* output_idx_to_input_idx_map_buffer,
                  Tensor& output_tensor) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input_tensor.Data<T>());
    RestoreFromMaskImpl<CudaT>(prop, stream, total_element_count,
                               zero_point_value,
                               input_data,
                               output_idx_to_input_idx_map_buffer,
                               reinterpret_cast<CudaT*>(output_tensor.MutableData<T>()));
  }
};

}  // namespace

Status ModeRestore::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* mask_input_tensor = context->Input<Tensor>(1);
  const Tensor* output_shape_tensor = context->Input<Tensor>(2);  // Parse the 1-D shape tensor.
  TensorShape output_shape(output_shape_tensor->Data<int64_t>(), output_shape_tensor->Shape().Size());
  const int64_t total_element_count = output_shape.Size();

  IAllocatorUniquePtr<int> restored_output_mask = GetScratchBuffer<int>(total_element_count,
                                                                        context->GetComputeStream());
  FillOutputFromMaskImpl(Stream(context),
                         total_element_count,
                         static_cast<const BitmaskElementType*>(mask_input_tensor->DataRaw()),
                         restored_output_mask.get());

  size_t temp_storage_bytes = 0;
  GetZeroPointRestoreTempStorageBytesImpl(Stream(context),
                                          temp_storage_bytes,
                                          static_cast<int>(total_element_count));

  IAllocatorUniquePtr<int> output_idx_to_input_idx_map_buffer = GetScratchBuffer<int>(total_element_count,
                                                                                      context->GetComputeStream());
  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(temp_storage_bytes, context->GetComputeStream());
  CalculateInputOffsetForEachOutputImpl(Stream(context),
                                        workspace.get(),
                                        temp_storage_bytes,
                                        restored_output_mask.get(),
                                        output_idx_to_input_idx_map_buffer.get(),
                                        static_cast<int>(total_element_count));

  Tensor* output_tensor = context->Output(0, output_shape);
  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(input_tensor->GetElementType());
  t_disp.Invoke<RestoreFromMaskFunctor>(GetDeviceProp(),
                                        Stream(context),
                                        total_element_count,
                                        mode_,
                                        *input_tensor,
                                        output_idx_to_input_idx_map_buffer.get(),
                                        *output_tensor);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
