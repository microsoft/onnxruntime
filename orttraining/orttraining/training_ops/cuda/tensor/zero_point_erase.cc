// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/zero_point_erase.h"
#include "orttraining/training_ops/cuda/tensor/zero_point_erase_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ZeroPointErase,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
        .TypeConstraint("T_MASK", DataTypeImpl::GetTensorType<BitmaskElementType>())
        .TypeConstraint("T_INT", DataTypeImpl::GetTensorType<int64_t>())
        .OutputMemoryType(OrtMemTypeCPUOutput, 2),
    ZeroPointErase);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct GetTempStorageBytesFunctor {
  void operator()(Stream* stream,
                  const int64_t total_element_count,
                  const float zero_point_value,
                  size_t& temp_storage_bytes) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    GetTempStorageBytesImpl<CudaT>(
        static_cast<cudaStream_t>(stream->GetHandle()),
        temp_storage_bytes,
        zero_point_value,
        static_cast<int>(total_element_count));
  }
};

template <typename T>
struct CopyOnConditionFunctor {
  void operator()(const CudaKernel* cuda_kernel,
                  OpKernelContext* context,
                  void* d_temp_storage,
                  const int64_t total_element_count,
                  const float zero_point_value,
                  const Tensor& input_tensor,
                  size_t temp_storage_bytes) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input_tensor.Data<T>());
    IAllocatorUniquePtr<CudaT> temp_buffer = cuda_kernel->GetScratchBuffer<CudaT>(total_element_count,
                                                                                  context->GetComputeStream());
    std::cout << "total_element_count: " << total_element_count << ", temp_storage_bytes: " << temp_storage_bytes
              << std::endl;
    IAllocatorUniquePtr<int> d_num_selected_out = cuda_kernel->GetScratchBuffer<int>(1,
                                                                                     context->GetComputeStream());

    CopyOnConditionImpl<CudaT>(cuda_kernel->Stream(context),
                               d_temp_storage,
                               temp_storage_bytes,
                               input_data,
                               temp_buffer.get(),
                               *d_num_selected_out.get(),
                               zero_point_value,
                               static_cast<int>(total_element_count));
    // cudaStreamSynchronize is needed since the value of d_num_selected_out will be used by host after this function.
    // CUDA_CALL_THROW(cudaStreamSynchronize(cuda_kernel->Stream(context)));
    int d_num_selected_host = 0;
    CUDA_CALL_THROW(cudaMemcpyAsync(&d_num_selected_host, d_num_selected_out.get(),
                                    sizeof(int), cudaMemcpyDeviceToHost, cuda_kernel->Stream(context)));

    Tensor* output_tensor = context->Output(0, {d_num_selected_host});
    CUDA_CALL_THROW(cudaMemcpyAsync(output_tensor->MutableDataRaw(),
                                    temp_buffer.get(),
                                    d_num_selected_host * sizeof(CudaT),
                                    cudaMemcpyDeviceToDevice,
                                    cuda_kernel->Stream(context)));
  }
};

template <typename T>
struct SetMaskOutputFunctor {
  void operator()(const cudaDeviceProp& prop,
                  cudaStream_t stream,
                  const int64_t total_element_count,
                  const int64_t mask_element_count,
                  const float zero_point_value,
                  const Tensor& X,
                  void* mask_data) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.Data<T>());
    SetMaskOutputImpl<CudaT>(prop, stream, total_element_count,
                             mask_element_count, zero_point_value,
                             X_data, mask_data);
  }
};

}  // namespace

Status ZeroPointErase::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(input_tensor, "input_tensor is not available.");
  const TensorShape& input_shape = input_tensor->Shape();
  const int64_t total_element_count = input_shape.Size();

  // Set input shape output tensor.
  int64_t rank = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
  Tensor* input_shape_tensor = context->Output(2, {rank});
  input_shape.CopyDims(input_shape_tensor->MutableData<int64_t>(), static_cast<size_t>(rank));

  int64_t mask_element_count = (total_element_count + kNumBitsPerBitmaskElement - 1) / kNumBitsPerBitmaskElement;

  size_t temp_storage_bytes = 0;
  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(input_tensor->GetElementType());
  t_disp.Invoke<GetTempStorageBytesFunctor>(context->GetComputeStream(),
                                            total_element_count,
                                            default_zero_point_value_,
                                            temp_storage_bytes);

  IAllocatorUniquePtr<void> workspace = GetScratchBuffer<void>(temp_storage_bytes, context->GetComputeStream());
  t_disp.Invoke<CopyOnConditionFunctor>(this,
                                        context,
                                        workspace.get(),
                                        total_element_count,
                                        default_zero_point_value_,
                                        *input_tensor,
                                        temp_storage_bytes);

  Tensor* mask_output_tensor = context->Output(1, {mask_element_count});
  t_disp.Invoke<SetMaskOutputFunctor>(GetDeviceProp(), Stream(context), total_element_count, mask_element_count,
                                      default_zero_point_value_, *input_tensor,
                                      mask_output_tensor->MutableDataRaw());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
