// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "isfinite.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_ISFINITE_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      IsFinite,                                                       \
      kOnnxDomain,                                                    \
      9,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()), \
      IsFiniteOp<T>);

template <typename TSrc>
Status IsFiniteOp<TSrc>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<TSrc>::MappedType CudaTSrc;
  const Tensor& input = *context->Input<Tensor>(0);
  Tensor& output = *context->Output(0, input.Shape());
  IsFinite(
      reinterpret_cast<const CudaTSrc*>(input.Data<TSrc>()),
      output.MutableData<bool>(), input.Shape().Size());

  return Status::OK();
}

REGISTER_ISFINITE_KERNEL_TYPED(MLFloat16)
REGISTER_ISFINITE_KERNEL_TYPED(float)
REGISTER_ISFINITE_KERNEL_TYPED(double)

#define REGISTER_ISALLFINITE_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      IsAllFinite,                                                   \
      kOnnxDomain,                                                   \
      9,                                                             \
      T,                                                             \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .OutputMemoryType<OrtMemTypeCPUOutput>(0)                   \
          .TypeConstraint("V", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()), \
      IsAllFiniteOp<T>);

template <typename TSrc>
Status IsAllFiniteOp<TSrc>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<TSrc>::MappedType TSrcCuda;

  // Get Input tensor count.
  auto total_tensor_count = Node().InputArgCount().front();
  auto kernel_launch_count = (total_tensor_count + MAX_TENSOR_COUNT - 1) / MAX_TENSOR_COUNT;

  Tensor& output = *context->Output(0, {});
  // Allocate GPU memory to capture the result computed by GPU kernel. 
  // The GPU result will be copied to the output which locates on CPU memory.
  IAllocatorUniquePtr<bool> deviceOutput = GetScratchBuffer<bool>(1);
  CUDA_RETURN_IF_ERROR(cudaMemset(deviceOutput.get(), int(true), 1));

  for (int launch_index = 0; launch_index < kernel_launch_count; ++launch_index) {
    ChunkGroup<TSrcCuda> chunks;

    int chunk_size = 1024;
    // One kernel launch processes at most MAX_TENSOR_COUNT tensors, so we prepare
    // at most MAX_TENSOR_COUNT tensors for the launch in this iteration.
    const int tensor_start_index = launch_index * MAX_TENSOR_COUNT;
    for (int tensor_index = 0; tensor_index < MAX_TENSOR_COUNT; ++tensor_index) {
      if (tensor_start_index + tensor_index >= total_tensor_count) {
        break;
      }
      const Tensor& input = *context->Input<Tensor>(tensor_start_index + tensor_index);
      const int input_size = static_cast<int>(input.Shape().Size());
      chunks.tensor_ptrs[tensor_index] = reinterpret_cast<const TSrcCuda*>(input.Data<TSrc>());
      chunks.tensor_sizes[tensor_index] = input_size;
      chunk_size = std::max(chunk_size, input_size / MAX_BLOCK_COUNT);
    }

    int block_index = 0;
    for (int tensor_index = 0; tensor_index < MAX_TENSOR_COUNT; ++tensor_index) {
      if (tensor_start_index + tensor_index >= total_tensor_count) {
        break;
      }
      const Tensor& input = *context->Input<Tensor>(tensor_start_index + tensor_index);
      const int chunk_count = (static_cast<int>(input.Shape().Size()) + chunk_size - 1) / chunk_size;
      for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        chunks.block_index_to_tensor_index[block_index] = tensor_index;
        chunks.block_index_to_chunk_start_index[block_index] = chunk_index * chunk_size;

        ++block_index;

        // Now we have block_index chunks stored in "chunks", so we update the count.
        chunks.chunk_count = block_index;
        chunks.chunk_size = chunk_size;
        if (block_index == MAX_BLOCK_COUNT) {
          IsAllFinite(chunks, deviceOutput.get());
          block_index = 0;
        }
      }
    }

    if (block_index != 0) {
      IsAllFinite(chunks, deviceOutput.get());
    }
  }

  // Copy GPU result to CPU memory.
  // From this operator's schema, it's output is in CPU memory.
  CUDA_RETURN_IF_ERROR(
    cudaMemcpy(
      output.MutableData<bool>(),
      deviceOutput.get(),
      sizeof(bool),
      cudaMemcpyDeviceToHost));

  return Status::OK();
}

REGISTER_ISALLFINITE_KERNEL_TYPED(MLFloat16)
REGISTER_ISALLFINITE_KERNEL_TYPED(float)
REGISTER_ISALLFINITE_KERNEL_TYPED(double)

}  // namespace cuda
}  // namespace onnxruntime
