// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class DebugStep final : public ::onnxruntime::cuda::CudaKernel {
 public:
  DebugStep(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      DebugStep,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .Alias(0, 0),                                           \
      DebugStep<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
DebugStep<T>::DebugStep(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

extern void DumpTensor(cudaStream_t cuda_stream, const Tensor* tensor, const std::string& name);

template <typename T>
Status DebugStep<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  Tensor* output = ctx->Output(0, input->Shape());
  if(output->DataRaw() != input->DataRaw()) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output->MutableDataRaw(), input->DataRaw(), input->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(ctx)));
  }
  DumpTensor(Stream(ctx), input, "debug_input");
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
