// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class SiluAndMul final : public ::onnxruntime::cuda::CudaKernel {
 public:
  SiluAndMul(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

#define REGISTER_KERNEL_TYPED(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      SiluAndMul,                                                \
      kMSDomain,                                                 \
      1,                                                         \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      (*KernelDefBuilder::Create())                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SiluAndMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
SiluAndMul<T>::SiluAndMul(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

extern void DumpTensor(cudaStream_t cuda_stream, const Tensor* tensor, const std::string& name);

template <typename T>
void LaunchSiluMulKernel(
    cudaStream_t stream,
    T* out,  // [num_tokens, d]
    const T* input,
    const int64_t* input_shape);
template <typename T>
Status SiluAndMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);

  auto output_shape = input->Shape();
  output_shape[output_shape.NumDimensions() - 1] /= 2;
  Tensor* output = ctx->Output(0, output_shape);
  typedef typename ::onnxruntime::cuda::ToCudaType<T>::MappedType CudaT;

  auto input_shape = input->Shape();
  if (input_shape.NumDimensions() > 2) {
    input_shape[0] *= input_shape[1];
    input_shape[1] = input_shape[2];
  }
  LaunchSiluMulKernel<CudaT>(
      Stream(ctx),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      input_shape.GetDims().data());
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
