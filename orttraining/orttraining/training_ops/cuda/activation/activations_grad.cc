// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/activations_grad.h"
#include "core/framework/op_kernel.h"
#include "torch/torch.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_ACTIVATION_GRAD_KERNEL(x, ver, domain, T)       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      x,                                                         \
      domain,                                                    \
      ver,                                                       \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      x<T>);

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                         \
  template <>                                                                                                    \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    BinaryElementwisePreparation prepare;                                                                        \
    Prepare(context, &prepare);                                                                                  \
    CudaAsyncBuffer<Ctx##x> func_ctx(this, MakeFuncCtx(), 1);                                                    \
    if (!std::is_same<CtxNull, Ctx##x>::value) ORT_RETURN_IF_ERROR(func_ctx.CopyToGpu());                        \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                                \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),     \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),     \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()), \
        func_ctx.GpuPtr(), prepare.output_tensor->Shape().Size());                                               \
    return Status::OK();                                                                                         \
  }

#define ACTIVATION_GRAD_OP_TYPED(name, ver, domain, T)  \
  REGISTER_ACTIVATION_GRAD_KERNEL(name, ver, domain, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

#define ACTIVATION_GRAD_OP_HFD(name, ver, domain)        \
  ACTIVATION_GRAD_OP_TYPED(name, ver, domain, MLFloat16) \
  ACTIVATION_GRAD_OP_TYPED(name, ver, domain, float)     \
  ACTIVATION_GRAD_OP_TYPED(name, ver, domain, double)

ACTIVATION_GRAD_OP_HFD(FastGeluGrad, 1, kMSDomain);

// Type mapping for MLFloat16 to half
template <typename T>
at::ScalarType get_torch_type();

template <typename T>
Status GeluGrad<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  int torch_device = 0;
  cudaGetDevice(&torch_device);
  auto torch_tensor_options = torch::TensorOptions().dtype(get_torch_type<T>()).device(torch::kCUDA, torch_device);
  auto left_tensor = context->Input<Tensor>(0);
  auto right_tensor = context->Input<Tensor>(1);

  torch::Tensor torch_left = torch::from_blob(const_cast<T*>(left_tensor->template Data<T>()), left_tensor->Shape().GetDims(), torch_tensor_options);
  torch::Tensor torch_right = torch::from_blob(const_cast<T*>(right_tensor->template Data<T>()), right_tensor->Shape().GetDims(), torch_tensor_options);

  auto output_tensor = context->Output(0, left_tensor->Shape());

  torch::Tensor torch_output = at::gelu_backward(torch_left, torch_right);

  CudaT* output_data = reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>());
  cudaMemcpy(output_data, torch_output.data_ptr(), output_tensor->Shape().Size() * sizeof(T), cudaMemcpyDeviceToDevice);

  return Status::OK();
}

REGISTER_ACTIVATION_GRAD_KERNEL(GeluGrad, 1, kMSDomain, float);
REGISTER_ACTIVATION_GRAD_KERNEL(GeluGrad, 1, kMSDomain, MLFloat16);
REGISTER_ACTIVATION_GRAD_KERNEL(GeluGrad, 1, kMSDomain, double);

}  //namespace cuda
}  // namespace onnxruntime
