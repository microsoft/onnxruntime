// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activations.h"
#include "core/framework/op_kernel.h"
#include "torch/torch.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_ACTIVATION_KERNEL(x, ver, domain, T)            \
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

#define UNARY_ACTIVATION_COMPUTE(x, T)                                                                     \
  template <>                                                                                              \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                           \
    UnaryElementwisePreparation p;                                                                         \
    UnaryElementwise::Prepare(context, &p);                                                                \
    Ctx##x func_ctx = MakeFuncCtx();                                                                       \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                          \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(p.input_tensor->template Data<T>()),   \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(p.output_tensor->template MutableData<T>()), \
        &func_ctx, p.output_tensor->Shape().Size());                                                       \
                                                                                                           \
    return Status::OK();                                                                                   \
  }

#define UNARY_ACTIVATION_OP_TYPED(name, ver, domain, T) \
  REGISTER_ACTIVATION_KERNEL(name, ver, domain, T)      \
  UNARY_ACTIVATION_COMPUTE(name, T)

#define UNARY_ACTIVATION_OP_HFD(name, ver, domain)        \
  UNARY_ACTIVATION_OP_TYPED(name, ver, domain, MLFloat16) \
  UNARY_ACTIVATION_OP_TYPED(name, ver, domain, float)     \
  UNARY_ACTIVATION_OP_TYPED(name, ver, domain, double)

UNARY_ACTIVATION_OP_HFD(Affine, 1, kOnnxDomain);
UNARY_ACTIVATION_OP_HFD(ParametricSoftplus, 1, kOnnxDomain);
UNARY_ACTIVATION_OP_HFD(ScaledTanh, 1, kOnnxDomain);
//UNARY_ACTIVATION_OP_HFD(Gelu, 1, kMSDomain);

REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, kOnnxDomain, MLFloat16)
REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, kOnnxDomain, float)
REGISTER_ACTIVATION_KERNEL(ThresholdedRelu, 1, kOnnxDomain, double)

#define REGISTER_GELU_KERNEL(ver, domain, T)            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      Gelu,                                                         \
      domain,                                                    \
      ver,                                                       \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      Gelu<T>);

// Type mapping for MLFloat16 to half
template <typename T>
at::ScalarType get_torch_type();

template <>
at::ScalarType get_torch_type<MLFloat16>() {
  return at::kHalf;
}

template <>
at::ScalarType get_torch_type<float>() {
  return at::kFloat;
}

template <>
at::ScalarType get_torch_type<double>() {
  return at::kDouble;
}

template <>
at::ScalarType get_torch_type<int8_t>() {
  return at::kChar;
}

template <>
at::ScalarType get_torch_type<int16_t>() {
  return at::kShort;
}

template <>
at::ScalarType get_torch_type<int32_t>() {
  return at::kInt;
}

template <>
at::ScalarType get_torch_type<int64_t>() {
  return at::kLong;
}

template <typename T>
Status Gelu<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  auto input_tensor = context->Input<Tensor>(0);
  auto output_tensor = context->Output(0, input_tensor->Shape());


  int torch_device = 0;
  cudaGetDevice(&torch_device);
  auto torch_tensor_options = torch::TensorOptions().dtype(get_torch_type<T>()).device(torch::kCUDA, torch_device);
  torch::Tensor torch_input = torch::from_blob(const_cast<T*>(input_tensor->template Data<T>()), input_tensor->Shape().GetDims(), torch_tensor_options);
  torch::Tensor torch_output = at::gelu(torch_input);

  CudaT* output_data = reinterpret_cast<CudaT*>(output_tensor->template MutableData<T>());
  cudaMemcpy(output_data, torch_output.data_ptr(), output_tensor->Shape().Size() * sizeof(T), cudaMemcpyDeviceToDevice);

  return Status::OK();
}

REGISTER_GELU_KERNEL(1, kMSDomain, MLFloat16);
REGISTER_GELU_KERNEL(1, kMSDomain, float);
REGISTER_GELU_KERNEL(1, kMSDomain, double);

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
