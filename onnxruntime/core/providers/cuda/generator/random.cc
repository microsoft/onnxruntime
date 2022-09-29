// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/generator/random.h"
#include "core/providers/cuda/generator/random_impl.h"

namespace onnxruntime {
namespace cuda {

using namespace ONNX_NAMESPACE;

ONNX_OPERATOR_KERNEL_EX(RandomNormal, kOnnxDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
                        RandomNormal);

ONNX_OPERATOR_KERNEL_EX(RandomNormalLike, kOnnxDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
                            .TypeConstraint("T2", DataTypeImpl::AllIEEEFloatTensorTypes()),
                        RandomNormalLike);

ONNX_OPERATOR_KERNEL_EX(RandomUniform, kOnnxDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
                        RandomUniform);

ONNX_OPERATOR_KERNEL_EX(RandomUniformLike, kOnnxDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T1", DataTypeImpl::AllTensorTypes())
                            .TypeConstraint("T2", DataTypeImpl::AllIEEEFloatTensorTypes()),
                        RandomUniformLike);

#define RANDOM_COMPUTE_IMPL(name)                                                                        \
  template <typename T>                                                                                  \
  struct name##ComputeImpl {                                                                             \
    void operator()(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const float alpha, \
                    const float beta, PhiloxGenerator& generator, Tensor& Y) const {                     \
      typedef typename ToCudaType<T>::MappedType CudaT;                                                  \
      CudaT* Y_data = reinterpret_cast<CudaT*>(Y.MutableData<T>());                             \
      name##KernelImpl<CudaT>(prop, stream, N, alpha, beta, generator, Y_data);                          \
    }                                                                                                    \
  };

RANDOM_COMPUTE_IMPL(RandomNormal)
RANDOM_COMPUTE_IMPL(RandomUniform)

#undef RANDOM_COMPUTE_IMPL

Status RandomNormalBase::ComputeNormal(const CudaKernel& cuda_kernel, OpKernelContext& ctx, const TensorShape& shape, int dtype) const {
  Tensor& Y = *ctx.Output(0, shape);
  const int64_t N = shape.Size();
  PhiloxGenerator& generator = GetPhiloxGenerator();
  utils::MLTypeCallDispatcher<float, MLFloat16, double> t_disp(dtype);
  t_disp.Invoke<RandomNormalComputeImpl>(cuda_kernel.GetDeviceProp(), cuda_kernel.Stream(), N, scale_, mean_, generator, Y);
  return Status::OK();
}

Status RandomNormalLike::ComputeInternal(OpKernelContext* p_ctx) const {
  const Tensor* p_X = p_ctx->Input<Tensor>(0);

  if (!p_X) {
    return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  }

  int dtype = GetDType();
  if (dtype == TensorProto_DataType_UNDEFINED && !p_X->IsDataType<float>() && !p_X->IsDataType<double>() &&
      !p_X->IsDataType<MLFloat16>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Output data type is required to be one of float types, but got incompatible data type ",
                           p_X->DataType(), " from input tensor.");
  }

  if (dtype == TensorProto_DataType_UNDEFINED)
    dtype = p_X->GetElementType();

  return ComputeNormal(*this, *p_ctx, p_X->Shape(), dtype);
}

Status RandomUniformBase::ComputeUniform(const CudaKernel& cuda_kernel, OpKernelContext& ctx, const TensorShape& shape, int dtype) const {
  Tensor& Y = *ctx.Output(0, shape);
  const int64_t N = shape.Size();
  PhiloxGenerator& generator = GetPhiloxGenerator();
  utils::MLTypeCallDispatcher<float, MLFloat16, double> t_disp(dtype);
  t_disp.Invoke<RandomUniformComputeImpl>(cuda_kernel.GetDeviceProp(), cuda_kernel.Stream(), N, range_, from_, generator, Y);
  return Status::OK();
}

Status RandomUniformLike::ComputeInternal(OpKernelContext* p_ctx) const {
  const Tensor* p_X = p_ctx->Input<Tensor>(0);

  if (!p_X) {
    return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  }

  int dtype = GetDType();
  if (dtype == TensorProto_DataType_UNDEFINED && !p_X->IsDataType<float>() && !p_X->IsDataType<double>() &&
      !p_X->IsDataType<MLFloat16>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Output data type is required to be one of float types, but got incompatible data type ",
                           p_X->DataType(), " from input tensor.");
  }

  if (dtype == TensorProto_DataType_UNDEFINED)
    dtype = p_X->GetElementType();

  return ComputeUniform(*this, *p_ctx, p_X->Shape(), dtype);
}

}  // namespace cuda
}  // namespace onnxruntime
