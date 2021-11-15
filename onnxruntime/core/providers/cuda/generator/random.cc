// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/generator/random.h"

namespace onnxruntime {
namespace cuda {

using namespace ONNX_NAMESPACE;

namespace {

static TensorProto::DataType InferDataType(const Tensor& tensor) {
  auto elem_type = tensor.GetElementType();
  int dtype = TensorProto_DataType_UNDEFINED;
  if (elem_type == TensorProto_DataType_FLOAT || elem_type == TensorProto_DataType_DOUBLE ||
      elem_type == TensorProto_DataType_FLOAT16) {
    dtype = elem_type;
  }
  return static_cast<TensorProto::DataType>(dtype);
}

};  // namespace

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

Status RandomNormalBase::Compute(OpKernelContext* p_ctx, const TensorShape& shape, int dtype) const {
  Tensor& Y = *p_ctx->Output(0, shape);
  const int64_t N = shape.Size();
  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();
  utils::MLTypeCallDispatcher<float, MLFloat16, double> t_disp(dtype);
  t_disp.Invoke<RandomNormalComputeImpl>(GetDeviceProp(), Stream(), N, scale_, mean_, generator, Y);
  return Status::OK();
}

Status RandomNormal::ComputeInternal(OpKernelContext* p_ctx) const { return Compute(p_ctx, shape_, dtype_); }

Status RandomNormalLike::ComputeInternal(OpKernelContext* p_ctx) const {
  const Tensor* p_X = p_ctx->Input<Tensor>(0);
  if (!p_X) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = p_X->Shape();
  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(*p_X);
  if (dtype == TensorProto_DataType_UNDEFINED) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Could not infer data type from input tensor with data type ",
                           p_X->DataType());
  }
  return Compute(p_ctx, shape, dtype);
}

Status RandomUniformBase::Compute(OpKernelContext* p_ctx, const TensorShape& shape, int dtype) const {
  Tensor& Y = *p_ctx->Output(0, shape);
  const int64_t N = shape.Size();
  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();
  utils::MLTypeCallDispatcher<float, MLFloat16, double> t_disp(dtype);
  t_disp.Invoke<RandomUniformComputeImpl>(GetDeviceProp(), Stream(), N, range_, from_, generator, Y);
  return Status::OK();
}

Status RandomUniform::ComputeInternal(OpKernelContext* p_ctx) const { return Compute(p_ctx, shape_, dtype_); }

Status RandomUniformLike::ComputeInternal(OpKernelContext* p_ctx) const {
  const Tensor* p_X = p_ctx->Input<Tensor>(0);
  if (!p_X) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = p_X->Shape();
  auto dtype = dtype_ != TensorProto_DataType_UNDEFINED ? dtype_ : InferDataType(*p_X);
  if (dtype == TensorProto_DataType_UNDEFINED) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Could not infer data type from input tensor with data type ",
                           p_X->DataType());
  }
  return Compute(p_ctx, shape, dtype);
}

}  // namespace cuda
}  // namespace onnxruntime