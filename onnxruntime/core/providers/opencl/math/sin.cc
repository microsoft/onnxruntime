// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "trigonometric.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace {

#define CONTENT_NAME sin_kernel_src
#include "opencl_generated/math/kernels/sin.cl.inc"

}  // namespace

namespace onnxruntime {
namespace opencl {

class sin : public OpenCLKernel {
 public:
  explicit sin(const OpKernelInfo& info) : OpenCLKernel(info) {
    LoadProgram(sin_kernel_src, sin_kernel_src_len);
    LoadKernel("sin_float");
    // LoadKernel("sin_double");
    // LoadKernel("sin_fp16");
  }

  Status Compute(OpKernelContext* context) const override;

  template <typename T>
  Status ComputeImp_(OpKernelContext* context) const;
};

template <>
Status sin::ComputeImp_<float>(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  Tensor* Y = context->Output(0, X->Shape());
  const int64_t input_size = X->Shape().Size();
  if (input_size == 0)
    return Status::OK();
  ORT_ENFORCE(input_size < std::numeric_limits<std::ptrdiff_t>::max());

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("sin_float")}
          .SetBuffers(*X, *Y)
          .Launch(*exec_, {X->SizeInBytes() / 4, 1, 1}));

  return Status::OK();
}

Status sin::Compute(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  auto element_type = X->GetElementType();

  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return ComputeImp_<float>(ctx);
    // case ONNX_NAMESPACE::TensorProto_DataType_xxx:
    //   return ComputeImp_<fp16>(ctx);
    // case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    //   return ComputeImp_<double>(ctx);
    default:
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                    "Unsupported input type for Sin operator.");
  }
}

ONNX_OPENCL_OPERATOR_KERNEL(
    Sin,
    7,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()}),
    sin);

}  // namespace opencl
}  // namespace onnxruntime
