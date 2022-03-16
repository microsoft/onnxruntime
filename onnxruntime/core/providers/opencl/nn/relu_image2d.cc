// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "relu.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME relu_kernel_src
#include "opencl_generated/nn/kernels/relu_image2d.cl.inc"
}  // namespace

class ReLU : public OpenCLKernel {
 public:
  explicit ReLU(const OpKernelInfo& info)
      : OpenCLKernel(info) {
    LoadProgram(relu_kernel_src, relu_kernel_src_len);
    LoadKernel("ReLU");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("ReLU6::Compute");
    VLOG_CL_NODE();

    const auto* X = context->Input<Tensor>(0);
    const auto* Y = context->Output(0, X->Shape());
    VLOG_CL_IMAGE2D("Input[0]", X);
    VLOG_CL_IMAGE2D("Output[0]", Y);

    auto desc = Image2DDesc::PackFromTensorNCHW(X->Shape());
    ZoneNamedN(_tracy_ReLUNCHW, "ReLU (kernel launch)", true);
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("ReLU")}
            .SetArg<cl_int>(desc.Width())
            .SetArg<cl_int>(desc.Height())
            .SetImage2D(*X)
            .SetImage2D(*Y)
            .SetArg<cl_float>(0.0)
            .Launch(*exec_, desc.AsNDRange()));

    return Status::OK();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6, 12,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReLU)

ONNX_OPENCL_OPERATOR_KERNEL(
    Relu,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReLU)

ONNX_OPENCL_OPERATOR_KERNEL(
    Relu,
    14,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReLU)

}  // namespace opencl
}  // namespace onnxruntime
