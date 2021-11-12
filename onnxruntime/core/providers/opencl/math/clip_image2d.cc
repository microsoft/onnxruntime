// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "clip.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME clip_kernel_src
#include "opencl_generated/math/kernels/clip_image2d.cl.inc"
}  // namespace

Status ClipComputeImpl(
    const OpenCLExecutionProvider* exec,
    cl_kernel clip_kernel,
    const Tensor* X,
    const Tensor* Y,
    cl_float lower_bound,
    cl_float upper_bound) {
  ZoneScopedN("ClipComputeImpl");

  const auto& X_shape = X->Shape();
  auto desc = Image2DDesc::PackFromTensorNCHW(X_shape);
  ZoneNamedN(_tracy_ClipNCHW, "Clip (kernel launch)", true);
  ORT_RETURN_IF_ERROR(
      KernelLauncher{clip_kernel}
          .SetArg<cl_int>(desc.Width())
          .SetArg<cl_int>(desc.Height())
          .SetImage2D(*X)
          .SetImage2D(*Y)
          .SetArg<cl_int>(lower_bound)
          .SetArg<cl_int>(upper_bound)
          .Launch(*exec, desc.AsNDRange()));

  return Status::OK();
}

class Clip6 : public OpenCLKernel {
 public:
  explicit Clip6(const OpKernelInfo& info)
      : OpenCLKernel(info) {
    LoadProgram(clip_kernel_src, clip_kernel_src_len);
    LoadKernel("Clip");
    info.GetAttrOrDefault("min", &lower_bound_, std::numeric_limits<float>::lowest());
    info.GetAttrOrDefault("max", &upper_bound_, std::numeric_limits<float>::max());
  };

  Status Compute(OpKernelContext* context) const override {
    VLOG_CL_NODE() << ", min:" << lower_bound_ << ", max:" << upper_bound_;

    const auto* X = context->Input<Tensor>(0);
    const auto* Y = context->Output(0, X->Shape());
    VLOG_CL_IMAGE2D("Input[0]", X);
    VLOG_CL_IMAGE2D("Output[0]", Y);
    return ClipComputeImpl(exec_, GetKernel("Clip"), X, Y, lower_bound_, upper_bound_);
  }

 private:
  float lower_bound_;
  float upper_bound_;
};

class Clip : public OpenCLKernel {
 public:
  explicit Clip(const OpKernelInfo& info)
      : OpenCLKernel(info) {
    LoadProgram(clip_kernel_src, clip_kernel_src_len);
    LoadKernel("Clip");
  }
  Status Compute(OpKernelContext* context) const override {
    VLOG_CL_NODE();
    const auto* X = context->Input<Tensor>(0);
    const auto* min = context->Input<Tensor>(1);
    const auto* max = context->Input<Tensor>(2);
    const auto* Y = context->Output(0, X->Shape());
    float min_val = std::numeric_limits<float>::lowest();
    float max_val = std::numeric_limits<float>::max();
    if (min) {
      ORT_RETURN_IF_NOT(min->Shape().IsScalar(), "min should be a scalar.");
      min_val = *(min->Data<float>());
    }
    if (max) {
      ORT_RETURN_IF_NOT(max->Shape().IsScalar(), "max should be a scalar.");
      max_val = *(max->Data<float>());
    }

    return ClipComputeImpl(exec_, GetKernel("Clip"), X, Y, min_val, max_val);
  }
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Clip,
    6,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Clip6);
ONNX_OPENCL_OPERATOR_KERNEL(
    Clip,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, {1, 2}),
    Clip);
}  // namespace opencl
}  // namespace onnxruntime
