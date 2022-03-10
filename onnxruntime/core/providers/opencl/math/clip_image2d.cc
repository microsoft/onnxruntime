#include "clip.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME clip_kernel_src
#include "opencl_generated/math/kernels/clip_image2d.cl.inc"
}  // namespace

Status ClipComputeImpl(const OpenCLExecutionProvider* exec, cl_kernel clip_kernel, const Tensor* X, const Tensor* Y, cl_float lower_bound, cl_float upper_bound) {
  ZoneScopedN("ClipComputeImpl");
  VLOG_CL_IMAGE2D("Input[0]", X);
  VLOG_CL_IMAGE2D("Output[0]", Y);

  const auto& X_shape = X->Shape();
  auto desc = Image2DDesc::PackFromTensorNCHW(X_shape);
  ZoneNamedN(_tracy_ClipNCHW, "Clip (kernel launch)", true);
  ORT_RETURN_IF_ERROR(
      KernelLauncher{clip_kernel}
          .setArg<cl_int>(desc.Width())
          .setArg<cl_int>(desc.Height())
          .setImage2D(*X)
          .setImage2D(*Y)
          .setArg(lower_bound)
          .setArg(upper_bound)
          .Launch(*exec, desc.AsNDRange()));

  return Status::OK();
}

class Clip6 : public OpenCLKernel {
 public:
  explicit Clip6(const OpKernelInfo& info)
      : OpenCLKernel(info) {
    VLOGS_DEFAULT(0) << "Init Clip (OpenCLKernel)";
    LoadProgram(clip_kernel_src, clip_kernel_src_len);
    LoadKernel("Clip");
    info.GetAttrOrDefault("min", &lower_bound_, std::numeric_limits<float>::lowest());
    info.GetAttrOrDefault("max", &upper_bound_, std::numeric_limits<float>::max());
  };

  Status Compute(OpKernelContext* context) const override {
    VLOG_CL_NODE() << ", min:" << lower_bound_ << ", max:" << upper_bound_;

    const auto* X = context->Input<Tensor>(0);
    const auto* Y = context->Output(0, X->Shape());
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
    VLOGS_DEFAULT(0) << "Init Clip (OpenCLKernel)";
    LoadProgram(clip_kernel_src, clip_kernel_src_len);
    LoadKernel("Clip");
  }
  Status Compute(OpKernelContext* context) const override {
    VLOG_CL_NODE();
    const auto* X = context->Input<Tensor>(0);
    const auto* Y = context->Output(0, X->Shape());
    float lower_bound = std::numeric_limits<float>::lowest();
    float upper_bound = std::numeric_limits<float>::max();
    if (context->InputCount() > 1) {
      lower_bound = *(context->Input<Tensor>(1)->Data<float>());
    }
    if (context->InputCount() > 2) {
      upper_bound = *(context->Input<Tensor>(2)->Data<float>());
    }

    return ClipComputeImpl(exec_, GetKernel("Clip"), X, Y, lower_bound, upper_bound);
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
