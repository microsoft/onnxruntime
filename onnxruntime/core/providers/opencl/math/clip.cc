#include "clip.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME clip_kernel_src
#include "opencl_generated/math/kernels/clip_image2d.cl.inc"
}  // namespace

class Clip6 : public OpenCLKernel {
 public:
  explicit Clip6(const OpKernelInfo& info)
      : OpenCLKernel(info) {
    VLOGS_DEFAULT(0) << "Init Clip (OpenCLKernel)";
    LoadProgram(clip_kernel_src, clip_kernel_src_len);
    LoadKernel("ClipNCHW");
    info.GetAttrOrDefault("min", &lower_bound_, std::numeric_limits<float>::lowest());
    info.GetAttrOrDefault("max", &upper_bound_, std::numeric_limits<float>::max());
  };

  Status Compute(OpKernelContext* context) const override {
    VLOG_CL_NODE();

    const auto* X = context->Input<Tensor>(0);
    const auto* Y = context->Output(0, X->Shape());
    VLOG_CL_IMAGE2D("Input[0]", X);
    VLOG_CL_IMAGE2D("Output[0]", Y);

    const auto& X_shape = X->Shape();
    auto desc = Image2DDesc::PackFromTensorNCHW(X_shape);
    cl_int C = X_shape[1];
    cl_int W = X_shape[3];

    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("ClipNCHW")}
            .setArg<cl_int>(desc.Width())
            .setArg<cl_int>(desc.Height())
            .setImage2D(*X)
            .setImage2D(*Y)
            .setArg(C)
            .setArg(W)
            .setArg(lower_bound_)
            .setArg(upper_bound_)
            .Launch(exec_->GetCommandQueue(), desc.AsNDRange()));

    return Status::OK();
  }

 private:
  cl_float lower_bound_;
  cl_float upper_bound_;
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Clip,
    6,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    Clip6)

}  // namespace opencl
}  // namespace onnxruntime
