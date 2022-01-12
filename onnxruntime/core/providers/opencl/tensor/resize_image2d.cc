#include "resize_image2d.h"

#include "core/providers/cpu/tensor/upsample.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME resize_kernel_src
#include "opencl_generated/tensor/kernels/resize_image2d.cl.inc"
}  // namespace

class Resize : public OpenCLKernel, UpsampleBase {
 public:
  explicit Resize(const OpKernelInfo& info)
      : OpenCLKernel(info), UpsampleBase(info) {
    VLOGS_DEFAULT(0) << "Init Resize (OpenCLKernel)";
    LoadProgram(resize_kernel_src, resize_kernel_src_len);
    LoadKernel("ResizeBilinear2D");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("Resize::Compute");
    VLOG_CL_NODE();
    ORT_RETURN_IF(mode_ != UpsampleMode::LINEAR, "only supports linear interpolation");

    const auto* X = context->Input<Tensor>(0);
    const auto& X_shape = X->Shape();
    ORT_RETURN_IF(X_shape.NumDimensions() != 4, "only support 4D NCHW input");

    std::vector<int64_t> Y_shape(X->Shape().GetDims().size());
    ComputeOutputShape(scales_, X_shape.GetDims(), Y_shape);
    const auto* Y = context->Output(0, Y_shape);
    VLOG_CL_IMAGE2D("Input", X);
    VLOG_CL_IMAGE2D("Output", Y);

    auto desc = Image2DDesc::PackFromTensorNCHW(Y->Shape());

    ZoneNamedN(_tracy_ResizeBilinear2D, "ResizeBilinear2D (kernel launch)", true);
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("ResizeBilinear2D")}
            .setInt2(desc.Width(), desc.Height())
            .setImage2Ds(*X, *Y)
            .setInt2(X_shape[3], X_shape[2])
            .setInt2(Y_shape[3], Y_shape[2])
            .setArg<cl_float>(1.0 / scales_[3])
            .setArg<cl_float>(1.0 / scales_[2])
            .Launch(*exec_, desc.AsNDRange()));

    return Status::OK();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    11, 12,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .InputMemoryType(OrtMemTypeCPUInput, {1, 2, 3})
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    Resize)

ONNX_OPENCL_OPERATOR_KERNEL(
    Resize,
    13,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .InputMemoryType(OrtMemTypeCPUInput, {1, 2, 3})
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    Resize)

}  // namespace opencl
}  // namespace onnxruntime
