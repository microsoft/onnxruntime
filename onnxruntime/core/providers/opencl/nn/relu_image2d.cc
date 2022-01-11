#include "relu_image2d.h"

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
    VLOGS_DEFAULT(0) << "Init ReLU (OpenCLKernel)";
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
            .setArg<cl_int>(desc.Width())
            .setArg<cl_int>(desc.Height())
            .setImage2D(*X)
            .setImage2D(*Y)
            .setArg<cl_float>(0.0)
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
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    ReLU)

ONNX_OPENCL_OPERATOR_KERNEL(
    Relu,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    ReLU)

ONNX_OPENCL_OPERATOR_KERNEL(
    Relu,
    14,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0),
    ReLU)




}  // namespace opencl
}  // namespace onnxruntime
