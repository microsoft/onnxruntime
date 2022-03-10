#include "max_pool.h"

#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME max_pool_kernel_src
#include "opencl_generated/nn/kernels/max_pool_image2d.cl.inc"
}  // namespace

class MaxPool : public OpenCLKernel {
 public:
  explicit MaxPool(const OpKernelInfo& info)
      : OpenCLKernel(info), attrs_(info, info.GetKernelDef().OpName(), info.node().SinceVersion()) {
    if (attrs_.global_pooling) ORT_THROW("[CL] MaxPool does not support global max pooling");
    VLOGS_DEFAULT(0) << "Init MaxPool (OpenCLKernel)";
    LoadProgram(max_pool_kernel_src, max_pool_kernel_src_len);
    LoadKernel("MaxPool");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("MaxPool::Compute");
    VLOG_CL_NODE();

    const auto* X = context->Input<Tensor>(0);
    const auto& X_shape = X->Shape();
    ORT_RETURN_IF(X_shape.NumDimensions() != 4, "Input dimension must be 4, aka, NCHW tensor");

    TensorShapeVector pads = attrs_.pads;
    TensorShapeVector Y_shape = attrs_.SetOutputSize(X_shape, X_shape[1], &pads);
    const auto* Y = context->Output(0, Y_shape);
    VLOG_CL_IMAGE2D("Input", X);
    VLOG_CL_IMAGE2D("Output", Y);
    VLOGS_DEFAULT(0) << "[CL] MaxPool, X:" << X->Shape() << " Y:" << Y->Shape()
                     << " K:" << attrs_.kernel_shape << " S:" << attrs_.strides << " P:" << pads;

    // auto input_desc = Image2DDesc::PackFromTensorNCHW(X_shape);
    const auto& N = Y_shape[0];
    const auto& C = Y_shape[1];
    const auto& H_out = Y_shape[2];
    const auto& W_out = Y_shape[3];

    const auto& H_in = X_shape[2];
    const auto& W_in = X_shape[3];

    int64_t channel_blocks = CeilDiv(C, 4);

    ZoneNamedN(_tracy_MaxPool, "MaxPool (kernel launch)", true);
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("MaxPool")}
            .setInt3(channel_blocks, W_out, N * H_out)
            .setImage2Ds(*X, *Y)
            .setInt2(W_in, H_in)
            .setArg<cl_int>(H_out)
            .setInt2(attrs_.kernel_shape[0], attrs_.kernel_shape[1])
            .setInt2(attrs_.strides[0], attrs_.strides[1])
            .setInt2(pads[0], pads[1])
            .Launch(*exec_, {channel_blocks, W_out, N * H_out}));

    return Status::OK();
  }

 private:
  PoolAttributes attrs_;
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MaxPool,
    kOnnxDomain,
    8, 11,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MaxPool)

ONNX_OPENCL_OPERATOR_KERNEL(
    MaxPool,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MaxPool)

}  // namespace opencl
}  // namespace onnxruntime
