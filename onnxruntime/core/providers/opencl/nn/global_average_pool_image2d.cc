#include "global_average_pool.h"
#include "core/providers/opencl/opencl_kernel.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME global_avgpool_src
#include "opencl_generated/nn/kernels/global_average_pool_image2d.cl.inc"
}  // namespace

class GlobalAveragePool : public OpenCLKernel {
 public:
  explicit GlobalAveragePool(const OpKernelInfo& info) : OpenCLKernel(info) {
    VLOGS_DEFAULT(0) << "[CL] Init GlobalAveragePool (OpenCLKernel)";
    LoadProgram(global_avgpool_src, global_avgpool_src_len);
    LoadKernel("GlobalAveragePool");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("GlobalAveragePool::Compute");
    VLOG_CL_NODE();
    const auto* X = context->Input<Tensor>(0);
    const auto& X_shape = X->Shape();
    auto rank = X_shape.NumDimensions();
    ORT_RETURN_IF(rank < 2, "rank error");
    ORT_RETURN_IF(rank != 4, "only supports NCHW tensor");
    TensorShapeVector Y_shape(X_shape.Slice(0, 2).AsShapeVector());
    Y_shape.resize(rank, 1);
    const auto* Y = context->Output(0, Y_shape);
    VLOG_CL_IMAGE2D("Input", X);
    VLOG_CL_IMAGE2D("Output", Y);

    auto N = X_shape[0];
    auto C = X_shape[1];
    auto H = X_shape[2];
    auto W = X_shape[3];
    float invHW = 1.0 / (H * W);

    ZoneNamedN(_tracy_GlobalAveragePool, "GlobalAveragePool (kernel launch)", true);
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("GlobalAveragePool")}
            .setArg<cl_int>(CeilDiv(C, 4))
            .setArg<cl_int>(N)
            .setImage2Ds(*X, *Y)
            .setInt2(W, H)
            .setArg<cl_float>(invHW)
            .Launch(*exec_, {static_cast<uint32_t>(C), static_cast<uint32_t>(N)}));

    return Status::OK();
  }
};

ONNX_OPENCL_OPERATOR_KERNEL(
    GlobalAveragePool,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    GlobalAveragePool);

}  // namespace opencl
}  // namespace onnxruntime
