#include "conv_image2d.h"

#include <sstream>

#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/opencl/opencl_kernel.h"

namespace {
#define CONTENT_NAME conv_kernel_src
#include "opencl_generated/nn/kernels/conv_image2d.cl.inc"
}  // namespace

namespace onnxruntime {
namespace opencl {

class Conv : public OpenCLKernel {
 public:
  explicit Conv(const OpKernelInfo& info) : OpenCLKernel(info), attrs_{info} {
    VLOGS_DEFAULT(0) << "[CL] Init Conv (OpenCLKernel), auto_pad:" << static_cast<int>(attrs_.auto_pad) << ", dilations: " << attrs_.dilations << ", group: " << attrs_.group;
    LoadProgram(conv_kernel_src, conv_kernel_src_len);
    LoadKernel("Conv2D");
    LoadKernel("DepthwiseConv2D");
  };

  Status Compute(OpKernelContext* context) const override {
    VLOGS_DEFAULT(0) << "[CL] Node: " << context->GetNodeName()
                     << ", num inputs: " << context->InputCount()
                     << ", num outputs: " << context->OutputCount();
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* W = context->Input<Tensor>(1);
    const Tensor* B = context->InputCount() >= 2 ? context->Input<Tensor>(2) : nullptr;

    ORT_RETURN_IF_ERROR(attrs_.ValidateInputShape(X, W));
    auto N = X->Shape()[0];
    auto co_total = W->Shape()[0];
    auto co_per_group = co_total / attrs_.group;
    auto ci_per_group = W->Shape()[1];

    std::vector<int64_t> K;
    ORT_RETURN_IF_ERROR(attrs_.ComputeKernelShape(W->Shape(), K));

    auto rank = K.size();
    std::vector<int64_t> P(attrs_.pads);
    if (P.empty()) {
      P.resize(rank * 2, 0);
    }
    std::vector<int64_t> D(attrs_.dilations);
    if (D.empty()) {
      D.resize(rank, 1);
    }
    std::vector<int64_t> S(attrs_.strides);
    if (S.empty()) {
      S.resize(rank, 1);
    }

    for (size_t i = 0; i < P.size() / 2; ++i) {
      ORT_ENFORCE(P[i] == P[2 * i], "padding can only be symmetric");
    }

    std::vector<int64_t> Y_spatial_shape;
    ORT_RETURN_IF_ERROR(attrs_.InferOutputShape(X->Shape().Slice(2), K, S, D, P, Y_spatial_shape));
    std::vector<int64_t> Y_shape;
    Y_shape.reserve(2 + rank);
    Y_shape.insert(Y_shape.end(), {N, co_total});
    Y_shape.insert(Y_shape.end(), Y_spatial_shape.begin(), Y_spatial_shape.end());
    Tensor* Y = context->Output(0, Y_shape);

    VLOGS_DEFAULT(0) << "[CL]  Input X shape " << X->Shape() << " " << X->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*X)() << ")";
    VLOGS_DEFAULT(0) << "[CL]  Input W shape " << W->Shape() << " " << W->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*W)() << ")";
    if (B != nullptr) {
      VLOGS_DEFAULT(0) << "[CL]  Input B shape " << B->Shape() << " " << B->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*B)() << ")";
    }
    VLOGS_DEFAULT(0) << "[CL] Output Y shape " << Y->Shape() << " " << Y->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*Y)() << ")";

    if (rank == 2) {
      if (ci_per_group == 1 && co_per_group == 1) {
        return DepthwiseConv2D(X, W, B, Y, K, S, P, D, attrs_.group);
      }
      return Conv2D(X, W, B, Y, K, S, P, D, attrs_.group);
    }

    ORT_NOT_IMPLEMENTED("Conv of rank ", rank, " is not implemented");
  }

 private:
  Status DepthwiseConv2D(const Tensor* X,
                         const Tensor* W,
                         const Tensor* B,
                         Tensor* Y,
                         const std::vector<int64_t>& K,
                         const std::vector<int64_t>& S,
                         const std::vector<int64_t>& P,
                         const std::vector<int64_t>& D,
                         const int group) const {
    VLOGS_DEFAULT(0) << "[CL] DepthwiseConv2D, X:" << X->Shape() << " W:" << W->Shape()
                     << " B:" << B->Shape() << " Y:" << Y->Shape()
                     << " K:" << K << " S:" << S << " P:" << P << " D:" << D << " group:" << group;

    auto C_in = X->Shape()[1];
    auto H_in = X->Shape()[2];
    auto W_in = X->Shape()[3];
    auto shape = Y->Shape();
    auto N = shape[0];
    auto C_out = shape[1];
    auto H_out = shape[2];
    auto W_out = shape[3];
    ORT_ENFORCE(C_in == C_out, "depthwise conv2d enforcement failure");
    uint32_t gsx = CeilDiv(C_out, 4) * CeilDiv(W_out, 4);
    uint32_t gsy = N * H_out;
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("DepthwiseConv2D")}
            .setArg<cl_int>(gsx)
            .setArg<cl_int>(gsy)
            .setImage2Ds(*X, *W, *B, *Y)
            .setInt2(W_in, H_in)
            .setInt2(W_out, H_out)
            .setInt2(K[0], K[1])
            .setInt2(S[0], S[1])
            .setInt2(P[0], P[1])
            .setInt2(D[0], D[1])
            .setArg<cl_int>(0)
            .Launch(GetCommandQueue(), {gsx, gsy}));

    return Status::OK();
  }

  Status Conv2D(const Tensor* X,
                const Tensor* W,
                const Tensor* B,
                Tensor* Y,
                const std::vector<int64_t>& K,
                const std::vector<int64_t>& S,
                const std::vector<int64_t>& P,
                const std::vector<int64_t>& D,
                const int group) const {
    VLOGS_DEFAULT(0) << "[CL] Conv2D, X:" << X->Shape() << " W:" << W->Shape()
                     << " B:" << B->Shape() << " Y:" << Y->Shape()
                     << " K:" << K << " S:" << S << " P:" << P << " D:" << D << " group:" << group;
    ORT_ENFORCE(group == 1, "group != 1 is not supported currently in Conv2D");

    const auto& xshape = X->Shape();
    const auto& yshape = Y->Shape();

    auto C_in = xshape[1];
    auto H_in = xshape[2];
    auto W_in = xshape[3];
    auto N = yshape[0];
    auto C_out = yshape[1];
    auto H_out = yshape[2];
    auto W_out = yshape[3];
    uint32_t gsx = CeilDiv(C_out, 4) * CeilDiv(W_out, 4);
    uint32_t gsy = N * H_out;
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Conv2D")}
            .setArg<cl_int>(gsx)
            .setArg<cl_int>(gsy)
            .setImage2Ds(*X, *W, *B, *Y)
            .setInt2(W_in, H_in)
            .setArg<cl_int>(CeilDiv(C_in, 4))
            .setInt2(W_out, H_out)
            .setInt2(K[0], K[1])
            .setInt2(S[0], S[1])
            .setInt2(P[0], P[1])
            .setInt2(D[0], D[1])
            .setArg<cl_int>(CeilDiv(W_out, 4))
            .setArg<cl_int>(0)
            .Launch(GetCommandQueue(), {gsx, gsy}));

    return Status::OK();
  }

  ConvAttributes attrs_;
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Conv,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)   /* X */
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 1)   /* W */
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 2)   /* B */
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0), /* Y */
    Conv)

}  // namespace opencl
}  // namespace onnxruntime
