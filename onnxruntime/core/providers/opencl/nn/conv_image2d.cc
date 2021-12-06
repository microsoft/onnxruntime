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
    auto C_out = W->Shape()[0];

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

    std::vector<int64_t> Y_spatial_shape;
    ORT_RETURN_IF_ERROR(attrs_.InferOutputShape(X->Shape().Slice(2), K, S, D, P, Y_spatial_shape));
    std::vector<int64_t> Y_shape;
    Y_shape.reserve(2 + rank);
    Y_shape.insert(Y_shape.end(), {N, C_out});
    Y_shape.insert(Y_shape.end(), Y_spatial_shape.begin(), Y_spatial_shape.end());
    Tensor* Y = context->Output(0, Y_shape);

    VLOGS_DEFAULT(0) << "[CL]  Input X shape " << X->Shape() << " " << X->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*X)() << ")";
    VLOGS_DEFAULT(0) << "[CL]  Input W shape " << W->Shape() << " " << W->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*W)() << ")";
    if (B != nullptr) {
      VLOGS_DEFAULT(0) << "[CL]  Input B shape " << B->Shape() << " " << B->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*B)() << ")";
    }
    VLOGS_DEFAULT(0) << "[CL] Output Y shape " << Y->Shape() << " " << Y->DataRaw() << " --> cl::Image(" << CL_IMAGE2D_FROM_TENSOR(*Y)() << ")";

    if (rank == 2) {
      if ()
      return Conv2D(X, W, B, Y, K, S, P, D, attrs_.group);
    }

    ORT_NOT_IMPLEMENTED("Conv of rank ", rank, " is not implemented");
  }

 private:
  Status Conv2D(const Tensor* X,
                const Tensor* W,
                const Tensor* B,
                Tensor* Y,
                const std::vector<int64_t>& K,
                const std::vector<int64_t>& S,
                const std::vector<int64_t>& P,
                const std::vector<int64_t>& D,
                const int group) const {
    ORT_ENFORCE(group == 1, "group != 1 is not supported currently in Conv2D");
    for (int i = 0; i < K.size() / 2; ++i) {
      ORT_ENFORCE(P[i] == P[2 * i], "padding can only be symmetric");
    }

    VLOGS_DEFAULT(0) << "[CL] Conv2D, X:" << X->Shape() << " W:" << W->Shape()
                     << " B:" << B->Shape() << " Y:" << Y->Shape()
                     << " K:" << K << " S:" << S << " P:" << P << " D:" << D;

    auto C_in = X->Shape()[1];
    auto H_in = X->Shape()[2];
    auto W_in = X->Shape()[3];
    auto H_out = Y->Shape()[2];
    auto W_out = Y->Shape()[3];
    auto Y_desc = Image2DDesc::PackFromTensorNCHW(Y->Shape());
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Conv2D")}
            .setArg<cl_int>(CeilDiv(Y_desc.Width(), 4))
            .setArg<cl_int>(Y_desc.Height())
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
            .Launch(GetCommandQueue(), {CeilDiv(Y_desc.UWidth(), 4), Y_desc.UHeight()}));

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
