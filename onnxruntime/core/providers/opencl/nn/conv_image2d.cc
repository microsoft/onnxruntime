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

// TODO: This is shared across C++ code and opencl kernel code
// unify them in a shared header
enum ActivationKind {
  ActivationKind_None = 0,
  ActivationKind_ReLU = 1,
  ActivationKind_Clip = 5,
};

struct FusedConvAct {
  ActivationKind kind;
  float param0;
  float param1;

  FusedConvAct() : kind{ActivationKind_None}, param0{std::numeric_limits<float>::quiet_NaN()}, param1{std::numeric_limits<float>::quiet_NaN()} {}

  Status LoadInfo(const OpKernelInfo& info) {
    std::string activation_type;
    info.GetAttrOrDefault<std::string>("activation", &activation_type, "None");
    size_t activation_params_count = 0;
    if (activation_type == "None") {
      kind = ActivationKind_None;
    } else if (activation_type == "Relu") {
      kind = ActivationKind_ReLU;
    } else if (activation_type == "Clip") {
      kind = ActivationKind_Clip;
      activation_params_count = 2;
    } else {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "unimplemented activation: " + activation_type);
    }

    std::vector<float> activation_params = info.GetAttrsOrDefault<float>("activation_params");
    ORT_RETURN_IF(activation_params.size() < activation_params_count, "insufficient size of activation_params");
    if (activation_params_count >= 1) {
      param0 = activation_params[0];
    }
    if (activation_params_count >= 2) {
      param1 = activation_params[1];
    }

    return Status::OK();
  }
};

class Conv : public OpenCLKernel {
 public:
  explicit Conv(const OpKernelInfo& info) : OpenCLKernel(info), attrs_{info} {
    ORT_THROW_IF_ERROR(act_info_.LoadInfo(info));
    VLOGS_DEFAULT(0) << "[CL] Init Conv (OpenCLKernel), auto_pad:" << static_cast<int>(attrs_.auto_pad) << ", dilations: " << attrs_.dilations << ", group: " << attrs_.group;

    LoadProgram(conv_kernel_src, conv_kernel_src_len);
    LoadKernel("Conv2D");
    LoadKernel("Conv2DK1");
    LoadKernel("Conv2DK1S1");
    LoadKernel("DepthwiseConv2D");
    LoadKernel("DepthwiseConv2DS1");
  };

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("Conv::Compute");

    VLOG_CL_NODE();
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

    VLOG_CL_IMAGE2D("Input X", X);
    VLOG_CL_IMAGE2D("Input W", W);
    if (B != nullptr) {
      VLOG_CL_IMAGE2D("Input B", B);
    }
    VLOG_CL_IMAGE2D("Output Y", B);

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
    ZoneScopedN("DepthwiseConv2D");
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

    bool S1 = S[0] == 1 && S[1] == 1 && D[0] == 1 && D[1] == 1;

    if (S1) {
      ZoneScopedN("DepthwiseConv2DS1 (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel("DepthwiseConv2DS1")}
              .setArg<cl_int>(gsx)
              .setArg<cl_int>(gsy)
              .setImage2Ds(*X, *W, *B, *Y)
              .setInt2(W_in, H_in)
              .setInt2(W_out, H_out)
              .setInt2(K[0], K[1])
              .setInt2(P[0], P[1])
              .setArg<cl_int>(act_info_.kind)
              .setArg<cl_float>(act_info_.param0)
              .setArg<cl_float>(act_info_.param1)
              .Launch(*exec_, {gsx, gsy}));
    } else {
      ZoneScopedN("DepthwiseConv2D (kernel launch)");
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
              .setArg<cl_int>(act_info_.kind)
              .setArg<cl_float>(act_info_.param0)
              .setArg<cl_float>(act_info_.param1)
              .Launch(*exec_, {gsx, gsy}));
    }

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
    ZoneScopedN("Conv2D");
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

    bool K1 = K[0] == 1 && K[1] == 1 && P[0] == 0 && P[1] == 0;
    bool S1 = S[0] == 1 && S[1] == 1 && D[0] == 1 && D[1] == 1;

    if (K1 && S1) {
      ZoneScopedN("Conv2DK1S1 (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel("Conv2DK1S1")}
              .setArg<cl_int>(gsx)
              .setArg<cl_int>(gsy)
              .setImage2Ds(*X, *W, *B, *Y)
              .setInt2(W_in, H_in)
              .setArg<cl_int>(CeilDiv(C_in, 4))
              .setArg<cl_int>(CeilDiv(W_out, 4))
              .setArg<cl_int>(act_info_.kind)
              .setArg<cl_float>(act_info_.param0)
              .setArg<cl_float>(act_info_.param1)
              .Launch(*exec_, {gsx, gsy}));
    } else if (K1) {
      ZoneScopedN("Conv2DK1 (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel("Conv2DK1")}
              .setArg<cl_int>(gsx)
              .setArg<cl_int>(gsy)
              .setImage2Ds(*X, *W, *B, *Y)
              .setInt2(W_in, H_in)
              .setArg<cl_int>(CeilDiv(C_in, 4))
              .setInt2(W_out, H_out)
              .setInt2(S[0], S[1])
              .setInt2(D[0], D[1])
              .setArg<cl_int>(CeilDiv(W_out, 4))
              .setArg<cl_int>(act_info_.kind)
              .setArg<cl_float>(act_info_.param0)
              .setArg<cl_float>(act_info_.param1)
              .Launch(*exec_, {gsx, gsy}));
    } else {
      ZoneScopedN("Conv2D (kernel launch)");
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
              .setArg<cl_int>(act_info_.kind)
              .setArg<cl_float>(act_info_.param0)
              .setArg<cl_float>(act_info_.param1)
              .Launch(*exec_, {gsx, gsy}));
    }
    return Status::OK();
  }

  ConvAttributes attrs_;
  FusedConvAct act_info_;
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
    Conv);

ONNX_OPERATOR_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0)   /* X */
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 1)   /* W */
        .InputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 2)   /* B */
        .OutputMemoryType((OrtMemType)CLMemType::OPENCL_IMAGE_2D, 0), /* Y */
    Conv                                                              // register the Conv OpKernel as the FusedConv impl
);

}  // namespace opencl
}  // namespace onnxruntime
