// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"

#include <sstream>

#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/opencl/opencl_allocator.h"
#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_data_transfer.h"
#include "core/providers/opencl/opencl_execution_provider.h"
#include "core/providers/opencl/opencl_program_manager.h"
#include "core/providers/opencl/nn/conv_winograd_helper.h"
#include "contrib_ops/cpu/fused_activation.h"
namespace {
#define CONTENT_NAME generic_conv_kernel_src
#include "opencl_generated/nn/kernels/conv_image2d_generic.cl.inc"
#define CONTENT_NAME depthwise_conv_kernel_src
#include "opencl_generated/nn/kernels/conv_image2d_depthwise.cl.inc"
#define CONTENT_NAME winograd_conv_kernel_src
#include "opencl_generated/nn/kernels/conv_image2d_winograd.cl.inc"

namespace kernel_name {
constexpr const char* Conv2D = "Conv2D";
constexpr const char* Conv2DK1 = "Conv2DK1";
constexpr const char* Conv2DK1S1 = "Conv2DK1S1";
constexpr const char* DepthwiseConv2D = "DepthwiseConv2D";
constexpr const char* DepthwiseConv2DS1 = "DepthwiseConv2DS1";
constexpr const char* TransformToMatrixV = "TransformToMatrixV";
constexpr const char* MatrixInnerProduct = "MatrixInnerProduct";
constexpr const char* TransformFromMatrixM = "TransformFromMatrixM";
constexpr const char* CopyGenericWeight = "CopyGenericConv2DWeightBufferToImage";
constexpr const char* CopyDepthwiseWeight = "CopyDepthwiseConv2DWeightBufferToImage";
constexpr const char* CopyWinogradWeight = "CopyBuffer2DToImage2D";
}  // namespace kernel_name

}  // namespace

namespace onnxruntime {
namespace opencl {

// TODO: This is shared across C++ code and opencl kernel code
// unify them in a shared header
typedef MLAS_ACTIVATION_KIND ActivationKind;
struct KernelUnitParam {
  std::vector<uint32_t> global_work_size = {};
  std::vector<uint32_t> local_work_size = {};
};

// local size 2d calculate, special for conv default.
std::vector<uint32_t> Conv2dCommonLocalWS2D(std::vector<uint32_t>& gws,
                                            const uint32_t max_workgroup_size,
                                            const uint32_t subgroup_size) {
  // TODO: impl
  ORT_UNUSED_PARAMETER(gws);
  ORT_UNUSED_PARAMETER(max_workgroup_size);
  ORT_UNUSED_PARAMETER(subgroup_size);
  return {};
}

Status CalWGSizeForWino(const Tensor* X, const Tensor* Y, std::vector<KernelUnitParam>& winokernel) {
  const auto& input_dims = X->Shape();
  const auto& output_dims = Y->Shape();

  const auto batch = output_dims[0];
  const auto output_channel = output_dims[1];
  const auto output_height = output_dims[2];
  const auto output_width = output_dims[3];

  const auto input_channel = input_dims[1];

  const auto round_up_ouptut_width = CeilDiv(output_width, 2);
  const auto round_up_output_height = CeilDiv(output_height, 2);
  const auto batch_round_h = batch * round_up_output_height;
  const auto output_channel_blocks = CeilDiv(output_channel, 4);
  const auto input_channel_blocks = CeilDiv(input_channel, 4);
  const auto round_up_4x4_ouptut_width = CeilDiv(round_up_ouptut_width, 4);

  winokernel[0].global_work_size = {static_cast<uint32_t>(input_channel_blocks * round_up_ouptut_width),
                                    static_cast<uint32_t>(batch_round_h)};
  winokernel[0].local_work_size = Conv2dCommonLocalWS2D(
      winokernel[0].global_work_size, 0, 0);

  winokernel[1].global_work_size = {static_cast<uint32_t>(output_channel_blocks * round_up_4x4_ouptut_width),
                                    static_cast<uint32_t>(16 * batch_round_h)};

  winokernel[2].global_work_size = {static_cast<uint32_t>(output_channel_blocks * round_up_ouptut_width),
                                    static_cast<uint32_t>(batch_round_h)};
  winokernel[2].local_work_size = Conv2dCommonLocalWS2D(
      winokernel[2].global_work_size, 0, 0);
  return Status::OK();
}

enum class ConvKind : uint8_t {
  Generic,
  Depthwise,
  Winograd,
};

class Conv : public OpenCLKernel {
 public:
  explicit Conv(const OpKernelInfo& info) : OpenCLKernel(info), attrs_{info} {
    ORT_ENFORCE(GetFusedActivationAttr(info, act_info_).IsOK());
    auto status = InitConvKind();
    if (!status.IsOK()) {
      conv_kind_ = ConvKind::Generic;
      LOGS_DEFAULT(WARNING) << "InitConvKind Error: " << status.ErrorMessage()
                            << ", using ConvKind::Generic, this might harm inference performance.";
    }

    // TODO: maybe use transformer pass to seperate them into individual OpKernels
    switch (conv_kind_) {
      case ConvKind::Winograd:
        LoadProgram(winograd_conv_kernel_src, winograd_conv_kernel_src_len);
        LoadKernel(kernel_name::TransformToMatrixV);
        LoadKernel(kernel_name::MatrixInnerProduct);
        LoadKernel(kernel_name::TransformFromMatrixM);
        LoadKernel(kernel_name::CopyWinogradWeight);
        break;
      case ConvKind::Depthwise:
        LoadProgram(depthwise_conv_kernel_src, depthwise_conv_kernel_src_len);
        LoadKernel(kernel_name::DepthwiseConv2D);
        LoadKernel(kernel_name::DepthwiseConv2DS1);
        LoadKernel(kernel_name::CopyDepthwiseWeight);
        break;
      case ConvKind::Generic:
        LoadProgram(generic_conv_kernel_src, generic_conv_kernel_src_len);
        LoadKernel(kernel_name::Conv2D);
        LoadKernel(kernel_name::Conv2DK1);
        LoadKernel(kernel_name::Conv2DK1S1);
        LoadKernel(kernel_name::CopyGenericWeight);
        break;
    }
  };

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr /*alloc*/,
                 bool& is_packed, PrePackedWeights* /*prepacked_weights*/) override {
    is_packed = false;

    // only kernel weight is PrePack-ed
    if (input_idx == 1) {
      switch (conv_kind_) {
        case ConvKind::Winograd:
          ORT_RETURN_IF_ERROR(PackWinogradWeight(tensor));
          break;
        case ConvKind::Depthwise:
          ORT_RETURN_IF_ERROR(PackDepthwiseWeight(tensor));
          break;
        case ConvKind::Generic:
          ORT_RETURN_IF_ERROR(PackGenericWeight(tensor));
          break;
      }
      W_shape_ = tensor.Shape();
      is_packed = true;
    }
    return Status::OK();
  }

  Status Compute(OpKernelContext* context) const override {
    ZoneScopedN("Conv::Compute");
    VLOG_CL_NODE();

    size_t num_inputs = OpKernel::Node().InputDefs().size();
    const Tensor* X = context->Input<Tensor>(0);
    //const Tensor* W =packed_weight_;
    const Tensor* B = num_inputs >= 3 ? context->Input<Tensor>(2) : nullptr;
    const Tensor* Sum = num_inputs >= 4 ? context->Input<Tensor>(3) : nullptr;

    ORT_RETURN_IF_ERROR(attrs_.ValidateInputShape(X->Shape(), W_shape_));
    auto N = X->Shape()[0];
    auto co_total = W_shape_[0];

    TensorShapeVector K;
    ORT_RETURN_IF_ERROR(attrs_.ComputeKernelShape(W_shape_, K));

    auto rank = K.size();
    ConvAttributes::ConvPadVector P(attrs_.pads);
    if (P.empty()) {
      P.resize(rank * 2, 0);
    }
    TensorShapeVector D(attrs_.dilations);
    if (D.empty()) {
      D.resize(rank, 1);
    }
    TensorShapeVector S(attrs_.strides);
    if (S.empty()) {
      S.resize(rank, 1);
    }

    TensorShapeVector Y_spatial_shape;
    ORT_RETURN_IF_ERROR(attrs_.InferOutputShape(X->Shape().Slice(2), K, S, D, P, Y_spatial_shape));
    TensorShapeVector Y_shape;
    Y_shape.reserve(2 + rank);
    Y_shape.insert(Y_shape.end(), {N, co_total});
    Y_shape.insert(Y_shape.end(), Y_spatial_shape.begin(), Y_spatial_shape.end());
    Tensor* Y = context->Output(0, Y_shape);

    VLOG_CL_IMAGE2D("Input X", X);
    VLOG_CL_PREPACK("Input W", GetPackedWeight(), W_shape_);
    if (B != nullptr) {
      VLOG_CL_IMAGE2D("Input B", B);
    }
    VLOG_CL_IMAGE2D("Output Y", Y);

    if (rank == 2) {
      switch (conv_kind_) {
        case ConvKind::Winograd:
          return WinogradConv2D(X, B, Sum, Y, P);
        case ConvKind::Depthwise:
          return DepthwiseConv2D(X, B, Sum, Y, K, S, P, D, attrs_.group);
        case ConvKind::Generic:
          return Conv2D(X, B, Sum, Y, K, S, P, D, attrs_.group);
      }
    }

    ORT_NOT_IMPLEMENTED("Conv of rank ", rank, " is not implemented");
  }

 private:
  Status InitConvKind() {
    // kernel_shape in ConvAttributes is spatial dims, and it may not be
    // specified. So we use NodeArg here.
    const auto* weight_arg = Node().InputDefs()[1];

    // get number of output channel
    auto dim_channel_out = weight_arg->Shape()->dim(0);
    ORT_RETURN_IF(!utils::HasDimValue(dim_channel_out), "Kernel channel out dim value is not available");
    auto co_total = dim_channel_out.dim_value();
    auto co_per_group = co_total / attrs_.group;

    // get number of input channel
    auto dim_channel_in = weight_arg->Shape()->dim(1);
    ORT_RETURN_IF(!utils::HasDimValue(dim_channel_out), "Kernel channel in dim value is not available");
    auto ci_per_group = dim_channel_in.dim_value();

    if (ci_per_group == 1 && co_per_group == 1) {
      // TODO: relax co_per_group requirement
      conv_kind_ = ConvKind::Depthwise;
      return Status::OK();
    }

    if (attrs_.strides.size() == 2 && attrs_.strides[0] == 1 && attrs_.strides[1] == 1 &&
        co_total >= 32 && ci_per_group >= 32) {
      // get kernel spatial shape
      auto dim_kernel_h = weight_arg->Shape()->dim(2);
      auto dim_kernel_w = weight_arg->Shape()->dim(3);
      ORT_RETURN_IF(!utils::HasDimValue(dim_kernel_h), "Kernel spatial h dim value is not available");
      ORT_RETURN_IF(!utils::HasDimValue(dim_kernel_w), "Kernel spatial w dim value is not available");
      auto kernel_h = dim_kernel_h.dim_value();
      auto kernel_w = dim_kernel_w.dim_value();

      const auto* input_arg = Node().InputDefs()[0];
      auto dim_in_heigth = input_arg->Shape()->dim(3);
      ORT_RETURN_IF(!utils::HasDimValue(dim_in_heigth), "Input spatial h dim value is not available");
      auto in_heigth = dim_in_heigth.dim_value();

      if (kernel_w == 3 && kernel_h == 3 && (in_heigth * 1.0 / ci_per_group) <= 4) {
        conv_kind_ = ConvKind::Winograd;
        return Status::OK();
      }
    }

    conv_kind_ = ConvKind::Generic;
    return Status::OK();
  }

  Status PackGenericWeight(const Tensor& src) {
    ZoneScopedN("PackGenericWeight");
    auto shape = src.Shape();
    auto desc = Image2DDesc::PackFromConv2DWeight(shape);
    packed_weight_ = exec_->GetScratchImage2D(desc);
    CL_CHECK_MEM_OBJECT_IS_IMAGE_2D(GetPackedWeight());
    VLOGF_DEFAULT(V_COPY, "Copy    host(%p) --> Image2D(%p)", src.DataRaw(), GetPackedWeight());

    auto tmp = exec_->GetScratchBuffer(src.SizeInBytes());
    // TODO: refactor out clEnqueueWriteBuffer, backend api exposed
    ORT_RETURN_IF_CL_ERROR(clEnqueueWriteBuffer(exec_->GetCommandQueue(), tmp.get(),
                                                /*blocking_write=*/CL_FALSE, /*offset=*/0,
                                                src.SizeInBytes(), src.DataRaw(), 0, nullptr, nullptr));
    ORT_RETURN_IF_ERROR(KernelLauncher{GetKernel(kernel_name::CopyGenericWeight)}
                            .SetArg<cl_int>(desc.Width())
                            .SetArg<cl_int>(desc.Height())
                            .SetBuffer(tmp.get())
                            .SetInt4(shape[0], shape[1], shape[2], shape[3])
                            .SetArg<cl_int>(shape[2] * shape[3])
                            .SetImage2D(static_cast<cl_mem>(GetPackedWeight()))
                            .Launch(*exec_, desc.AsNDRange()));
    // TODO: refactor out clFinish, backend api exposed
    // Do sync copy, since we cannot extend the lifetime of src or tmp
    ORT_RETURN_IF_CL_ERROR(clFinish(exec_->GetCommandQueue()));
    return Status::OK();
  }

  Status PackDepthwiseWeight(const Tensor& src) {
    ZoneScopedN("PackDepthwiseWeight");
    auto shape = src.Shape();
    auto desc = Image2DDesc::PackFromDepthwiseConv2DWeight(shape);
    packed_weight_ = exec_->GetScratchImage2D(desc);
    CL_CHECK_MEM_OBJECT_IS_IMAGE_2D(GetPackedWeight());
    VLOGF_DEFAULT(V_COPY, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), GetPackedWeight());

    ORT_RETURN_IF(shape[1] != 1, "input channel per group must be 1 for depthwise conv2d");
    auto tmp = exec_->GetScratchBuffer(src.SizeInBytes());
    // TODO: refactor out clEnqueueWriteBuffer, backend api exposed
    ORT_RETURN_IF_CL_ERROR(clEnqueueWriteBuffer(exec_->GetCommandQueue(), tmp.get(),
                                                /*blocking_write=*/CL_FALSE, /*offset=*/0,
                                                src.SizeInBytes(), src.DataRaw(), 0, nullptr, nullptr));
    ORT_RETURN_IF_ERROR(KernelLauncher{GetKernel(kernel_name::CopyDepthwiseWeight)}
                            .SetArg<cl_int>(desc.Width())
                            .SetArg<cl_int>(desc.Height())
                            .SetBuffer(tmp.get())
                            .SetInt4(shape[0], shape[1], shape[2], shape[3])
                            .SetArg<cl_int>(/*shape[1] * */ shape[2] * shape[3])  // C_i * K_h * K_w, C_i == 1
                            .SetImage2D(static_cast<cl_mem>(GetPackedWeight()))
                            .Launch(*exec_, desc.AsNDRange()));
    // TODO: refactor out clFinish, backend api exposed
    // Do sync copy, since we cannot extend the lifetime of src or tmp
    ORT_RETURN_IF_CL_ERROR(clFinish(exec_->GetCommandQueue()));
    return Status::OK();
  }

  Status PackWinogradWeight(const Tensor& src) {
    ZoneScopedN("PackWinogradWeight");
    auto shape = src.Shape();
    auto desc = Image2DDesc::PackFromWinogradTransform(shape);
    packed_weight_ = exec_->GetScratchImage2D(desc);
    CL_CHECK_MEM_OBJECT_IS_IMAGE_2D(GetPackedWeight());

    // wino initialize
    ORT_RETURN_IF(shape[2] != 3 || shape[3] != 3, "Winograd only support 3x3 kernel");
    const auto output_channel = shape[0];
    const auto input_channel = shape[1];
    const auto kernel_size = shape[3];
    const auto unit_output = 2;  // 4x3

    auto cpu_alloc = exec_->GetAllocator(0, OrtMemTypeCPU);
    WinogradHelper helper(cpu_alloc, static_cast<int>(unit_output), static_cast<int>(kernel_size));
    auto transformd_weight = helper.TransformWeight(&src, output_channel, input_channel);

    VLOGF_DEFAULT(V_COPY, "[CL] copy    host(%p) --> Image2D(%p)", src.DataRaw(), GetPackedWeight());

    auto tmp = exec_->GetScratchBuffer(transformd_weight->SizeInBytes());
    ORT_RETURN_IF_CL_ERROR(clEnqueueWriteBuffer(exec_->GetCommandQueue(), tmp.get(),
                                                /*blocking_write=*/CL_FALSE, /*offset=*/0,
                                                transformd_weight->SizeInBytes(), transformd_weight->DataRaw(),
                                                0, nullptr, nullptr));
    ORT_RETURN_IF_ERROR(KernelLauncher{GetKernel(kernel_name::CopyWinogradWeight)}
                            .SetBuffer(tmp.get())
                            .SetImage2D(static_cast<cl_mem>(GetPackedWeight()))
                            .SetArg<cl_int>(desc.Width())
                            .SetArg<cl_int>(desc.Height())
                            .Launch(*exec_, desc.AsNDRange()));
    ORT_RETURN_IF_CL_ERROR(clFinish(exec_->GetCommandQueue()));
    return Status::OK();
  }

  Status DepthwiseConv2D(const Tensor* X,
                         const Tensor* B,
                         const Tensor* Sum,
                         Tensor* Y,
                         const TensorShapeVector& K,
                         const TensorShapeVector& S,
                         const ConvAttributes::ConvPadVector& P,
                         const TensorShapeVector& D,
                         const int64_t group) const {
    ZoneScopedN("DepthwiseConv2D");
    VLOGS_DEFAULT(V_ATTR) << "DepthwiseConv2D, K:" << K << " S:" << S << " P:" << TensorShape{P}
                          << " D:" << D << " group:" << group;

    auto C_in = X->Shape()[1];
    auto H_in = X->Shape()[2];
    auto W_in = X->Shape()[3];
    auto shape = Y->Shape();
    auto N = shape[0];
    auto C_out = shape[1];
    auto H_out = shape[2];
    auto W_out = shape[3];
    ORT_RETURN_IF(C_in != C_out, "depthwise conv2d requires the same number of in channel and out channel");
    uint32_t gsx = gsl::narrow_cast<uint32_t>(CeilDiv(C_out, 4) * CeilDiv(W_out, 4));
    uint32_t gsy = gsl::narrow_cast<uint32_t>(N * H_out);

    bool S1 = S[0] == 1 && S[1] == 1 && D[0] == 1 && D[1] == 1;

    if (S1) {
      ZoneScopedN("DepthwiseConv2DS1 (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(kernel_name::DepthwiseConv2DS1)}
              .SetArg<cl_int>(gsx)
              .SetArg<cl_int>(gsy)
              .SetImage2Ds(*X, static_cast<cl_mem>(packed_weight_.get()), (B ? *B : *X), (Sum ? *Sum : *X), *Y)
              .SetInt2(W_in, H_in)
              .SetInt2(W_out, H_out)
              .SetInt2(K[0], K[1])
              .SetInt2(P[0], P[1])
              .SetArg<cl_int>(B != nullptr)
              .SetArg<cl_int>(Sum != nullptr)
              .SetArg<cl_int>(act_info_.ActivationKind)
              .SetArg<cl_float>(act_info_.Parameters.Values[0])
              .SetArg<cl_float>(act_info_.Parameters.Values[1])
              .Launch(*exec_, {gsx, gsy}));
    } else {
      ZoneScopedN("DepthwiseConv2D (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(kernel_name::DepthwiseConv2D)}
              .SetArg<cl_int>(gsx)
              .SetArg<cl_int>(gsy)
              .SetImage2Ds(*X, static_cast<cl_mem>(packed_weight_.get()), (B ? *B : *X), (Sum ? *Sum : *X), *Y)
              .SetInt2(W_in, H_in)
              .SetInt2(W_out, H_out)
              .SetInt2(K[0], K[1])
              .SetInt2(S[0], S[1])
              .SetInt2(P[0], P[1])
              .SetInt2(D[0], D[1])
              .SetArg<cl_int>(B != nullptr)
              .SetArg<cl_int>(Sum != nullptr)
              .SetArg<cl_int>(act_info_.ActivationKind)
              .SetArg<cl_float>(act_info_.Parameters.Values[0])
              .SetArg<cl_float>(act_info_.Parameters.Values[1])
              .Launch(*exec_, {gsx, gsy}));
    }

    return Status::OK();
  }

  Status WinogradConv2D(const Tensor* X,
                        const Tensor* B,
                        const Tensor* Sum,
                        Tensor* Y,
                        const ConvAttributes::ConvPadVector& P) const {
    ZoneScopedN("WinogradConv2D");
    VLOGS_DEFAULT(V_ATTR) << "WinogradConv2D, K:" << TensorShape{3, 3} << " S:" << TensorShape{1, 1}
                          << " P:" << TensorShape{P} << " D:" << TensorShape{1, 1} << " group:" << 1;

    const auto& xshape = X->Shape();
    const auto& yshape = Y->Shape();
    const auto output_channel = yshape[1];
    const auto output_height = yshape[2];
    const auto output_width = yshape[3];

    const auto batch = yshape[0];
    const auto input_channel = xshape[1];
    const auto input_height = xshape[2];
    const auto input_width = xshape[3];

    const auto round_up_ouptut_width = CeilDiv(output_width, 2);
    const auto round_up_output_height = CeilDiv(output_height, 2);
    const auto batch_round_h = batch * round_up_output_height;
    const auto output_channel_blocks = CeilDiv(output_channel, 4);
    const auto input_channel_blocks = CeilDiv(input_channel, 4);
    const auto round_up_4x4_ouptut_width = CeilDiv(round_up_ouptut_width, 4);

    opencl::Image2DDesc desc{input_channel_blocks * round_up_ouptut_width, 16 * batch * round_up_output_height};
    auto ocl_v_ = exec_->GetScratchImage2D(desc);
    desc = {output_channel_blocks * round_up_ouptut_width, 16 * batch * round_up_output_height};
    auto ocl_m_ = exec_->GetScratchImage2D(desc);
    std::vector<KernelUnitParam> winokernel(3);
    ORT_RETURN_IF_ERROR(CalWGSizeForWino(X, Y, winokernel));
    ORT_RETURN_IF_ERROR(KernelLauncher{GetKernel(kernel_name::TransformToMatrixV)}
                            .SetArg<cl_int>(winokernel[0].global_work_size[0])
                            .SetArg<cl_int>(winokernel[0].global_work_size[1])
                            .SetImage2Ds(*X, ocl_v_.get())
                            .SetInt2(input_height, input_width)
                            .SetArg<cl_int>(input_channel)
                            .SetArg<cl_int>(round_up_output_height)
                            .SetArg<cl_int>(round_up_ouptut_width)
                            .SetInt2(P[0], P[1])
                            .Launch(*exec_, {winokernel[0].global_work_size[0], winokernel[0].global_work_size[1]}));
    ORT_RETURN_IF_ERROR(KernelLauncher{GetKernel(kernel_name::MatrixInnerProduct)}
                            .SetArg<cl_int>(winokernel[1].global_work_size[0])
                            .SetArg<cl_int>(winokernel[1].global_work_size[1])
                            .SetImage2Ds(ocl_v_.get(), static_cast<cl_mem>(packed_weight_.get()), ocl_m_.get())
                            .SetArg<cl_int>(round_up_ouptut_width)
                            .SetArg<cl_int>(round_up_4x4_ouptut_width)
                            .SetArg<cl_int>(batch_round_h)
                            .SetArg<cl_int>(output_channel_blocks)
                            .SetArg<cl_int>(input_channel_blocks)
                            .Launch(*exec_, {winokernel[1].global_work_size[0], winokernel[1].global_work_size[1]}));
    ORT_RETURN_IF_ERROR(KernelLauncher{GetKernel(kernel_name::TransformFromMatrixM)}
                            .SetArg<cl_int>(winokernel[2].global_work_size[0])
                            .SetArg<cl_int>(winokernel[2].global_work_size[1])
                            .SetImage2Ds(ocl_m_.get(), (B ? *B : *X), (Sum ? *Sum : *X), *Y)
                            .SetArg<cl_int>(round_up_ouptut_width)
                            .SetArg<cl_int>(round_up_output_height)
                            .SetArg<cl_int>(output_width)
                            .SetArg<cl_int>(output_height)
                            .SetArg<cl_int>(act_info_.ActivationKind)
                            .SetArg<cl_float>(act_info_.Parameters.Values[0])
                            .SetArg<cl_float>(act_info_.Parameters.Values[1])
                            .SetArg<cl_int>(B != nullptr)
                            .SetArg<cl_int>(Sum != nullptr)
                            .Launch(*exec_, {winokernel[2].global_work_size[0], winokernel[2].global_work_size[1]}));
    return Status::OK();
  }
  Status Conv2D(const Tensor* X,
                const Tensor* B,
                const Tensor* Sum,
                Tensor* Y,
                const TensorShapeVector& K,
                const TensorShapeVector& S,
                const ConvAttributes::ConvPadVector& P,
                const TensorShapeVector& D,
                const int64_t group) const {
    ZoneScopedN("Conv2D");
    VLOGS_DEFAULT(V_ATTR) << "Conv2D, K:" << K << " S:" << S << " P:" << TensorShape{P}
                          << " D:" << D << " group:" << group;
    ORT_RETURN_IF(group != 1, "group != 1 is not supported currently in Conv2D");

    const auto& xshape = X->Shape();
    const auto& yshape = Y->Shape();

    auto C_in = xshape[1];
    auto H_in = xshape[2];
    auto W_in = xshape[3];
    auto N = yshape[0];
    auto C_out = yshape[1];
    auto H_out = yshape[2];
    auto W_out = yshape[3];
    uint32_t gsx = static_cast<uint32_t>(CeilDiv(C_out, 4) * CeilDiv(W_out, 4));
    uint32_t gsy = static_cast<uint32_t>(N * H_out);

    bool K1 = K[0] == 1 && K[1] == 1 && P[0] == 0 && P[1] == 0;
    bool S1 = S[0] == 1 && S[1] == 1 && D[0] == 1 && D[1] == 1;
    if (K1 && S1) {
      ZoneScopedN("Conv2DK1S1 (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(kernel_name::Conv2DK1S1)}
              .SetArg<cl_int>(gsx)
              .SetArg<cl_int>(gsy)
              .SetImage2Ds(*X, static_cast<cl_mem>(packed_weight_.get()), (B ? *B : *X), (Sum ? *Sum : *X), * Y)
              .SetInt2(W_in, H_in)
              .SetArg<cl_int>(CeilDiv(C_in, 4))
              .SetArg<cl_int>(CeilDiv(W_out, 4))
              .SetArg<cl_int>(B != nullptr)
              .SetArg<cl_int>(Sum != nullptr)
              .SetArg<cl_int>(act_info_.ActivationKind)
              .SetArg<cl_float>(act_info_.Parameters.Values[0])
              .SetArg<cl_float>(act_info_.Parameters.Values[1])
              .Launch(*exec_, {gsx, gsy}));
    } else if (K1) {
      ZoneScopedN("Conv2DK1 (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(kernel_name::Conv2DK1)}
              .SetArg<cl_int>(gsx)
              .SetArg<cl_int>(gsy)
              .SetImage2Ds(*X, static_cast<cl_mem>(packed_weight_.get()), (B ? *B : *X), (Sum ? *Sum : *X), *Y)
              .SetInt2(W_in, H_in)
              .SetArg<cl_int>(CeilDiv(C_in, 4))
              .SetInt2(W_out, H_out)
              .SetInt2(S[0], S[1])
              .SetArg<cl_int>(CeilDiv(W_out, 4))
              .SetArg<cl_int>(B != nullptr)
              .SetArg<cl_int>(Sum != nullptr)
              .SetArg<cl_int>(act_info_.ActivationKind)
              .SetArg<cl_float>(act_info_.Parameters.Values[0])
              .SetArg<cl_float>(act_info_.Parameters.Values[1])
              .Launch(*exec_, {gsx, gsy}));
    } else {
      ZoneScopedN("Conv2D (kernel launch)");
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(kernel_name::Conv2D)}
              .SetArg<cl_int>(gsx)
              .SetArg<cl_int>(gsy)
              .SetImage2Ds(*X, static_cast<cl_mem>(packed_weight_.get()), (B ? *B : *X), (Sum ? *Sum : *X), *Y)
              .SetInt2(W_in, H_in)
              .SetArg<cl_int>(CeilDiv(C_in, 4))
              .SetInt2(W_out, H_out)
              .SetInt2(K[0], K[1])
              .SetInt2(S[0], S[1])
              .SetInt2(P[0], P[1])
              .SetInt2(D[0], D[1])
              .SetArg<cl_int>(CeilDiv(W_out, 4))
              .SetArg<cl_int>(B != nullptr)
              .SetArg<cl_int>(Sum != nullptr)
              .SetArg<cl_int>(act_info_.ActivationKind)
              .SetArg<cl_float>(act_info_.Parameters.Values[0])
              .SetArg<cl_float>(act_info_.Parameters.Values[1])
              .Launch(*exec_, {gsx, gsy}));
    }
    return Status::OK();
  }

  ConvAttributes attrs_;
  MLAS_ACTIVATION act_info_;
  ConvKind conv_kind_;
  TensorShape W_shape_;
  IAllocatorUniquePtrToClMem packed_weight_;

  cl_mem GetPackedWeight() const {
    return packed_weight_.get();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1, 10,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // conv kernel weight will be handled via PrePack
    Conv);

ONNX_OPENCL_OPERATOR_KERNEL(
    Conv,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // conv kernel weight will be handled via PrePack
    Conv);

ONNX_OPERATOR_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    kOpenCLExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // conv kernel weight will be handled via PrePack
    Conv                                          // register the Conv OpKernel as the FusedConv impl
);

}  // namespace opencl
}  // namespace onnxruntime
