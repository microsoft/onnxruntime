// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/cpuid_info.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include <xnnpack.h>

#ifdef PROFILE
#include <iostream>
#include <chrono>
#endif

namespace onnxruntime {
namespace xnnpack_ep {
class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info),
                                                   conv_attrs_(info) {
    channels_last_ = (info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(0)) != 0);
    CreateOperator(info);
  }

  Status Compute(OpKernelContext* context) const override;

  void CreateOperator(const OpKernelInfo& info) {
    const Tensor* W = nullptr;
    if (!info.TryGetConstantInput(3, &W)) return;
    const auto& W_shape = W->Shape();

    const int64_t M = W_shape[0];
    const int64_t C = W_shape[1];

    const Tensor* X_zero_point;
    const Tensor* W_zero_point;
    const Tensor* Y_zero_point;
    if (!info.TryGetConstantInput(2, &X_zero_point)) return;
    if (!info.TryGetConstantInput(5, &W_zero_point)) return;
    if (!info.TryGetConstantInput(7, &Y_zero_point)) return;

    auto X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
    auto W_zero_point_value = *(W_zero_point->template Data<uint8_t>());
    auto Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());

    const Tensor* X_scale;
    const Tensor* W_scale;
    const Tensor* Y_scale;
    const Tensor* B;
    if (!info.TryGetConstantInput(1, &X_scale)) return;
    if (!info.TryGetConstantInput(4, &W_scale)) return;
    if (!info.TryGetConstantInput(6, &Y_scale)) return;
    if (!info.TryGetConstantInput(8, &B)) return;

    auto X_scale_value = *(X_scale->template Data<float>());
    auto W_scale_value = *(W_scale->template Data<float>());
    auto Y_scale_value = *(Y_scale->template Data<float>());

    std::vector<int64_t> kernel_shape;
    ORT_ENFORCE(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape).IsOK());

    const size_t kernel_rank = kernel_shape.size();

    std::vector<int64_t> pads(conv_attrs_.pads);
    if (pads.empty()) {
      pads.resize(kernel_rank * 2, 0);
    }
    std::vector<int64_t> dilations(conv_attrs_.dilations);
    if (dilations.empty()) {
      dilations.resize(kernel_rank, 1);
    }
    std::vector<int64_t> strides(conv_attrs_.strides);
    if (strides.empty()) {
      strides.resize(kernel_rank, 1);
    }

    int64_t group_count = conv_attrs_.group;
    int64_t group_input_channels = C;
    int64_t group_output_channels = M / group_count;

    const auto* Wdata = W->Data<uint8_t>();
    const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;

    auto status = xnn_create_convolution2d_nhwc_qu8(
        static_cast<uint32_t>(pads[0]),
        static_cast<uint32_t>(pads[3]),
        static_cast<uint32_t>(pads[2]),
        static_cast<uint32_t>(pads[1]),
        static_cast<uint32_t>(kernel_shape[0]),
        static_cast<uint32_t>(kernel_shape[1]),
        static_cast<uint32_t>(strides[0]),
        static_cast<uint32_t>(strides[1]),
        static_cast<uint32_t>(dilations[0]),
        static_cast<uint32_t>(dilations[1]),
        static_cast<uint32_t>(group_count),
        static_cast<size_t>(group_input_channels),
        static_cast<size_t>(group_output_channels),
        static_cast<size_t>(C * group_count),
        static_cast<size_t>(M),
        X_zero_point_value, X_scale_value,
        W_zero_point_value, W_scale_value,
        Wdata, Bdata,
        Y_zero_point_value, Y_scale_value,
        0, 255,
        0 /* flags */,
        &conv_op_);
    ORT_ENFORCE(status == xnn_status_success);
  }

 private:
  ConvAttributes conv_attrs_;
  bool channels_last_;
  xnn_operator_t conv_op_{nullptr};
};

Status QLinearConv::Compute(OpKernelContext* context) const {
#ifdef PROFILE
  auto start = std::chrono::steady_clock::now();
#endf

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(3);
  const auto& W_shape = W->Shape();

  const int64_t N = X->Shape()[0];
  const int64_t M = W_shape[0];
  const int64_t C = W_shape[1];

  const Tensor* X_zero_point = context->Input<Tensor>(2);
  const Tensor* W_zero_point = context->Input<Tensor>(5);
  const Tensor* Y_zero_point = context->Input<Tensor>(7);

  auto X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
  auto W_zero_point_value = *(W_zero_point->template Data<uint8_t>());
  auto Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());

  const Tensor* X_scale = context->Input<Tensor>(1);
  const Tensor* W_scale = context->Input<Tensor>(4);
  const Tensor* Y_scale = context->Input<Tensor>(6);

  auto X_scale_value = *(X_scale->template Data<float>());
  auto W_scale_value = *(W_scale->template Data<float>());
  auto Y_scale_value = *(Y_scale->template Data<float>());

  const Tensor* B = context->Input<Tensor>(8);

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W_shape, channels_last_));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));

  const size_t kernel_rank = kernel_shape.size();

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
  }

  const size_t spatial_dim_start = channels_last_ ? 1 : 2;
  const size_t spatial_dim_end = spatial_dim_start + kernel_rank;

  std::vector<int64_t> Y_dims({N});
  if (!channels_last_) {
    Y_dims.push_back(M);
  }
  TensorShape input_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  if (channels_last_) {
    Y_dims.push_back(M);
  }
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(spatial_dim_start, spatial_dim_end);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  int64_t group_count = conv_attrs_.group;
  int64_t group_input_channels = C;
  int64_t group_output_channels = M / group_count;

  const auto* Xdata = X->Data<uint8_t>();
  const auto* Wdata = W->Data<uint8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->MutableData<uint8_t>();

  xnn_operator_t conv_op = conv_op_;
  if (conv_op == nullptr) {
    auto status = xnn_create_convolution2d_nhwc_qu8(
        static_cast<uint32_t>(pads[0]),
        static_cast<uint32_t>(pads[3]),
        static_cast<uint32_t>(pads[2]),
        static_cast<uint32_t>(pads[1]),
        static_cast<uint32_t>(kernel_shape[0]),
        static_cast<uint32_t>(kernel_shape[1]),
        static_cast<uint32_t>(strides[0]),
        static_cast<uint32_t>(strides[1]),
        static_cast<uint32_t>(dilations[0]),
        static_cast<uint32_t>(dilations[1]),
        static_cast<uint32_t>(group_count),
        static_cast<size_t>(group_input_channels),
        static_cast<size_t>(group_output_channels),
        static_cast<size_t>(C * group_count),
        static_cast<size_t>(M),
        X_zero_point_value, X_scale_value,
        W_zero_point_value, W_scale_value,
        Wdata, Bdata,
        Y_zero_point_value, Y_scale_value,
        0, 255,
        group_input_channels == 1 && group_output_channels == 1 ? XNN_FLAG_DEPTHWISE_CONVOLUTION : 0 /* flags */,
        &conv_op);
    ORT_ENFORCE(status == xnn_status_success);
  }

#ifdef PROFILE
  auto start_setup = std::chrono::steady_clock::now();
#endif
  ORT_ENFORCE(xnn_status_success == xnn_setup_convolution2d_nhwc_qu8(conv_op, N, input_shape[0], input_shape[1], Xdata, Ydata, nullptr));

#ifdef PROFILE
  auto start_run = std::chrono::steady_clock::now();
#endif

  ORT_ENFORCE(xnn_status_success == xnn_run_operator(conv_op, nullptr));
  
#ifdef PROFILE
  auto end = std::chrono::steady_clock::now();
  std::cout << Node().Name()
            << "C:" << std::chrono::duration<double, std::milli>(start_setup - start).count()
            << ", S:" << std::chrono::duration<double, std::milli>(start_run - start_setup).count()
            << ", R:" << std::chrono::duration<double, std::milli>(end - start_run).count() << std::endl;
#endif

  return Status::OK();
}
}  // namespace xnnpack_ep

// Register an alternate version of this kernel that supports the channels_last
// attribute in order to consume and produce NHWC tensors.
ONNX_OPERATOR_KERNEL_EX(
    QLinearConv,
    kMSDomain,
    1,
    kXNNPackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    xnnpack_ep::QLinearConv);

}  // namespace onnxruntime
