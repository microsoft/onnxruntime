// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_transpose.h"

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace xnnpack {

// use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
Status ConvTranspose::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                              /*out*/ bool& is_packed,
                              /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  // only layout of weight input is adjusted via PrePack
  if ((conv_type_ == OpComputeType::op_compute_type_fp32 && input_idx == 1) ||
      (conv_type_ != OpComputeType::op_compute_type_fp32 && input_idx == 3)) {  // InputTensors::IN_W
    if (conv_transpose_attrs_.group > 1) {
      // Xnnpack [G, Oc, H, W Ic/G]
      // (ref: https://github.com/google/XNNPACK/blob/ecd8311c8fd3d9ab47edbc3df5f2b5de7dabe75f/test/deconvolution-operator-tester.h#L678)
      TensorShape orig_shape =
          {conv_transpose_attrs_.group,
           tensor.Shape()[0] / conv_transpose_attrs_.group,
           tensor.Shape()[1],
           tensor.Shape()[2],
           tensor.Shape()[3]};

      InlinedVector<size_t> perm{0, 2, 3, 4, 1};
      TensorShapeVector new_dims{
          orig_shape[0],
          orig_shape[2],
          orig_shape[3],
          orig_shape[4],
          orig_shape[1]};

      packed_w_ = Tensor(tensor.DataType(), TensorShape(new_dims), std::move(alloc));
      // g I/g O H W --> g O H W I/g
      SingleAxisTranspose(perm, tensor, packed_w_, /*from*/ 1, /*to*/ 4, &orig_shape);
    } else {
      // Transpose from {M, C/group, kH, kW} to {M, kH, kW, C/group}

      auto orig_shape = tensor.Shape();
      InlinedVector<size_t> perm{1, 2, 3, 0};
      TensorShapeVector new_dims{orig_shape[1],
                                 orig_shape[2],
                                 orig_shape[3],
                                 orig_shape[0]};

      packed_w_ = Tensor(tensor.DataType(), TensorShape(new_dims), std::move(alloc));
      // I O H W --> O H W I
      SingleAxisTranspose(perm, tensor, packed_w_, /*from*/ 0, /*to*/ 3);
    }
    is_packed = true;

    // we can create the kernel now
    auto ret = CreateKernel();
    ORT_RETURN_IF_ERROR(ret);
  }

  return Status::OK();
}

Status ConvTranspose::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // this is in NHWC format
  const auto& X_shape = X.Shape();
  const int64_t N = X_shape[0];  // input is NHWC
  const int64_t H = X_shape[1];
  const int64_t W = X_shape[2];

  TensorShapeVector Y_dims(output_shape_);
  Y_dims[0] = N;

  Tensor* Y = context->Output(0, TensorShape(Y_dims));

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  pthreadpool_t threadpool = GetThreadPool();

  auto output_pad_0 = gsl::narrow_cast<uint32_t>(conv_transpose_attrs_.output_padding[0]);
  auto output_pad_1 = gsl::narrow_cast<uint32_t>(conv_transpose_attrs_.output_padding[1]);
  xnn_status status = xnn_status_invalid_state;

  auto reshape_fn = xnn_reshape_deconvolution2d_nhwc_f32;
  if (conv_type_ == OpComputeType::op_compute_type_qs8) {
    reshape_fn = xnn_reshape_deconvolution2d_nhwc_qs8;
  } else if (conv_type_ == OpComputeType::op_compute_type_qu8) {
    reshape_fn = xnn_reshape_deconvolution2d_nhwc_qu8;
  }

  status = reshape_fn(op0_.get(), N, H, W, output_pad_0, output_pad_1,
                      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                      threadpool);

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_reshape_deconvolution2d_nhwc_",
                           OpTypeToString(conv_type_), " returned ", status);
  }

  if (conv_type_ == OpComputeType::op_compute_type_fp32) {
    status = xnn_setup_deconvolution2d_nhwc_f32(op0_.get(), X.Data<float>(), Y->MutableData<float>());
  } else if (conv_type_ == OpComputeType::op_compute_type_qs8) {
    status = xnn_setup_deconvolution2d_nhwc_qs8(op0_.get(), X.Data<int8_t>(), Y->MutableData<int8_t>());
  } else if (conv_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_deconvolution2d_nhwc_qu8(op0_.get(), X.Data<uint8_t>(), Y->MutableData<uint8_t>());
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_deconvolution2d_nhwc_",
                           OpTypeToString(conv_type_), " returned ", status);
  }

  status = xnn_run_operator(op0_.get(), threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(ConvTranspose, kMSInternalNHWCDomain, 11, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint(
                            "T", DataTypeImpl::GetTensorType<float>()),
                        ConvTranspose);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ConvTranspose, kMSInternalNHWCDomain, 1, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint(
                                      "T", DataTypeImpl::GetTensorType<float>()),
                                  ConvTranspose);

ONNX_OPERATOR_KERNEL_EX(QLinearConvTranspose, kMSInternalNHWCDomain, 1, kXnnpackExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint(
                                "T1",
                                {DataTypeImpl::GetTensorType<uint8_t>(),
                                 DataTypeImpl::GetTensorType<int8_t>()}),
                        ConvTranspose);

}  // namespace xnnpack
}  // namespace onnxruntime
