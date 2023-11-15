// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"

#include "core/common/gsl.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/xnnpack_init.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
namespace xnnpack {

// use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
Status Conv::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                     /*out*/ bool& is_packed,
                     /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  // only layout of weight input is adjusted via PrePack
  if ((conv_type_ == OpComputeType::op_compute_type_fp32 && input_idx == 1) ||
      (conv_type_ != OpComputeType::op_compute_type_fp32 && input_idx == 3)) {  // InputTensors::IN_W
    // Transpose from {M, C/group, kH, kW} to {M, kH, kW, C/group}
    auto orig_shape = tensor.Shape();

    InlinedVector<size_t> perm{0, 2, 3, 1};
    TensorShapeVector new_dims{orig_shape[0],
                               orig_shape[2],
                               orig_shape[3],
                               orig_shape[1]};

    packed_w_ = Tensor(tensor.DataType(), TensorShape(new_dims), std::move(alloc));

    SingleAxisTranspose(perm, tensor, packed_w_, /*from*/ 1, /*to*/ 3);

    is_packed = true;

    // we can create the kernel now
    ORT_RETURN_IF_ERROR(CreateKernel());
  }

  return Status::OK();
}

Status Conv::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // this is in NHWC format
  const auto& X_shape = X.Shape();
  const int64_t N = X_shape[0];  // input is NHWC
  const int64_t H = X_shape[1];
  const int64_t W = X_shape[2];

  // We don't need to call ValidateInputShape as we checked validity in ConvChecker.
  // We also can't use ValidateInputShape as-is as the weight tensor was pre-packed and the layout was changed there.
  // ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&X, &W));

  // CPU Conv starts with TensorShapeVector Y_dims({N, M}); and passes in X->Shape().Slice(2);
  // We know this is 2D in NHWC format so we need to start with 'N', pass in the H, W, and append M last
  TensorShapeVector Y_dims(output_shape_);
  Y_dims[0] = N;
  Tensor* Y = context->Output(0, TensorShape(Y_dims));

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  pthreadpool_t threadpool = GetThreadPool();

  // setup allocator/automated dellocate for workspace
  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  xnn_allocator* allocator = GetStoredAllocator().second;
  auto deallocator = [allocator](void* ptr) { allocator->aligned_deallocate(allocator->context, ptr); };
  std::unique_ptr<void, decltype(deallocator)> workspace(nullptr, deallocator);

  auto reshape_fn = xnn_reshape_convolution2d_nhwc_f32;
  if (conv_type_ == OpComputeType::op_compute_type_qs8) {
    reshape_fn = xnn_reshape_convolution2d_nhwc_qs8;
  } else if (conv_type_ == OpComputeType::op_compute_type_qu8) {
    reshape_fn = xnn_reshape_convolution2d_nhwc_qu8;
  } else if (conv_type_ == OpComputeType::op_compute_type_qs8_per_channel) {
    reshape_fn = xnn_reshape_convolution2d_nhwc_qs8_qc8w;
  }

  auto status = reshape_fn(op0_.get(), N, H, W,
                           &workspace_size, &workspace_alignment,
                           /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                           threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_reshape_convolution2d_nhwc_", OpTypeToString(conv_type_),
                           "returned ", status);
  }

  workspace.reset(allocator->aligned_allocate(allocator->context, XNN_ALLOCATION_ALIGNMENT, workspace_size));

  if (conv_type_ == OpComputeType::op_compute_type_fp32) {
    status = xnn_setup_convolution2d_nhwc_f32(op0_.get(), workspace.get(), X.Data<float>(),
                                              Y->MutableData<float>());
  } else if (conv_type_ == OpComputeType::op_compute_type_qs8) {
    status = xnn_setup_convolution2d_nhwc_qs8(op0_.get(), workspace.get(), X.Data<int8_t>(),
                                              Y->MutableData<int8_t>());
  } else if (conv_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_convolution2d_nhwc_qu8(op0_.get(), workspace.get(), X.Data<uint8_t>(),
                                              Y->MutableData<uint8_t>());
  } else if (conv_type_ == OpComputeType::op_compute_type_qs8_per_channel) {
    status = xnn_setup_convolution2d_nhwc_qs8_qc8w(op0_.get(), workspace.get(), X.Data<int8_t>(),
                                                   Y->MutableData<int8_t>());
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_convolution2d_nhwc_",
                           OpTypeToString(conv_type_), "returned ", status);
  }

  status = xnn_run_operator(op0_.get(), threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Conv, kMSInternalNHWCDomain, 1, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  Conv);

ONNX_OPERATOR_KERNEL_EX(Conv, kMSInternalNHWCDomain, 11, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Conv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSInternalNHWCDomain,
    10,
    uint8_t,
    kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    Conv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSInternalNHWCDomain,
    10,
    int8_t,
    kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    Conv);
}  // namespace xnnpack
}  // namespace onnxruntime
