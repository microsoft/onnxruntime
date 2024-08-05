// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_int.h"
#include "core/providers/cpu/math/matmul_helper.h"

// Todo -
// 1. Integrate activation layers - Cliping & Relu
// 2. Enable Quant ops
// 3. Review possible consolidation of MatMul & Gemm
//

namespace onnxruntime {
namespace xnnpack {

bool MatMulIntegerCommon::IsOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
  bool supported = false;
  const onnxruntime::Node& node = node_unit.GetNode();

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    auto input_defs = node.InputDefs();

    if (input_defs.size() < 2) {
      break;
    }

    const auto& A_arg = *input_defs[0];
    const auto& B_arg = *input_defs[1];

    const auto* A_shape = A_arg.Shape();
    const auto* B_shape = B_arg.Shape();

    if (A_shape == nullptr || B_shape == nullptr) {
      break;
    }

    size_t A_rank = A_shape->dim_size();
    size_t B_rank = B_shape->dim_size();

    // Support A [M, K] or [batch, M, K] x B [K, N] or [N]
    if (B_rank > 2 || (A_rank != B_rank && A_rank != B_rank + 1)) {
      break;
    }

    if (B_shape->dim(0).dim_value() == 0) {
      break;
    }

    if (B_rank == 2 && B_shape->dim(1).dim_value() == 0) {
      break;
    }

    // B matrix must be constant
    if (!graph.IsConstantInitializer(B_arg.Name(), true)) {
      break;
    }

    // b_zero_point must be constant
    if (input_defs.size() > 3) {
      if (!graph.IsConstantInitializer(input_defs[3]->Name(), true)) {
        break;
      }
    }

    supported = true;

  } while (false);

  return supported;
}

template<>
Status MatMulInteger<int8_t>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                       /*out*/ bool& is_packed,
                       /*out*/ PrePackedWeights* /*Not used*/) {
  is_packed = false;

  if (input_idx == 0) {
    return Status::OK();
  }

  if (input_idx == 1) {
    b_shape_ = tensor.Shape();
    B_ = &tensor;
  }

  if (input_idx == 2 && has_a_zero_point_) {
    a_zero_point_ = tensor.Data<int8_t>()[0];
  }

  if (input_idx == 3 && has_a_zero_point_) {
    b_zero_point_ = tensor.Data<int8_t>()[0];
  }

  if ((!has_a_zero_point_ || input_idx >= 2) && (!has_b_zero_point_ || input_idx >= 3)) {
    myAlloc = alloc;

    uint32_t flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
    xnn_status status = xnn_status::xnn_status_uninitialized;
    struct xnn_operator* p = nullptr;
    auto shape_broadcast = b_shape_.AsShapeVector();
    if (b_shape_.NumDimensions() == 1) {
      shape_broadcast.push_back(1);
    }
    status = xnn_create_fully_connected_nc_qs8(
      shape_broadcast[0],    // size_t input_channels,
      shape_broadcast[1],    // size_t output_channels,
      shape_broadcast[0],    // size_t input_stride,
      shape_broadcast[1],    // size_t output_stride,
      a_zero_point_,         // int8_t input_zero_point,
      1.0f,                  // float input_scale,
      1.0f,                  // float kernel_scale,
      B_->Data<int8_t>(),    // const int8_t* kernel,
      nullptr,               // const int32_t* bias,
      0,                     // int8_t output_zero_point,
      1.0f,                  // float output_scale,
      INT8_MIN,
      INT8_MAX,
      flags,
#ifdef XNN_CACHE_ENABLE
      GetCodeCache(),
      GetWeightsCache(),
#else
      nullptr,
      nullptr,
#endif
      &p);

    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_create_fully_connected_nc_qs8 returned ", status);

    op0_.reset(p);
  }

  is_packed = true;

  return Status::OK();
}

template<>
Status MatMulInteger<uint8_t>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                       /*out*/ bool& is_packed,
                       /*out*/ PrePackedWeights* /*Not used*/) {
  is_packed = false;

  if (input_idx == 0) {
    return Status::OK();
  }

  if (input_idx == 1) {
    b_shape_ = tensor.Shape();
    B_ = &tensor;
  }

  if (input_idx == 2 && has_a_zero_point_) {
    a_zero_point_ = tensor.Data<uint8_t>()[0];
  }

  if (input_idx == 3 && has_a_zero_point_) {
    b_zero_point_ = tensor.Data<uint8_t>()[0];
  }

  if (input_idx >= 1 && (!has_a_zero_point_ || input_idx >= 2) && (!has_b_zero_point_ || input_idx >= 3)) {
    myAlloc = alloc;

    uint32_t flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
    xnn_status status = xnn_status::xnn_status_uninitialized;

    struct xnn_operator* p = nullptr;
    auto shape_broadcast = b_shape_.AsShapeVector();
    if (b_shape_.NumDimensions() == 1) {
      shape_broadcast.push_back(1);
    }
    status = xnn_create_fully_connected_nc_qu8(
      shape_broadcast[0],    // size_t input_channels,
      shape_broadcast[1],    // size_t output_channels,
      shape_broadcast[0],    // size_t input_stride,
      shape_broadcast[1],    // size_t output_stride,
      a_zero_point_,         // uint8_t input_zero_point,
      1.0f,                  // float input_scale,
      b_zero_point_,         // uint8_t kernel_zero_point,
      1.0f,                  // float kernel_scale,
      B_->Data<uint8_t>(),   // const uint8_t* kernel,
      nullptr,               // const int32_t* bias,
      0,                     // uint8_t output_zero_point,
      1.0f,                  // float output_scale,
      0,
      UINT8_MAX,
      flags,
#ifdef XNN_CACHE_ENABLE
      GetCodeCache(),
      GetWeightsCache(),
#else
      nullptr,
      nullptr,
#endif
      &p);

    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_create_fully_connected_nc_qu8 returned ", status);

    op0_.reset(p);
  }

  is_packed = true;

  return Status::OK();
}

template<>
Status MatMulInteger<int8_t>::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  pthreadpool_t threadpool = GetThreadPool();
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  if (y->Shape().Size() == 0)
    return Status::OK();

  xnn_status status = xnn_status::xnn_status_uninitialized;

  auto a_shape = a->Shape();

  if (a_shape.NumDimensions() != b_shape_.NumDimensions()) {
    // A is [batch, ..., K] and B is [K, N] output is [batch, ..., N]
    size_t batch_size = a_shape[0];
    size_t M = a_shape[1];
    for (size_t i = 0; i < batch_size; i++) {
      size_t offset = i * M;

      status = xnn_reshape_fully_connected_nc_qs8(op0_.get(), M, threadpool);
      ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_reshape_fully_connected_nc_qs8 returned ", status);

      status = xnn_setup_fully_connected_nc_qs8(op0_.get(), a->Data<int8_t>() + offset, y->MutableData<int8_t>() + offset);
      ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_setup_fully_connected_nc_qs8 returned ", status);

      status = xnn_run_operator(op0_.get(), nullptr);
      ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_run_operator returned ", status);
    }
  } else {
    // A is [M, K] and B is [K, N]
    status = xnn_reshape_fully_connected_nc_qs8(op0_.get(), a_shape[0], threadpool);
    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_reshape_fully_connected_nc_qs8 returned ", status);

    status = xnn_setup_fully_connected_nc_qs8(op0_.get(), a->Data<int8_t>(), y->MutableData<int8_t>());
    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_setup_fully_connected_nc_qs8 returned ", status);

    status = xnn_run_operator(op0_.get(), nullptr);
    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

template<>
Status MatMulInteger<uint8_t>::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  pthreadpool_t threadpool = GetThreadPool();
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  if (y->Shape().Size() == 0)
    return Status::OK();

  xnn_status status = xnn_status::xnn_status_uninitialized;

  auto a_shape = a->Shape();

  if (a_shape.NumDimensions() != b_shape_.NumDimensions()) {
    // A is [batch, M, K] and B is [K, N]
    size_t batch_size = a_shape[0];
    size_t M = a_shape[1];
    for (size_t i = 0; i < batch_size; i++) {
      size_t offset = i * M;

      status = xnn_reshape_fully_connected_nc_qu8(op0_.get(), M, threadpool);
      ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_reshape_fully_connected_nc_qu8 returned ", status);

      status = xnn_setup_fully_connected_nc_qu8(op0_.get(), a->Data<uint8_t>() + offset, y->MutableData<uint8_t>() + offset);
      ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_setup_fully_connected_nc_qu8 returned ", status);

      status = xnn_run_operator(op0_.get(), nullptr);
      ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_run_operator returned ", status);
    }
  } else {
    // A is [M, K] and B is [K, N]
    status = xnn_reshape_fully_connected_nc_qu8(op0_.get(), a_shape[0], threadpool);
    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_reshape_fully_connected_nc_qu8 returned ", status);

    status = xnn_setup_fully_connected_nc_qu8(op0_.get(), a->Data<uint8_t>(), y->MutableData<uint8_t>());
    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_setup_fully_connected_nc_qu8 returned ", status);

    status = xnn_run_operator(op0_.get(), nullptr);
    ORT_RETURN_IF_NOT(status == xnn_status_success, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    uint8_t,
    kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t>);

}  // namespace xnnpack
}  // namespace onnxruntime
