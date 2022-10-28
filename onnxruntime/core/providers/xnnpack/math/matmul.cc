// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace xnnpack {

bool MatMul::IsMatMulOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
  bool supported = false;
  const onnxruntime::Node& node = node_unit.GetNode();

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    auto input_defs = node.InputDefs();

    if (input_defs.size() != 2) {
      break;
    }

    const auto& A_arg = *input_defs[0];
    const auto& B_arg = *input_defs[1];

    // Support only float
    const auto* A_type = A_arg.TypeAsProto();

    const auto* A_shape = A_arg.Shape();
    const auto* B_shape = B_arg.Shape();

    if (A_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      break;
    }

    if (A_shape->dim_size() != 2 || A_shape->dim(1).dim_value() == 0 || A_shape->dim(0).dim_value() == 0) {
      break;
    }

    if (B_shape->dim_size() >= 2 || B_shape->dim(1).dim_value()==0 || B_shape->dim(0).dim_value()==0) {
      break;
    }

    // B matrix must be constant
    if (!graph.IsConstantInitializer(B_arg.Name(), true)) {
      break;
    }

    supported = true;

  } while (false);

  return supported;
}

MatMul::MatMul(const OpKernelInfo& info) : XnnpackKernel(info) {
  int64_t temp;
  info.GetAttrOrDefault<int64_t>("transA", &temp, 0);
  trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;

  info.GetAttrOrDefault<int64_t>("transB", &temp, 0);
  trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;
}

Status MatMul::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);

  const bool trans_a = trans_A_ && a->Shape().NumDimensions() != 1;
  const bool trans_b = trans_B_ && b->Shape().NumDimensions() != 1;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), !trans_a, !trans_b));

  pthreadpool_t t_pool = GetThreadPool();
  
  uint32_t flags = trans_B_ == CblasNoTrans ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0;
  float output_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
  float output_max = clip_min_max_ ? clip_min_max_->second : INFINITY;

  const size_t M = std::max(static_cast<size_t>(helper.M()), (size_t)1);
  const size_t N = std::max(static_cast<size_t>(helper.N()), (size_t)1);
  const size_t K = std::max(static_cast<size_t>(helper.K()), (size_t)1);
   
  xnn_status status = xnn_status::xnn_status_uninitialized;
  struct xnn_operator* p = nullptr;
  status = xnn_create_fully_connected_nc_f32(
      K,  // size_t input_channels,
      N,  // size_t output_channels,
      K,  // size_t input_stride,
      N,  // size_t output_stride,
      b->Data<float>(),                                  // const float* kernel,
      nullptr,                                               // const float* bias,
      output_min,
      output_max,
      flags,
#ifdef XNN_CACHE_ENABLE
      &xnn_caches_,
#else
      0,
#endif
      &p);

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_create_fully_connected_nc_f32 returned ", status);
  }

  Tensor* y = ctx->Output(0, helper.OutputShape());

  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();

  status = xnn_setup_fully_connected_nc_f32(
      p,
      M,
      a->Data<float>(),
      y_data,
      t_pool);
  
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_fully_connected_nc_f32 returned ", status);
  }

  status = xnn_run_operator(p, nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }
  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(MatMul, kOnnxDomain, 1, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  MatMul);

ONNX_OPERATOR_KERNEL_EX(MatMul, kOnnxDomain, 13, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        MatMul);

}  // namespace xnnpack
}  // namespace onnxruntime