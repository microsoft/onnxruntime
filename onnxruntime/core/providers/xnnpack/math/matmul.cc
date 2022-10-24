// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace xnnpack {

bool MatMul::IsMatMulOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
  bool supported = false;
  const onnxruntime::Node& node = node_unit.GetNode();

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    auto input_defs = node.InputDefs();

    const auto& A_arg = *input_defs[0];
    const auto& B_arg = *input_defs[1];

    // Support only float
    const auto* A_type = A_arg.TypeAsProto();
    const auto* B_type = B_arg.TypeAsProto();

    const auto* A_shape = A_arg.Shape();
    const auto* B_shape = B_arg.Shape();

    if (A_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
        B_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      break;
    }

    if (!A_shape || A_shape->dim_size() != 2 ||
        (A_shape->dim(0).dim_value() != 1 && A_shape->dim(1).dim_value() != 1)) {
      break;
    }

    if (!B_shape || B_shape->dim_size() != 2 || 
        (B_shape->dim(0).dim_value() != 1 && B_shape->dim(1).dim_value() != 1)) {
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
  info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
  info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
  info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
}

Status MatMul::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                       /*out*/ bool& is_packed,
                       /*out*/ PrePackedWeights* /*Not used*/) {
  is_packed = false;

  if (input_idx == 0 || input_idx == 2) {
    return Status::OK();
  }

  myAlloc = alloc;

  is_packed = true;

  uint32_t flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
  float output_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
  float output_max = clip_min_max_ ? clip_min_max_->second : INFINITY;
  xnn_status status = xnn_status::xnn_status_uninitialized;

  struct xnn_operator* p = nullptr;
  b_shape_ = tensor.Shape();
  status = xnn_create_fully_connected_nc_f32(
      tensor.Shape()[0],     // size_t input_channels,
      tensor.Shape()[1],     // size_t output_channels,
      tensor.Shape()[0],     // size_t input_stride,
      tensor.Shape()[1],     // size_t output_stride,
      tensor.Data<float>(),  // const float* kernel,
      nullptr,               // const float* bias,
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

  op0_.reset(p);

  return Status::OK();
}

Status MatMul::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  pthreadpool_t t_pool = GetThreadPool();
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();

  xnn_status status = xnn_setup_fully_connected_nc_f32(
      op0_.get(),
      a->Shape()[0],
      a->Data<float>(),
      y_data,
      t_pool);
  
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_fully_connected_nc_f32 returned ", status);
  }

  status = xnn_run_operator(op0_.get(), nullptr);
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