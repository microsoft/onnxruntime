// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"

// Todo -
// 1. Integrate activation layers - Cliping & Relu
// 2. Enable Quant ops
// 3. Review possible consolidation of MatMul & Gemm
//

namespace onnxruntime {
namespace xnnpack {

bool MatMul::IsOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
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

    if (A_shape == nullptr || A_shape->dim_size() > 2 ||
        (A_shape->dim_size() == 2 && A_shape->dim(1).dim_value() == 0) ||
        A_shape->dim(0).dim_value() == 0) {
      break;
    }

    if (B_shape == nullptr || B_shape->dim_size() > 2 ||
        (B_shape->dim_size() == 2 && B_shape->dim(1).dim_value() == 0) ||
        B_shape->dim(0).dim_value() == 0) {
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

MatMul::MatMul(const OpKernelInfo& info) : XnnpackKernel(info, /*enable_caches*/ true) {}

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
  float output_min = -INFINITY;
  float output_max = INFINITY;
  xnn_status status = xnn_status::xnn_status_uninitialized;

  struct xnn_operator* p = nullptr;
  b_shape_ = tensor.Shape();
  auto shape_broadcast = b_shape_.AsShapeVector();
  if (b_shape_.NumDimensions() == 1) {
    shape_broadcast.push_back(1);
  }
  status = xnn_create_fully_connected_nc_f32(
      shape_broadcast[0],    // size_t input_channels,
      shape_broadcast[1],    // size_t output_channels,
      shape_broadcast[0],    // size_t input_stride,
      shape_broadcast[1],    // size_t output_stride,
      tensor.Data<float>(),  // const float* kernel,
      nullptr,               // const float* bias,
      output_min,
      output_max,
      flags,
#ifdef XNN_CACHE_ENABLE
      GetCodeCache(),
      GetWeightsCache(),
#else
      nullptr,
      nullptr,
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
  pthreadpool_t threadpool = GetThreadPool();
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();

  xnn_status status = xnn_reshape_fully_connected_nc_f32(op0_.get(), a->Shape()[0], threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_reshape_fully_connected_nc_f32 returned ", status);
  }

  status = xnn_setup_fully_connected_nc_f32(op0_.get(), a->Data<float>(), y_data);
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
