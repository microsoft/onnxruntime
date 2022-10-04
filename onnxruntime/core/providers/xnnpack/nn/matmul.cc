// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul.h"
#include "core/graph/constants.h"
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

    if (input_defs.size() != 2) {
      break;
    }

    const auto& A_arg = *input_defs[0];
    const auto& B_arg = *input_defs[1];

    // Support only float
    const auto* A_type = A_arg.TypeAsProto();
    const auto* B_type = B_arg.TypeAsProto();

    if (A_type == nullptr || B_type == nullptr ||
        A_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
        B_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ) {
        break;
    }

    // B matrix must be constant
    if (B_arg.Exists() && graph.GetConstantInitializer(B_arg.Name(), true) == nullptr) {
        break;
    }

    // making sure we are dealing with MatMul
    const auto* B_shape = B_arg.Shape();

    if (!B_shape || B_shape->dim_size() > 3) {
        break;
    }

    supported = true;

  } while (false);

  return supported;
}

MatMul::MatMul(const OpKernelInfo& info) : OpKernel(info){
  info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
  info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
  info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
  int64_t trans_batch_a_attr, trans_batch_b_attr;
  info.GetAttrOrDefault<int64_t>("transBatchA", &trans_batch_a_attr, 0);
  info.GetAttrOrDefault<int64_t>("transBatchB", &trans_batch_b_attr, 0);
  trans_batch_a_ = trans_batch_a_attr != 0;
  trans_batch_b_ = trans_batch_b_attr != 0;
}

Status MatMul::PrePack(const Tensor& tensor,int input_idx, AllocatorPtr alloc,
                     /*out*/ bool& is_packed,
                     /*out*/ PrePackedWeights*) {

  is_packed = false;

  if (input_idx == 0) {
    return Status::OK();
  }

  myAlloc = alloc;

  is_packed = true;

  uint32_t flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
  float output_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
  float output_max = clip_min_max_ ? clip_min_max_->second : INFINITY;
  xnn_status status = xnn_status::xnn_status_uninitialized;

  struct xnn_operator* p = nullptr;
  this->b_shape_ = tensor.Shape();
  status = xnn_create_fully_connected_nc_f32(
      tensor.Shape()[0],          // size_t input_channels,
      tensor.Shape()[1],          // size_t output_channels,
      tensor.Shape()[0],          // size_t input_stride,
      tensor.Shape()[1],          // size_t output_stride,
      tensor.Data<float>(),       // const float* kernel,
      nullptr,                    // const float* bias,
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
  const auto& b_shape = b_shape_;
  const bool trans_a = trans_a_attr_ && a->Shape().NumDimensions() != 1;
  const bool trans_b = trans_b_attr_ && b_shape.NumDimensions() != 1;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, trans_a, trans_b, trans_batch_a_, trans_batch_b_));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->MutableData<float>();
  std::unique_ptr<Tensor> packed_w_;

  const size_t max_len = a->Shape().NumDimensions() > 2 ? a->Shape()[1] : 1;

  xnn_status status = xnn_setup_fully_connected_nc_f32(
      op0_.get(),
      max_len,
      a->Data<float>(),
      y_data,
      nullptr);

  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }
  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(MatMul, kOnnxDomain, 7, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  MatMul);

ONNX_OPERATOR_KERNEL_EX(MatMul, kOnnxDomain, 13, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        MatMul);

}  // namespace xnnpack
}  // namespace onnxruntime
