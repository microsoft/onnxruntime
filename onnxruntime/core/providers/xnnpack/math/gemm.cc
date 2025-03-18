// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

#include "gemm.h"

#include <limits>

#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/xnnpack_init.h"

namespace onnxruntime {
namespace xnnpack {

// Todo -
// 1. Integrate activation layers - Cliping & Relu
// 2. Enable C matrix broadcasting - reuse "GemmBroadcastBias" function / logic
// 3. Enable Quant ops
// 4. Review possible consolidation of MatMul & Gemm
//

bool Gemm::IsOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
  bool supported = false;
  const onnxruntime::Node& node = node_unit.GetNode();

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    ConstPointerContainer<std::vector<NodeArg*>> input_defs = node.InputDefs();

    const auto alpha = node.GetAttributes().find("alpha");
    if ((*alpha).second.f() != 1.0) break;

    const auto beta = node.GetAttributes().find("beta");
    if ((*beta).second.f() != 1.0) break;

    const NodeArg* A_arg = input_defs[0];
    const NodeArg* B_arg = input_defs[1];
    const NodeArg* C_arg = input_defs.size() == 2 ? nullptr : input_defs[2];

    // we only support float currently
    const auto* A_type = A_arg->TypeAsProto();

    if (A_type == nullptr ||
        (A_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         A_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)) {
      break;
    }

    // B & C matrices must be constant
    if (!graph.IsConstantInitializer(B_arg->Name(), true)) {
      break;
    }

    if (input_defs.size() == 3 && !graph.IsConstantInitializer(C_arg->Name(), true)) {
      break;
    }

    // making sure we are dealing with MatMul
    const ONNX_NAMESPACE::TensorShapeProto* A_shape = A_arg->Shape();
    const ONNX_NAMESPACE::TensorShapeProto* B_shape = B_arg->Shape();
    const ONNX_NAMESPACE::TensorShapeProto* C_shape = C_arg->Shape();

    if (!A_shape || A_shape->dim_size() >= 3) {
      break;
    }

    if (!B_shape || B_shape->dim_size() >= 3) {
      break;
    }

    if (!C_shape || C_shape->dim_size() >= 3) {
      break;
    }

    if (C_arg && C_arg->Exists() && (C_shape->dim(0).dim_value() != B_shape->dim(1).dim_value() && C_shape->dim(0).dim_value() != B_shape->dim(0).dim_value())) {
      break;
    }

    supported = true;

  } while (false);
  return supported;
}

Gemm::Gemm(const OpKernelInfo& info) : GemmBase(info), XnnpackKernel(info, /*enable_caches*/ true) {
  info.GetAttrOrDefault<float>("alpha", &alpha_, 1.f);
  info.GetAttrOrDefault<float>("beta", &beta_, 1.f);

  const auto& node{Node()};
  const auto& input_defs = node.InputDefs();
  const auto* shapeA = input_defs[0]->Shape();
  const auto* shapeB = input_defs[1]->Shape();

  const NodeArg& X = *input_defs[0];
  auto input_dtype = X.TypeAsProto()->tensor_type().elem_type();
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_compute_type_ = OpComputeType::op_compute_type_fp32;
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    op_compute_type_ = OpComputeType::op_compute_type_fp16;
  }

  const NodeArg* C_arg = input_defs.size() == 2 ? nullptr : input_defs[2];

  C_matrix_exists_ = C_arg && C_arg->Exists();

  // A - MxK
  if (trans_A_ == CblasNoTrans) {
    M_ = shapeA->dim(0).dim_value() > 1 ? shapeA->dim(0).dim_value() : 1;
    K_ = shapeA->dim(1).dim_value();
  } else {
    M_ = shapeA->dim(1).dim_value();
    K_ = shapeA->dim(0).dim_value() > 1 ? shapeA->dim(0).dim_value() : 1;
  }
  // B - KxN
  if (trans_B_ == CblasNoTrans) {
    N_ = shapeB->dim(1).dim_value();
  } else {
    N_ = shapeB->dim(0).dim_value() > 1 ? shapeB->dim(0).dim_value() : 1;
  }
}

Status Gemm::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr,
                     /*out*/ bool& is_packed,
                     /*out*/ PrePackedWeights*) {
  is_packed = false;

  if (input_idx == 0) {
    return Status::OK();
  }

  if (input_idx == 1) {
    B_ = &tensor;
    if (C_matrix_exists_) {
      return Status::OK();
    }
  }

  is_packed = true;

  // flags - 1 - for no transpose - 0 for transpose
  uint32_t flags = trans_B_ == CblasTrans ? 0 : XNN_FLAG_TRANSPOSE_WEIGHTS;
  auto code_cache = GetCodeCache();
  auto weights_cache = GetWeightsCache();
  xnn_status status = xnn_status::xnn_status_uninitialized;
  struct xnn_operator* p = nullptr;
  float foutput_min = clip_min_max_ ? clip_min_max_->first : -std::numeric_limits<float>::infinity();
  float foutput_max = clip_min_max_ ? clip_min_max_->second : std::numeric_limits<float>::infinity();
  if (op_compute_type_ == OpComputeType::op_compute_type_fp32) {
    const float* bias_data = nullptr;
    if (C_matrix_exists_) {
      bias_data = tensor.Data<float>();
    }
    status = xnn_create_fully_connected_nc_f32(
        trans_B_ == CblasNoTrans ? B_->Shape()[0] : B_->Shape()[1],  // size_t input_channels,
        trans_B_ == CblasNoTrans ? B_->Shape()[1] : B_->Shape()[0],  // size_t output_channels,
        trans_B_ == CblasNoTrans ? B_->Shape()[0] : B_->Shape()[1],  // size_t input_stride,
        trans_B_ == CblasNoTrans ? B_->Shape()[1] : B_->Shape()[0],  // size_t output_stride,
        B_->Data<float>(),                                           // const float* kernel,
        bias_data,                                                   // const float* bias,
        foutput_min, foutput_max,
        flags,
        code_cache, weights_cache,
        &p);
  } else if (op_compute_type_ == OpComputeType::op_compute_type_fp16) {
    const MLFloat16* bias_data = nullptr;
    if (C_matrix_exists_) {
      bias_data = tensor.Data<MLFloat16>();
    }
    status = xnn_create_fully_connected_nc_f16(
        trans_B_ == CblasNoTrans ? B_->Shape()[0] : B_->Shape()[1],  // size_t input_channels,
        trans_B_ == CblasNoTrans ? B_->Shape()[1] : B_->Shape()[0],  // size_t output_channels,
        trans_B_ == CblasNoTrans ? B_->Shape()[0] : B_->Shape()[1],  // size_t input_stride,
        trans_B_ == CblasNoTrans ? B_->Shape()[1] : B_->Shape()[0],  // size_t output_stride,
        B_->Data<MLFloat16>(),                                       // const MLFloat16* kernel,
        bias_data,                                                   // const float* bias,
        foutput_min, foutput_max,
        flags,
        code_cache, weights_cache,
        &p);
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_create_fully_connected_nc_",
                           OpTypeToString(op_compute_type_), " returned ", status);
  }
  op0_.reset(p);

  return Status::OK();
}

Status Gemm::Compute(OpKernelContext* context) const {
  pthreadpool_t threadpool = GetThreadPool();
  const auto* A = context->Input<Tensor>(0);
  auto Y = context->Output(0, {M_, N_});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M_ == 0 || N_ == 0) {
    return Status::OK();
  }

  auto reshape_func = xnn_reshape_fully_connected_nc_f32;
  if (op_compute_type_ == OpComputeType::op_compute_type_fp16) {
    reshape_func = xnn_reshape_fully_connected_nc_f16;
  }
  xnn_status status = reshape_func(op0_.get(),
                                   // Number of rows to multiply
                                   trans_A_ == CblasNoTrans ? M_ : K_,
                                   threadpool);

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_reshape_fully_connected_nc_",
                           OpTypeToString(op_compute_type_), " returned ", status);
  }

  status = xnn_status_invalid_state;
  if (op_compute_type_ == op_compute_type_fp32) {
    status = xnn_setup_fully_connected_nc_f32(op0_.get(), A->Data<float>(), Y->MutableData<float>());
  } else if (op_compute_type_ == OpComputeType::op_compute_type_fp16) {
    status = xnn_setup_fully_connected_nc_f16(op0_.get(), A->Data<MLFloat16>(), Y->MutableData<MLFloat16>());
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_fully_connected_nc_",
                           OpTypeToString(op_compute_type_), " returned ", status);
  }

  status = xnn_run_operator(op0_.get(), nullptr);

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }
  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Gemm, kOnnxDomain, 7, 8, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<MLFloat16>()}),
                                  Gemm);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Gemm, kOnnxDomain, 9, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<MLFloat16>()}),
                                  Gemm);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Gemm, kOnnxDomain, 11, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<MLFloat16>()}),
                                  Gemm);

ONNX_OPERATOR_KERNEL_EX(Gemm, kOnnxDomain, 13, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                DataTypeImpl::GetTensorType<MLFloat16>()}),
                        Gemm);

}  // namespace xnnpack
}  // namespace onnxruntime
