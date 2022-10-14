// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.

#include "gemm.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "gsl/gsl-lite.hpp"

#include <iostream>

    namespace onnxruntime {
  namespace xnnpack {

  bool Gemm::IsGemmOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
    bool supported = false;
    const onnxruntime::Node& node = node_unit.GetNode();

    // use do {} while(false) so it's easier to set a breakpoint on the return
    do {
      const auto& input_defs = node.InputDefs();

      if (input_defs.size() < 2) {
        break;
      }

      const auto& A_arg = *input_defs[0];
      const auto& B_arg = *input_defs[1];
      const auto& C_arg = *input_defs[2];

      // we only support float currently
      const auto* A_type = A_arg.TypeAsProto();
      const auto* B_type = B_arg.TypeAsProto();
      const auto* C_type = C_arg.TypeAsProto();

      if (A_type == nullptr || B_type == nullptr || C_type == nullptr ||
          A_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
          B_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
          C_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        break;
      }

      // B & C matrices must be constant
      if (!graph.IsConstantInitializer(B_arg.Name(), true)) {
        break;
      }

      if (!graph.IsConstantInitializer(C_arg.Name(), true)) {
        break;
      }

      // making sure we are dealing with MatMul
      const auto* A_shape = A_arg.Shape();
      const auto* B_shape = B_arg.Shape();
      const auto* C_shape = C_arg.Shape();
      
       if (!A_shape || A_shape->dim_size() > 3 || (A_shape->dim(0).dim_value() != 1 && A_shape->dim(1).dim_value() != 1)) {
        break;
      }

      if (!B_shape || B_shape->dim_size() > 3 || (B_shape->dim(0).dim_value() != 1 && B_shape->dim(1).dim_value() != 1)) {
        break;
      }

      if (!C_shape || C_shape->dim_size() > 3) {
        break;
      }

      ProtoHelperNodeContext nc(node_unit.GetNode());
      OpNodeProtoHelper info(&nc);

      supported = true;

    } while (false);

    return supported;
  }

  Gemm::Gemm(const OpKernelInfo& info) : GemmBase(info), OpKernel(info) {
    const auto& node{Node()};
    
    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    info.GetAttrOrDefault<float>("beta", &beta_, 1.f);

    const auto& input_defs = node.InputDefs();
    const NodeArg& A = *input_defs[0];
    const NodeArg& B = *input_defs[1];

    if (trans_A_ != CblasNoTrans) {
      M = A.Shape()->dim_size() == 3 ? A.Shape()->dim(2).dim_value() : A.Shape()->dim(1).dim_value();
      K = A.Shape()->dim_size() == 3 ? A.Shape()->dim(1).dim_value() : A.Shape()->dim(0).dim_value() > 1 ? A.Shape()->dim(0).dim_value()
                                                                                                         : 1;
    } else {
      K = A.Shape()->dim_size() == 3 ? A.Shape()->dim(2).dim_value() : A.Shape()->dim(1).dim_value();
      M = A.Shape()->dim_size() == 3 ? A.Shape()->dim(1).dim_value() : A.Shape()->dim(0).dim_value() > 1 ? A.Shape()->dim(0).dim_value()                                                                                                 : 1;
    }

    if (trans_B_ == CblasNoTrans) {
      N = B.Shape()->dim_size() == 3 ? B.Shape()->dim(2).dim_value() : B.Shape()->dim(1).dim_value();
    } else {
      //N = B.Shape()->dim_size() == 3 ? B.Shape()->dim(1).dim_value() : B.Shape()->dim(2).dim_value();
      N = B.Shape()->dim_size() == 3 ? B.Shape()->dim(1).dim_value() : B.Shape()->dim(0).dim_value() > 1 ? B.Shape()->dim(0).dim_value()                                                                                                  : 1;
    }
  }

  Status Gemm::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                       /*out*/ bool& is_packed,
                       /*out*/ PrePackedWeights*) {
    is_packed = false;

    if (input_idx == 0) {
      return Status::OK();
    }

    if (input_idx == 1) {
      B_ = Tensor(tensor.DataType(), TensorShape(tensor.Shape()), alloc);
      SingleAxisTranspose(std::vector<size_t>{0, 1}, tensor, B_, /*from*/ 1, /*to*/ 1);

      return Status::OK();
    }

    is_packed = true;

    uint32_t flags = trans_B_ != CblasTrans ? XNN_FLAG_TRANSPOSE_WEIGHTS:0;

    float output_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
    float output_max = clip_min_max_ ? clip_min_max_->second : INFINITY;

    if (input_idx == 2) {
      xnn_status status = xnn_status::xnn_status_uninitialized;
      struct xnn_operator* p = nullptr;
      status = xnn_create_fully_connected_nc_f32(
          B_.Shape()[0],         // size_t input_channels,
          B_.Shape()[1],         // size_t output_channels,
          B_.Shape()[0],         // size_t input_stride,
          B_.Shape()[1],         // size_t output_stride,
          B_.Data<float>(),      // const float* kernel,
          tensor.Data<float>(),  // const float* bias,
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
    }

    return Status::OK();
  }

  Status Gemm::Compute(OpKernelContext* context) const {
    // concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
    const auto* A = context->Input<Tensor>(0);
    auto Y = context->Output(0, {M, N});

    // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
    if (M == 0 || N == 0)
      return Status::OK();
    
    xnn_status status = xnn_setup_fully_connected_nc_f32(
        op0_.get(),
        trans_A_ != CblasNoTrans ? K : M,
        A->Data<float>(),
        Y->MutableData<float>(),
        nullptr);

    status = xnn_run_operator(op0_.get(), nullptr);
    
    if (status != xnn_status_success) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
    }
    return Status::OK();
  }

  ONNX_OPERATOR_VERSIONED_KERNEL_EX(Gemm, kOnnxDomain, 7, 12, kXnnpackExecutionProvider,
                                    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                    Gemm);

  ONNX_OPERATOR_KERNEL_EX(Gemm, kOnnxDomain, 13, kXnnpackExecutionProvider,
                          KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                          Gemm);

  }  // namespace xnnpack
}  // namespace onnxruntime