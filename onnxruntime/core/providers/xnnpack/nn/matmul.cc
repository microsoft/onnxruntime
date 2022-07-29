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

bool MatMul::IsOnnxNodeSupported(const onnxruntime::Node& node, const GraphViewer& graph) {

  bool supported = false;

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    auto input_defs = node.InputDefs();

    if (input_defs.size() != 2) {
      printf("MatMul XNNPACK not supported - Only A & B must be provided\n");
      break;
    }

    const auto& A_arg = *input_defs[0];
    const auto& B_arg = *input_defs[1];

    // we only support float currently
    const auto* A_type = A_arg.TypeAsProto();
    const auto* B_type = B_arg.TypeAsProto();

    if (A_type == nullptr || B_type == nullptr ||  
        A_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
        B_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ) {
        printf("MatMul XNNPACK not supported - currently only float Gemm is supported\n");
        break;
    }
    
    // B matrix must be constant
    if (B_arg.Exists() && graph.GetConstantInitializer(B_arg.Name(), true) == nullptr) {
        printf("MatMul XNNPACK not supported - B must be a const\n");
        break;
    }    

    // making sure we are dealing with MatMul
    const auto* B_shape = B_arg.Shape();  

    if (!B_shape || B_shape->dim_size() > 3) {
        printf("MatMul XNNPACK not supported - only up to 2D opps are supported\n");
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
                     /*out*/ PrePackedWeights* prepacked_weights) {

  prepacked_weights = nullptr;
  is_packed = false;

  if (input_idx == 0) {
    return Status::OK();
  }
    
  is_packed = true;

#ifdef DEBUG
    //Debug - printing the tensors
    printf("B shape is - %lld x %lld \n", B_->Shape()[0], B_->Shape()[1]);

    printf("B - \n");
    for (int i = 0; i < B_->Shape()[0]; i++) {
      printf("[");
      for (int j = 0; j < B_->Shape()[1]; j++) {
        printf("%lf, ", B_->Data<float>()[j + i * B_->Shape()[1]]);
      }
      printf("]\n");
    }

    return Status::OK();
#endif
    uint32_t flags = trans_b_attr_ != CblasNoTrans ? XNN_FLAG_TRANSPOSE_WEIGHTS:0;
    float output_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
    float output_max = clip_min_max_ ? clip_min_max_->second : INFINITY;
    xnn_status status = xnn_status::xnn_status_uninitialized;
#ifdef DEBUG
    //Debug - printing the tensors
    printf("C shape is - %lld \n", tensor.Shape()[0]);

    printf("C - \n");
    printf("[");
    for (int i = 0; i < tensor.Shape()[0]; i++) {

      printf("%f, ", tensor.Data<float>()[i]);
      
    }
    printf("]\n");
    ///
#endif
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
        &p);

    if (status != xnn_status_success) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_create_fully_connected_nc_f32 returned ", status);
    }

    op0_.reset(p);

    return Status::OK();
}

Status MatMul::Compute(OpKernelContext* ctx) const {

  //concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);
  const auto& b_shape = b_shape_;

  // match CUDA kernel implementation, ignore transpose for vectors
  const bool trans_a = trans_a_attr_ && a->Shape().NumDimensions() != 1;
  const bool trans_b = trans_b_attr_ && b_shape.NumDimensions() != 1;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, trans_a, trans_b, trans_batch_a_, trans_batch_b_));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  const auto* a_data = a->Data<float>();
  auto* y_data = y->MutableData<float>();

  const size_t max_len = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = helper.Lda(trans_a);
  const size_t ldb = helper.Ldb(trans_b);

  xnn_status status = xnn_setup_fully_connected_nc_f32(
        op0_.get(),
        1,
        a_data,
        y_data, 
        nullptr);

  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }  
#ifdef DEBUG
  // Debug
  printf("Y shape is - %lld x %lld \n", Y->Shape()[0], Y->Shape()[1]);

  printf("Y - \n");
  for (int i = 0; i < Y->Shape()[0]; i++) {
    printf("[");
    for (int j = 0; j < Y->Shape()[1]; j++) {
      printf("%lf, ", Y->Data<float>()[j + i * Y->Shape()[1]]);
    }
    printf("]\n");
  }
#endif
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
