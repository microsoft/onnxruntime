// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace xnnpack {

bool Gemm::IsOnnxNodeSupported(const onnxruntime::Node& node, const GraphViewer& graph) {

  bool supported = false;

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    auto input_defs = node.InputDefs();

    if (input_defs.size()<2) {
      printf("Gemm XNNPACK not supported - A & B must be provided");
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
        printf("Gemm XNNPACK not supported - currently only float Gemm is supported");
        break;
    }
    
    // B matrix must be constant
    if (B_arg.Exists() && graph.GetConstantInitializer(B_arg.Name(), true) == nullptr) {
        printf("Gemm XNNPACK not supported - B must be a const");
        break;
    }    

    // making sure we are dealing with MatMul
    const auto* A_shape = A_arg.Shape();
    const auto* B_shape = B_arg.Shape();
    const auto* C_shape = C_arg.Shape(); 

    if (!A_shape || A_shape->dim_size() > 3) {
        printf("Gemm XNNPACK not supported - only up to 2D opps are supported\n");
        break;
    }    

    if (!B_shape || B_shape->dim_size() > 3) {
        printf("Gemm XNNPACK not supported - only up to 2D opps are supported\n");
        break;
    }    

    if (!C_shape || C_shape->dim_size() > 3) {
        printf("Gemm XNNPACK not supported - only up to 2D opps are supported\n");
        break;
    } 

    // if there's a bias input it must be constant
    if (input_defs.size() == 3) {
      if (C_arg.Exists() && !graph.IsConstantInitializer(C_arg.Name(), true)) {
        break;
      }
    }

    ProtoHelperNodeContext nc(node_unit.GetNode());
    OpNodeProtoHelper info(&nc);

    supported = false;

  } while (false);

  return supported;
}

Gemm::Gemm(const OpKernelInfo& info) : GemmBase(info), OpKernel(info){
  // Shalva - Need to fix this !!!! -
  // get values from any fusion with an activation
  /*
  if (std::string activation; info.GetAttr<std::string>("activation", &info).IsOK()) {
    if (activation == "Clip" || activation == "Relu") {
      std::vector<float> activation_params;

      // min/max could be from Clip or Relu
      if (info.GetAttrs<float>("activation_params", activation_params).IsOK()) {
        if (activation_params.size() == 2) {
          clip_min_max_ = {activation_params[0], activation_params[1]};
        }
      }
    }
  }
  */

  const auto& node{Node()};

  const auto& input_defs = node.InputDefs();
  const NodeArg& A = *input_defs[0];
  const NodeArg& B = *input_defs[1];

  //printf("A Shape - %d\n", (int)A.Shape()->dim_size());
  //printf("B Shape - %d\n", (int)B.Shape()->dim_size());

  if (trans_A_ != CblasNoTrans) {
    M = A.Shape()->dim_size() == 3 ? A.Shape()->dim(2).dim_value() : A.Shape()->dim(1).dim_value();
    K = A.Shape()->dim_size() == 3 ? A.Shape()->dim(1).dim_value() : A.Shape()->dim(0).dim_value() > 1 ? A.Shape()->dim(0).dim_value(): 1;
  } else {
    K = A.Shape()->dim_size() == 3 ? A.Shape()->dim(2).dim_value() : A.Shape()->dim(1).dim_value();
    M = A.Shape()->dim_size() == 3 ? A.Shape()->dim(1).dim_value() : A.Shape()->dim(0).dim_value() > 1 ? A.Shape()->dim(0).dim_value(): 1;
  }

  if (trans_B_ == CblasNoTrans) {
    N = B.Shape()->dim_size() == 3 ? B.Shape()->dim(2).dim_value() : B.Shape()->dim(1).dim_value();
  } else {
    N = B.Shape()->dim_size() == 3 ? B.Shape()->dim(1).dim_value() : B.Shape()->dim(2).dim_value();
  } 

  //printf("M - %d\n", (int)M);
  //printf("N - %d\n", (int)N);
  //printf("K - %d\n", (int)K);

}

Status Gemm::PrePack(const Tensor& tensor,int input_idx, AllocatorPtr alloc,
                     /*out*/ bool& is_packed,
                     /*out*/ PrePackedWeights* prepacked_weights) {
  //printf("pre-pack GEMM XNNPACK\n");
  prepacked_weights = nullptr;
  is_packed = false;

  if (input_idx == 0) {
    return Status::OK();
  }
    
  is_packed = true;

  if (input_idx ==  1) {
    B_ = Tensor::Create(tensor.DataType(), TensorShape(tensor.Shape()), alloc);
    //memcpy(B_->Data<short>, tensor.DataRaw(), sizeof(tensor.DataType())*tensor.Shape()[0]*tensor.Shape()[1]);
    SingleAxisTranspose(std::vector<size_t> {0, 1}, tensor, *B_, /*from*/ 1, /*to*/ 1);
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
#endif
    return Status::OK();
  }
  uint32_t flags = trans_B_ != CblasNoTrans ? 0:XNN_FLAG_TRANSPOSE_WEIGHTS;

  float output_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
  float output_max = clip_min_max_ ? clip_min_max_->second : INFINITY;

  if (input_idx == 2) {
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
    status = xnn_create_fully_connected_nc_f32(
        B_->Shape()[0],          // size_t input_channels,
        B_->Shape()[1],          // size_t output_channels,
        B_->Shape()[0],          // size_t input_stride,
        B_->Shape()[1],          // size_t output_stride,
        B_->Data<float>(),       // const float* kernel,
        tensor.Data<float>(),    // const float* bias,
        output_min,
        output_max,
        flags,
        &p);

    if (status != xnn_status_success) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_create_fully_connected_nc_f32 returned ", status);
    }

    op0_.reset(p);
  }

  return Status::OK();
}

Status Gemm::Compute(OpKernelContext* context) const {
  //concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  const auto* A = context->Input<Tensor>(0);
#if 0
  FILE* fp;
  fopen_s(&fp, "XNNPACK.log", "a+");
  fprintf(fp, "current node id - %s\n", context->GetNodeName().c_str());
#endif
  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  auto Y = context->Output(0, {M, N}); 
  
  //const TensorShape* c_shape = C != nullptr ? &C->Shape() : nullptr;

  xnn_status status = xnn_setup_fully_connected_nc_f32(
        op0_.get(),
        trans_A_ != CblasNoTrans ? K : M,
        A->Data<float>(),
        Y->MutableData<float>(), 
        nullptr);
  //printf("executing GEMM XNNPACK\n");
  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }  
#if 0
  // Debug
  // const auto* C = context->Input<Tensor>(2);

  if (Y->Shape().NumDimensions() == 2) {
    fprintf(fp, "\nY shape is - %lld x %lld \n", Y->Shape()[0], Y->Shape()[1]);
    for (int i = 0; i < Y->Shape()[0]; i++) {
      fprintf(fp, "[");
      for (int j = 0; j < Y->Shape()[1]; j++) {
        fprintf(fp, "%lf, ", Y->Data<float>()[j + i * Y->Shape()[1]]);
      }
      fprintf(fp, "]\n");
    }
  } else {
    fprintf(fp, "\nY shape is - %lld \n", Y->Shape()[0]);
    fprintf(fp, "[");
    for (int i = 0; i < Y->Shape()[0]; i++) {
      fprintf(fp, "%lf, ", Y->Data<float>()[i]);
    }
    fprintf(fp, "]\n");
  }
  /*
  fprintf(fp, "\nB is packed\n");

  if (C->Shape().NumDimensions() == 2) {
    fprintf(fp, "\nC shape is - %lld x %lld \n", C->Shape()[0], C->Shape()[1]);
    for (int i = 0; i < C->Shape()[0]; i++) {
      fprintf(fp, "[");
      for (int j = 0; j < C->Shape()[1]; j++) {
        fprintf(fp, "%lf, ", C->Data<float>()[j + i * C->Shape()[1]]);
      }
      fprintf(fp, "]\n");
    }
  } else {
    fprintf(fp, "\nC shape is - %lld \n", C->Shape()[0]);
    fprintf(fp, "[");
    for (int i = 0; i < C->Shape()[0]; i++) {
      fprintf(fp, "%lf, ", C->Data<float>()[i]);
    }
    fprintf(fp, "]\n");
  }*/

  fclose(fp);
#endif
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
