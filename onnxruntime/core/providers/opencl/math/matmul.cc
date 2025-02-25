// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME matmul_kernel_src
#include "opencl_generated/math/kernels/matmul.cl.inc"
}  // namespace

template <typename T>
class MatMul : public OpenCLKernel {
 public:
  explicit MatMul(const OpKernelInfo& info) : OpenCLKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <>
class MatMul<float> : public OpenCLKernel {
 public:
  explicit MatMul(const OpKernelInfo& info) : OpenCLKernel(info) {
    info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
    info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
    info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
    int64_t trans_batch_a_attr, trans_batch_b_attr;
    info.GetAttrOrDefault<int64_t>("transBatchA", &trans_batch_a_attr, 0);
    info.GetAttrOrDefault<int64_t>("transBatchB", &trans_batch_b_attr, 0);
    trans_batch_a_ = trans_batch_a_attr != 0;
    trans_batch_b_ = trans_batch_b_attr != 0;
    LoadProgram(matmul_kernel_src, matmul_kernel_src_len);
    LoadKernel("MatMul_Batch_Float");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  TensorShape b_shape_;

  // For FusedMatMul contrib ops
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
  bool trans_batch_a_;
  bool trans_batch_b_;
};

Status MatMul<float>::Compute(OpKernelContext* context) const {
  VLOG_CL_NODE();
  bool packed_b_ = 0;  // no support for prepack
  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* b = packed_b_ ? nullptr : context->Input<Tensor>(1);
  const auto& b_shape = b ? b->Shape() : b_shape_;

  // match CUDA kernel implementation, ignore transpose for vectors
  const bool trans_a = trans_a_attr_ && a->Shape().NumDimensions() != 1;
  const bool trans_b = trans_b_attr_ && b_shape.NumDimensions() != 1;

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, trans_a, trans_b, trans_batch_a_, trans_batch_b_));
  Tensor* y = context->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  int Shape_sIze = 40;
  auto A_Shape = exec_->GetScratchBufferTmp(Shape_sIze);
  auto B_Shape = exec_->GetScratchBufferTmp(Shape_sIze);
  auto C_Shape = exec_->GetScratchBufferTmp(Shape_sIze);
  int64_t a_Shape[] = {1, 1, 1, 1, 1};
  for (size_t i = (y->Shape().NumDimensions() - a->Shape().NumDimensions()); i < (y->Shape().NumDimensions()); i++) {
    a_Shape[i] = a->Shape().GetDims().data()[i - (y->Shape().NumDimensions() - a->Shape().NumDimensions())];
  }

  int64_t b_Shape[] = {1, 1, 1, 1, 1};
  for (size_t i = (y->Shape().NumDimensions() - b->Shape().NumDimensions()); i < (y->Shape().NumDimensions()); i++) {
    b_Shape[i] = b->Shape().GetDims().data()[i - (y->Shape().NumDimensions() - b->Shape().NumDimensions())];
  }
  const int64_t* c_Shape = y->Shape().GetDims().data();
  exec_->WriteToCLBuffer(A_Shape, a_Shape, Shape_sIze);
  exec_->WriteToCLBuffer(B_Shape, b_Shape, Shape_sIze);
  exec_->WriteToCLBuffer(C_Shape, c_Shape, Shape_sIze);

  ORT_RETURN_IF_ERROR(
      KernelLauncher{GetKernel("MatMul_Batch_Float")}
          .SetArg<cl_int>((cl_int)(trans_a ? 1 : 0))
          .SetArg<cl_int>((cl_int)(trans_b ? 1 : 0))
          .SetBuffers(*a, *b, *y)
          .SetBuffers(A_Shape, B_Shape, C_Shape)
          .SetArg<cl_int>((cl_int)(y->Shape().NumDimensions()))
          .Launch(*exec_, {y->SizeInBytes() / 4, 1, 1}));

  exec_->ReleaseCLBuffer(A_Shape);
  exec_->ReleaseCLBuffer(B_Shape);
  exec_->ReleaseCLBuffer(C_Shape);
  return Status::OK();
}

ONNX_OPENCL_OPERATOR_KERNEL(
    MatMul,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);
}  // namespace opencl
}  // namespace onnxruntime
