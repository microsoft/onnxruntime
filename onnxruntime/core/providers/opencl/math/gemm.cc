// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"
#include "core/providers/cpu/math/gemm_helper.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME gemm_kernel_src
#include "opencl_generated/math/kernels/gemm.cl.inc"
}  // namespace

template <typename T>
class Gemm : public OpenCLKernel {
 public:
  explicit Gemm(const OpKernelInfo& info) : OpenCLKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <>
class Gemm<float> : public OpenCLKernel {
 public:
  explicit Gemm(const OpKernelInfo& info) : OpenCLKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp;

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    info.GetAttrOrDefault<float>("beta", &beta_, 1.f);
    LoadProgram(gemm_kernel_src, gemm_kernel_src_len);
    LoadKernel("Gemm_Float");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  TensorShape b_shape_;

  IAllocatorUniquePtr<void> packed_b_;

  float alpha_;
  float beta_;

  int64_t trans_A_;
  int64_t trans_B_;
};

Status Gemm<float>::Compute(OpKernelContext* context) const {
  if (packed_b_) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Gemm: Unsupported Pepacked.");
  }
  const auto* A = context->Input<Tensor>(0);
  const auto* B = packed_b_ ? nullptr : context->Input<Tensor>(1);
  const auto* C = context->Input<Tensor>(2);

  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(), trans_A_ != 0, B ? B->Shape() : b_shape_, trans_B_ != 0,
                    C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  ptrdiff_t M = helper.M();
  ptrdiff_t N = helper.N();
  ptrdiff_t K = helper.K();

  auto Y = context->Output(0, {M, N});

  // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
  if (M == 0 || N == 0)
    return Status::OK();

  const cl_int is_C = C != nullptr ? 1 : 0;

  if (B) {
    // ComputeGemm(trans_A_, trans_B_, M, N, K, alpha_, A->Data<float>(), B->Data<float>(), beta_,
    //             c_data, c_shape, y_data, thread_pool);

    cl_mem C_Shape;
    C_Shape = exec_->GetScratchBufferTmp((cl_int)16);
    if (is_C) {
      int64_t c_Shape[] = {1, 1};
      if (C->Shape().NumDimensions() == 1) {
        c_Shape[1] = C->Shape().GetDims().data()[0];
      } else if (C->Shape().NumDimensions() == 2) {
        c_Shape[0] = C->Shape().GetDims().data()[0];
        c_Shape[1] = C->Shape().GetDims().data()[1];
      }
      exec_->WriteToCLBuffer(C_Shape, c_Shape, (cl_int)16);
    }

    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Gemm_Float")}
            .SetArg<cl_int>((cl_int)(trans_A_ ? 1 : 0))
            .SetArg<cl_int>((cl_int)(trans_B_ ? 1 : 0))
            .SetArg<cl_long>((cl_long)(M))
            .SetArg<cl_long>((cl_long)(N))
            .SetArg<cl_long>((cl_long)(K))  // ptrdiff_t -> (cl_long)
            .SetBuffers(*A, *B, *C)
            .SetBuffers(C_Shape, *Y)
            .SetArg<cl_int>(is_C)
            .Launch(*exec_, {M, N, 1}));

    exec_->ReleaseCLBuffer(C_Shape);

    // Unsupported gemm + activation fused
    // ComputeActivation(y_data, SafeInt<size_t>(M) * N);

  } else {
    return Status(common::ONNXRUNTIME, common::FAIL, "Gemm: Unsupported Pepacked.");
  }

  return Status::OK();
}

ONNX_OPENCL_OPERATOR_KERNEL(
    Gemm,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);
}  // namespace opencl
}  // namespace onnxruntime
