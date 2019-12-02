// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/util/math_cpuonly.h"
#include "mkldnn.h"
#include "mkldnn.hpp"
#include "core/providers/mkldnn/mkldnn_fwd.h"

namespace onnxruntime {
namespace mkl_dnn {

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7,
    kMklDnnExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

template <>
Status Gemm<float>::Compute(OpKernelContext* ctx) const {
  const auto X = ctx->Input<Tensor>(0);
  const auto W = ctx->Input<Tensor>(1);
  const auto B = ctx->Input<Tensor>(2);
  GemmHelper helper(X->Shape(), trans_A_, W->Shape(), trans_B_, B->Shape());

  if (!helper.State().IsOK())
    return helper.State();

  mkldnn::memory::dim M = gsl::narrow_cast<int>(helper.M());
  mkldnn::memory::dim N = gsl::narrow_cast<int>(helper.N());
  mkldnn::memory::dim K = gsl::narrow_cast<int>(helper.K());
  auto Y = ctx->Output(0, TensorShape({M, N}));

  if (M <= 0)
	return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Empty Tensor not supported");

  if (beta_ != 0) {
    auto output_mat = EigenMatrixMapRowMajor<float>(
        Y->template MutableData<float>(),
        M,
        N);
    output_mat.setZero();

    auto& b_shape = B->Shape();
    // if B is (), (1,) or (1, 1), add the scalar
    if (b_shape.Size() == 1) {
      output_mat.array() += *(B->template Data<float>());
    }
    // B is (N,)
    else if (b_shape.NumDimensions() == 1) {
      auto bias_vec = ConstEigenVectorMap<float>(
          B->template Data<float>(),
          N);
      output_mat.rowwise() += bias_vec.transpose();
    } else if (b_shape.NumDimensions() == 2) {
      // B is (M, 1)
      if (b_shape[1] == 1) {
        auto bias_vec = ConstEigenVectorMap<float>(
            B->template Data<float>(),
            M);
        output_mat.colwise() += bias_vec;
      }
      // B is (1, N)
      else if (b_shape[0] == 1) {
        auto bias_vec = ConstEigenVectorMap<float>(
            B->template Data<float>(),
            N);
        output_mat.rowwise() += bias_vec.transpose();
      }
      // B is (M, N), no broadcast needed.
      else {
        auto bias_mat = ConstEigenMatrixMapRowMajor<float>(
            B->template Data<float>(),
            M,
            N);
        output_mat += bias_mat;
      }
    }
  }

  // mkldnn_sgemm expects row major matrices, so no need to swap the operands A and B
  auto status = mkldnn_sgemm(trans_A_ ? 'T' : 'N',
                             trans_B_ ? 'T' : 'N',
                             M, N, K,
                             alpha_, X->template Data<float>() , trans_A_ ? M : K,
                             W->template Data<float>(), trans_B_ ? K : N,
                             beta_, Y->template MutableData<float>(), N);
  if (status == mkldnn_success) {
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "mkldnn_sgemm failed with status: ", status);
  }
}

}  // namespace mkl_dnn
}  // namespace onnxruntime
