// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "gemm.h"
#include "dnnl.h"
#include "dnnl.hpp"
#include "core/providers/dnnl/dnnl_fwd.h"
#include "gsl/gsl"
#include "Eigen/Core"

namespace onnxruntime {
namespace ort_dnnl {

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7,
    kDnnlExecutionProvider,
    KernelDefBuilder::Create()->TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

class GemmHelper {
 public:
  GemmHelper(const TensorShape& left, bool trans_left, const TensorShape& right, bool trans_right, const TensorShape& bias) {
    ORT_ENFORCE(left.NumDimensions() == 2 || left.NumDimensions() == 1);
    ORT_ENFORCE(right.NumDimensions() == 2);

    if (trans_left) {
      M_ = left.NumDimensions() == 2 ? left[1] : left[0];
      K_ = left.NumDimensions() == 2 ? left[0] : 1;
    } else {
      M_ = left.NumDimensions() == 2 ? left[0] : 1;
      K_ = left.NumDimensions() == 2 ? left[1] : left[0];
    }

    int k_dim;
    if (trans_right) {
      N_ = right[0];
      k_dim = 1;
    } else {
      N_ = right[1];
      k_dim = 0;
    }

    if (right[k_dim] != K_)
      status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                "GEMM: Dimension mismatch, W: ",
                                right.ToString(),
                                " K: " + std::to_string(K_),
                                " N:" + std::to_string(N_));

    if (!IsValidBroadcast(bias, M_, N_))
      status_ = common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Gemm: Invalid bias shape for broadcast");

    // it is possible the input is empty tensor, for example the output of roipool in fast rcnn.
    ORT_ENFORCE(M_ >= 0 && K_ > 0 && N_ >= 0);
  }

  int64_t M() const { return M_; }
  int64_t N() const { return N_; }
  int64_t K() const { return K_; }
  Status State() const { return status_; }

 private:
  bool IsValidBroadcast(const TensorShape& bias_shape, int64_t M, int64_t N) {
    // valid shapes are (,) , (1, N) , (M, 1) , (M, N)
    if (bias_shape.NumDimensions() > 2)
      return false;
    // shape is (1,) or (1, 1), or (,)
    if (bias_shape.Size() == 1)
      return true;
    // valid bias_shape (s) are (N,) or (1, N) or (M, 1) or (M, N),
    // In last case no broadcasting needed, so don't fail it
    return ((bias_shape.NumDimensions() == 1 && bias_shape[0] == N) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == M && (bias_shape[1] == 1 || bias_shape[1] == N)) ||
            (bias_shape.NumDimensions() == 2 && bias_shape[0] == 1 && bias_shape[1] == N));
  }

 private:
  int64_t M_;
  int64_t K_;
  int64_t N_;
  Status status_;
};

template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
template <typename T>
using ConstEigenVectorMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <>
Status Gemm<float>::Compute(OpKernelContext* ctx) const {
  const auto X = ctx->Input<Tensor>(0);
  const auto W = ctx->Input<Tensor>(1);
  const auto B = ctx->Input<Tensor>(2);
  GemmHelper helper(X->Shape(), trans_A_, W->Shape(), trans_B_, B->Shape());

  if (!helper.State().IsOK())
    return helper.State();

  dnnl::memory::dim M = gsl::narrow_cast<int>(helper.M());
  dnnl::memory::dim N = gsl::narrow_cast<int>(helper.N());
  dnnl::memory::dim K = gsl::narrow_cast<int>(helper.K());
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

  // dnnl_sgemm expects row major matrices, so no need to swap the operands A and B
  auto status = dnnl_sgemm(trans_A_ ? 'T' : 'N',
                           trans_B_ ? 'T' : 'N',
                           M, N, K,
                           alpha_, X->template Data<float>(), trans_A_ ? M : K,
                           W->template Data<float>(), trans_B_ ? K : N,
                           beta_, Y->template MutableData<float>(), N);
  if (status == dnnl_success) {
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DNNL_sgemm failed with status: ", status);
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
