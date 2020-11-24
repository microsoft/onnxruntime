// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "Eigen/Core"
#include "Eigen/Dense"


#include "contrib_ops/cpu/matmul_integer16.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
using EigenMatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

// cast TA and TB to TC, and do matrix multiply in Eigen
// note that inputs/outputs is row-major, while Eigen is col-major
// so (M, K) x (K, N) -> (M, N) becomes (N, K) x (K, M) -> (N, M) in Eigen
template <typename TA, typename TB, typename TY>
void EigenCastGEMM(const TA* A_data, const TB* B_data, TY* Y_data, int M, int N, int K) {
  auto A = ConstEigenMatrixMap<TA>(A_data, K, M);
  auto B = ConstEigenMatrixMap<TB>(B_data, N, K);
  EigenMatrixMap<TY>(Y_data, N, M) = B.template cast<TY>() * A.template cast<TY>();
}

ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger16,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int16_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int16_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger16<int16_t, int16_t, int32_t>);

template <>
Status MatMulInteger16<int16_t, int16_t, int32_t>::Compute(OpKernelContext* ctx) const {
  auto A = ctx->Input<Tensor>(0);
  auto B = ctx->Input<Tensor>(1);
  ORT_ENFORCE(A != nullptr && B != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  for (int i = 0; i < static_cast<int>(helper.OutputOffsets().size()); i++) {
    EigenCastGEMM<int16_t, int16_t, int32_t>(
        A->template Data<int16_t>() + helper.LeftOffsets()[i],
        B->template Data<int16_t>() + helper.RightOffsets()[i],
        Y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
