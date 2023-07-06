// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Remove GetDeviceProp in LaunchFastGeluKernel.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "contrib_ops/rocm/bert/grouped_gemm.h"
#include "contrib_ops/rocm/bert/grouped_gemm_tunable.cuh"
#include "core/providers/rocm/tunable/gemm.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {


#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GroupedGemm,                                                \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GroupedGemm<T>);


REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;


template<typename T>
Status GroupedGemm::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<T>::MappedType HipT;

  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* msizes = ctx->Input<Tensor>(1);
  const Tensor* B = ctx->Input<Tensor>(2);
  const Tensor* C = ctx->Input<Tensor>(3);

  GroupedGemmHelper helper(A->Shape(), trans_A_, B->Shape(), trans_B_, C != nullptr ? C->Shape(): TensorShape({}));

  if (!helper.State().IsOK()) {
    return helper.State();
  }

  ptrdiff_t M = helper.M();
  ptrdiff_t N = helper.N();
  ptrdiff_t K = helper.K();
  ptrdiff_t num_matrix = helper.num_matrix();

  // allocate output memory
  auto* Y = ctx->Output(0, {M, N});
  HipT* out_data = reinterpret_cast<HipT*>(Y->MutableData<T>());

  // broadcast bias if it is present
  if (beta_ != 0. && C != nullptr) {
    auto bias_shape = C->Shape();
    const auto* bias_data = reinterpret_cast<HipT*>(C->Data<t>());

    if (bias_shape[0] == M) {
      // has same shape of output, directly copy bias into output buffer
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(out_data, bias_data, M * N * sizeof(T), hipMemcpyDeviceToDevice, Stream(ctx)));
    } else {
      // TODO: for bias shape is (m, N), need to broadcast each N into msizes.
      // here need to implement a kernel to do this.
    }
  }

  // call grouped_gemm kernel to compute grouped_gemm
  return tunable::blas::column_major::GroupedGemm(
      GetTunningContext(), Stream(ctx),
      GetRocblasHandle(ctx),
      trans_B_ ? BlasOp::Trans : BlasOp::NonTrans,
      trans_A_ ? BlasOp::Trans : BlasOp::NonTrans,
      N, M, K, num_matrix,
      alpha_,
      reinterpret_cast<HipT*>(B->Data<T>()), (trans_B_ ? K : N),
      msizes->Data<std::int64_t>(),
      reinterpret_cast<HipT*>(A->Data<T>()), (trans_A_ ? M : K),
      C != nullptr ? beta_ : 0.0f,
      out_data, N);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
