// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer.h"
#include "matmul_integer.cuh"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/shared_inc/integer_gemm.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(2)
        .InputMemoryType<OrtMemTypeCPUInput>(3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t, int8_t>);

template <>
Status MatMulInteger<int8_t, int8_t>::ComputeInternal(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  const int8_t* a_ptr = a->template Data<int8_t>();
  const int8_t* b_ptr = b->template Data<int8_t>();
  int32_t* output_ptr = Y->template MutableData<int32_t>();

  // validate zero points
  int8_t a_offset = 0;
  int8_t b_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *(a_zero_point->template Data<int8_t>());
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = *(b_zero_point->template Data<int8_t>());
  }

  // offset output c[i,j] to
  // k*a_offset*b_offset -
  // b_offset * (a[i,0] + a[i,1] ...+a[i,k]) -
  // a_offset * (b[0,j] + b[1,j] ... + b[k,j])
  // ReduceRowSumOnMatrixA computes the b_offset * (a[i,0] + a[i,1] ...+a[i,k]) part
  // ReduceColSumOnMatrixB computes the a_offset * (b[0,j] + b[1,j] ... + b[k,j]) part
  // OffsetOutput computes gets the final result
  IAllocatorUniquePtr<int32_t> a_row_buf;
  if (b_offset != 0) {
    a_row_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.N());
    ORT_RETURN_IF_ERROR(ReduceRowSumOnMatrixA(Stream(), a_ptr, a_row_buf.get(), b_offset, helper));
  }

  IAllocatorUniquePtr<int32_t> b_col_buf;
  if (a_offset != 0) {
    b_col_buf = GetScratchBuffer<int32_t>(helper.OutputShape().Size() / helper.M());
    ORT_RETURN_IF_ERROR(ReduceColSumOnMatrixB(Stream(), b_ptr, b_col_buf.get(), a_offset, helper));
  }

  int alpha = 1;
  int beta = 0;
  if (a_offset != 0 || b_offset != 0) {
    OffsetOutput(Stream(),
                 a_row_buf.get(),
                 b_col_buf.get(),
                 output_ptr,
                 a_offset,
                 b_offset,
                 helper);
    beta = 1;
  }

  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    GemmInt8(static_cast<int>(helper.M()),
             static_cast<int>(helper.N()),
             static_cast<int>(helper.K()),
             alpha,
             beta,
             a_ptr + helper.LeftOffsets()[batch],
             static_cast<int>(helper.K()),
             b_ptr + helper.RightOffsets()[batch],
             static_cast<int>(helper.N()),
             output_ptr + helper.OutputOffsets()[batch],
             static_cast<int>(helper.N()),
             this);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
