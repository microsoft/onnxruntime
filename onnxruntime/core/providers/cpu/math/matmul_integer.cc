// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "core/providers/cpu/math/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/gemmlowp_common_wrapper.h"
#include "core/nblas/nblas_igemv_mkl.h"
#include "core/nblas/nblas_igemv_avx2.h"
#include "core/common/cpuid_info.h"
#include "core/framework/allocator.h"

namespace onnxruntime {

// only register this operator if low precision computation is enabled.
ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t, int8_t, int32_t>);

Status GemmlowpMultiply(const uint8_t* lhs_data, const uint8_t* rhs_data,
                        int32_t* result_data, const int lhs_offset, const int rhs_offset,
                        int m, int n, int k) {
  const std::tuple<> empty_pipeline = {};
  // TODO exp ColMajor order for rhs and result. That may be faster
  const auto matOrder = gemmlowp::MapOrder::RowMajor;
  gemmlowp::MatrixMap<const std::uint8_t, matOrder> lhs(lhs_data, m, k);
  gemmlowp::MatrixMap<const std::uint8_t, matOrder> rhs(rhs_data, k, n);
  gemmlowp::MatrixMap<std::int32_t, matOrder> result(result_data, m, n);

  gemmlowp::GemmContext gemm_context;
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, empty_pipeline);

  return Status::OK();
}

template <>

Status MatMulInteger<uint8_t, int8_t, int32_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);

  ORT_ENFORCE(a != nullptr && b != nullptr);
  ORT_ENFORCE(a->Shape().NumDimensions() == 2);
  ORT_ENFORCE(b->Shape().NumDimensions() == 2);
  ORT_ENFORCE(a->Shape()[1] == b->Shape()[0]);

  MatMulComputeHelper helper;

  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));

  Tensor* y = ctx->Output(0, helper.OutputShape());
  auto a_data = const_cast<uint8_t*>(a->Data<uint8_t>());
  auto b_data = const_cast<int8_t*>(b->Data<int8_t>());

  auto q_y = y->MutableData<int32_t>();

  auto batch = a->Shape()[0];
  auto input_dim = a->Shape()[1];
  auto embed_dim = b->Shape()[1];

  // NOTE that NBLAS assumes weights to be transposed. Just store in a map here
  //std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  static thread_local std::unordered_map<int8_t*, IAllocatorUniquePtr<int8_t>> transposed_weights;

  if (transposed_weights.count(b_data) == 0) {
    AllocatorPtr alloc;
    auto status = ctx->GetTempSpaceAllocator(&alloc);
    ORT_RETURN_IF_ERROR(status);
    auto transposed = IAllocator::MakeUniquePtr<int8_t>(alloc, b->Shape().Size());

    for (int i = 0; i < input_dim; ++i)
      for (int e = 0; e < embed_dim; ++e)
        transposed.get()[e * input_dim + i] = b_data[i * embed_dim + e];
    transposed_weights.emplace(b_data, std::move(transposed));
  }

  b_data = transposed_weights.at(b_data).get();
  /*auto end_time = std::chrono::high_resolution_clock::now();
  /auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  if (dur > 0) {
    transposetime += dur;
    std::cout << "imatmul transpose time: " << transposetime << "This iteration time: " << dur << std::endl;
  }*/
  
  bool use_AVX2 = CPUIDInfo::GetCPUIDInfo().HasAVX2();

  if (batch == 1 && use_AVX2) {
    if (input_dim % 32 == 0)
      AVX2IntGemvS8U8S32R(b_data, a_data, input_dim, input_dim, embed_dim, q_y);
    else
      AVX2IntGemvS8U8S32REx(b_data, a_data, input_dim, embed_dim, q_y);
  } else {
    //MKLIntGemvS8U8S32R(b_data, a_data, embed_dim, batch, input_dim, q_y);
    ORT_ENFORCE(false, "Uexpected code path. MKL called for matmulinteger");
  }

  return Status::OK();
}
template<>
Status MatMulInteger<uint8_t, uint8_t, int32_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  int32_t a_offset = 0;
  int32_t b_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(a_zero_point->Shape().NumDimensions() == 0 ||
        (a_zero_point->Shape().NumDimensions() == 1 && a_zero_point->Shape().GetDims().size() == 1),
        "Currently only scalar zero_point is supported. TODO: add per channel zero point support.");
    a_offset = static_cast<int32_t>(*a_zero_point->template Data<uint8_t>());
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(b_zero_point->Shape().NumDimensions() == 0 ||
        (b_zero_point->Shape().NumDimensions() == 1 && b_zero_point->Shape().GetDims().size() == 1),
        "Currently only scalar zero_point is supported. TODO: add per channel zero point support.");
    b_offset = static_cast<int32_t>(*b_zero_point->template Data<uint8_t>());
  }

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    GemmlowpMultiply(a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                     b->template Data<uint8_t>() + helper.RightOffsets()[i],
                     y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
                     a_offset,
                     b_offset,
                     static_cast<int>(helper.M()),
                     static_cast<int>(helper.N()),
                     static_cast<int>(helper.K()));
  }

  return Status::OK();
}
}  // namespace onnxruntime
