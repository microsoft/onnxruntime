// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/group_gemm.h"
#include "contrib_ops/cuda/math/group_gemm_impl.h"
// #include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "core/common/gsl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_GROUP_GEMM_KERNEL_TYPED(op_name, T)              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      op_name,                                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GroupGemm<T>);

// REGISTER_GROUP_GEMM_KERNEL_TYPED(GroupGemm, float)
REGISTER_GROUP_GEMM_KERNEL_TYPED(GroupGemm, MLFloat16)

template <typename T>
Status GroupGemm<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  int l_batch_end_axis = left_X->GetBatchEndAxis();
  int r_batch_end_axis = right_X->GetBatchEndAxis();

  ORT_ENFORCE(l_batch_end_axis == r_batch_end_axis,
              "GroupGemm requires left_X and right_X to have the same number of batch.");

  ORT_ENFORCE(l_batch_end_axis == static_cast<int>(left_X->Shape().NumDimensions() - 3),
              "GroupGemm requires left_X to have at least 2 vadaric dimensions.");

  std::vector<std::vector<int64_t>> l_variant_strides = left_X->VariantStrides();
  gsl::span<const int64_t> l_batch_offsets = left_X->BatchOffset();

  size_t num_of_batches = l_batch_offsets.size();
  size_t problem_count = num_of_batches;

  // Compute the output shape:
  //  output_dims = [left_X.shape[:-2] + [left_X.shape[-2], right_X.shape[-1]]
  std::vector<int64_t> output_dims(left_X->Shape().NumDimensions());
  size_t dim_index = 0;
  for (; dim_index <= static_cast<size_t>(l_batch_end_axis); ++dim_index) {
    output_dims[dim_index] = left_X->Shape().GetDims()[dim_index];
  }

  for (; dim_index < output_dims.size(); ++dim_index) {
    const Tensor* x = dim_index == output_dims.size() - 1 ? right_X : left_X;
    if (x->IsVariantDim(dim_index)) {
      // dim_values_on_var_shape.insert({dim_index, });
      output_dims[dim_index] = -1;  // pass -1 for variant dim
    } else {
      output_dims[dim_index] = x->Shape().GetDims()[dim_index];
    }
  }

  std::unordered_map<int, std::vector<int64_t>> dim_values_on_var_shape;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes;
  problem_sizes.reserve(problem_count);
  for (int64_t i = 0; i < static_cast<int64_t>(problem_count); ++i) {
    std::vector<int64_t> lhs_shape, rhs_shape;
    left_X->GetShapeForBatch(i, lhs_shape);
    right_X->GetShapeForBatch(i, rhs_shape);

    problem_sizes.push_back(cutlass::gemm::GemmCoord(lhs_shape[0], rhs_shape[1], lhs_shape[1]));

    if (dim_values_on_var_shape.find(0) == dim_values_on_var_shape.end()) {
      dim_values_on_var_shape[0].push_back(lhs_shape[0]);
    } else {
      dim_values_on_var_shape.insert({0, {lhs_shape[0]}});
    }

    if (dim_values_on_var_shape.find(1) == dim_values_on_var_shape.end()) {
      dim_values_on_var_shape[1].push_back(rhs_shape[1]);
    } else {
      dim_values_on_var_shape.insert({1, {rhs_shape[1]}});
    }
  }

  Tensor* Y = ctx->Output(0, TensorShape(output_dims), dim_values_on_var_shape);

  typedef typename ToCudaType<T>::MappedType CudaT;

  CudaAsyncBuffer<cutlass::gemm::GemmCoord> problem_sizes_device(this, problem_count);
  CudaAsyncBuffer<int64_t> lda(this, problem_count);
  CudaAsyncBuffer<int64_t> ldb(this, problem_count);
  CudaAsyncBuffer<int64_t> ldc(this, problem_count);
  CudaAsyncBuffer<int64_t> ldd(this, problem_count);

  CudaAsyncBuffer<CudaT*> data_ptr_a(this, problem_count);
  CudaAsyncBuffer<CudaT*> data_ptr_b(this, problem_count);
  CudaAsyncBuffer<CudaT*> data_ptr_c(this, problem_count);
  CudaAsyncBuffer<CudaT*> data_ptr_d(this, problem_count);

  gsl::span<cutlass::gemm::GemmCoord> problem_sizes_span = problem_sizes_device.CpuSpan();
  gsl::span<int64_t> lda_span = lda.CpuSpan();
  gsl::span<int64_t> ldb_span = ldb.CpuSpan();
  gsl::span<int64_t> ldc_span = ldc.CpuSpan();
  gsl::span<int64_t> ldd_span = ldd.CpuSpan();

  gsl::span<CudaT*> data_ptr_a_span = data_ptr_a.CpuSpan();
  gsl::span<CudaT*> data_ptr_b_span = data_ptr_b.CpuSpan();
  gsl::span<CudaT*> data_ptr_c_span = data_ptr_c.CpuSpan();
  gsl::span<CudaT*> data_ptr_d_span = data_ptr_d.CpuSpan();

  // PrepareGroupGemArguments(problem_sizes, problem_sizes_span,
  //                          lda_span, ldb_span, ldc_span, ldd_span,
  //                          data_ptr_a_span, data_ptr_b_span, data_ptr_c_span, data_ptr_d_span,
  //                          reinterpret_cast<const CudaT*>(left_X->Data<T>()),
  //                          reinterpret_cast<const CudaT*>(right_X->Data<T>()),
  //                          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
  //                          reinterpret_cast<CudaT*>(Y->MutableData<T>()));

  for (size_t i = 0; i < problem_count; ++i) {
    auto& problem = problem_sizes[i];
    problem_sizes_span[i] = problem;  // cutlass::gemm::GemmCoord(problem.m, problem.n, problem.k);
    // lda_span[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
    // ldb_span[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
    // ldc_span[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    // ldd_span[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    data_ptr_a_span[i] = const_cast<CudaT*>(reinterpret_cast<const CudaT*>(left_X->Data<T>()));
    data_ptr_b_span[i] = const_cast<CudaT*>(reinterpret_cast<const CudaT*>(right_X->Data<T>()));
    data_ptr_c_span[i] = reinterpret_cast<CudaT*>(Y->MutableData<T>());
    data_ptr_d_span[i] = reinterpret_cast<CudaT*>(Y->MutableData<T>());
  }

  GenerateLdaLdbLdcLdd<true>(problem_sizes, lda_span, ldb_span, ldc_span, ldd_span);

  ORT_RETURN_IF_ERROR(problem_sizes_device.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(lda.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(ldb.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(ldc.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(ldd.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_a.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_b.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_c.CopyToGpu(ctx->GetComputeStream()));
  ORT_RETURN_IF_ERROR(data_ptr_d.CopyToGpu(ctx->GetComputeStream()));

  auto ret = GroupGemm_Impl<CudaT, true, true>(
      this,
      ctx->GetComputeStream(),
      problem_sizes,
      problem_sizes_device.GpuPtr(),
      problem_count,
      lda.GpuPtr(),
      ldb.GpuPtr(),
      ldc.GpuPtr(),
      ldd.GpuPtr(),
      data_ptr_a.GpuPtr(),
      data_ptr_b.GpuPtr(),
      data_ptr_c.GpuPtr(),
      data_ptr_d.GpuPtr());

  ORT_RETURN_IF_ERROR(ret);
  return Status::OK();
}

#undef REGISTER_GROUP_GEMM_KERNEL_TYPED

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
