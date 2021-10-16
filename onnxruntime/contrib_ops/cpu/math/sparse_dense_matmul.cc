// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/op_kernel.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/sparse_utils.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

namespace {
using SparseToDenseSupportedTypes = TypeList<float, double, int32_t, int64_t, uint32_t, uint64_t>;
}

// Currently supports batching only for the dense argument
class SparseToDenseMatMul final : public OpKernel {
 public:
  explicit SparseToDenseMatMul(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
    info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
    info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
  }

  Status Compute(OpKernelContext*) const override;

 private:
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
};

ONNX_OPERATOR_KERNEL_EX(
    SparseToDenseMatMul,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefSparseConstraintsFromTypeList<SparseToDenseSupportedTypes>())
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<SparseToDenseSupportedTypes>()),
    SparseToDenseMatMul);

namespace {
struct ComputeCtx {
  bool trans_A;
  bool trans_B;
  float alpha;
};

#if !defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)
template <typename T>
inline void SparseDenseMatMulImpl(const ComputeCtx& ctx, const ConstSparseMatrixMap<T>& map_A,
                                  const ConstEigenMatrixMapRowMajor<T>& map_B, EigenMatrixMapRowMajor<T>& output_map) {
  if (ctx.trans_A && ctx.trans_B) {
    output_map = map_A.transpose() * map_B.transpose();
  } else if (ctx.trans_A && !ctx.trans_B) {
    output_map = map_A.transpose() * map_B;
  } else if (!ctx.trans_A && ctx.trans_B) {
    output_map = map_A * map_B.transpose();
  } else {
    output_map = map_A * map_B;
  }
}

template <>
inline void SparseDenseMatMulImpl<float>(const ComputeCtx& ctx, const ConstSparseMatrixMap<float>& map_A,
                                         const ConstEigenMatrixMapRowMajor<float>& map_B, EigenMatrixMapRowMajor<float>& output_map) {
  if (ctx.trans_A && ctx.trans_B) {
    output_map = map_A.transpose() * ctx.alpha * map_B.transpose();
  } else if (ctx.trans_A && !ctx.trans_B) {
    output_map = map_A.transpose() * ctx.alpha * map_B;
  } else if (!ctx.trans_A && ctx.trans_B) {
    output_map = map_A * ctx.alpha * map_B.transpose();
  } else {
    output_map = map_A * ctx.alpha * map_B;
  }
}

// Handle CSR sparse format using Eigen
template <class T>
struct SparseToDenseCsr {
  void operator()(const ComputeCtx& ctx, const SparseTensor& A, const Tensor& B, Tensor& output) const {
    const auto& a_dims = A.DenseShape().GetDims();
    const auto& b_dims = B.Shape().GetDims();
    const auto& out_dims = output.Shape().GetDims();
    auto csr_view = A.AsCsr();

    ConstSparseMatrixMap<T> map_A(a_dims[0], a_dims[1], A.NumValues(),
                                  csr_view.Outer().Data<int64_t>(),
                                  csr_view.Inner().Data<int64_t>(),
                                  A.Values().Data<T>());
    ConstEigenMatrixMapRowMajor<T> map_B(B.Data<T>(), b_dims[0], b_dims[1]);
    EigenMatrixMapRowMajor<T> output_map(output.MutableData<T>(), out_dims[0], out_dims[1]);
    // XXX: Consider re-writing it as a parallel loop as Eigen requires it to use OpenMP
    // XXX: Consider vectorization
    SparseDenseMatMulImpl(ctx, map_A, map_B, output_map);
  }
};

#endif  //!defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)

template <typename T>
inline T Mul(T a_value, float, T b_value) {
  return a_value * b_value;
}

template <>
inline float Mul<float>(float a_value, float alpha, float b_value) {
  return a_value * alpha * b_value;
}

// Inspired by TensorFlow SparseTensorDenseMatmul
template <typename T>
struct SparseToDenseCoo {
  Status operator()(const ComputeCtx& ctx, const SparseTensor& A, const Tensor& B, Tensor& output) const {
    const auto& b_dims = B.Shape().GetDims();
    const auto& out_dims = output.Shape().GetDims();
    const auto nnz = A.NumValues();

    auto a_values = A.Values().DataAsSpan<T>();
    auto coo_view = A.AsCoo();
    const auto& ind_dims = coo_view.Indices().Shape().GetDims();
    ORT_RETURN_IF_NOT(ind_dims.size() == 2, "COO indices must be 2-D, got: ", ind_dims.size());

    ConstEigenMatrixMapRowMajor<int64_t> a_indicies_map(coo_view.Indices().Data<int64_t>(), ind_dims[0], ind_dims[1]);
    ConstEigenMatrixMapRowMajor<T> map_b(B.Data<T>(), b_dims[0], b_dims[1]);
    EigenMatrixMapRowMajor<T> output_map(output.MutableData<T>(), out_dims[0], out_dims[1]);
    output_map.setZero();

    const auto rhs_right = (ctx.trans_B) ? b_dims[0] : b_dims[1];
    const auto lhs_right = (ctx.trans_B) ? b_dims[1] : b_dims[0];
    const int lhs_index_a = (ctx.trans_A) ? 1 : 0;
    const int rhs_index_a = (ctx.trans_A) ? 0 : 1;
    const auto out_left = out_dims[0];

    // XXX: Make this parallel
    for (size_t i = 0; i < nnz; ++i) {
      const auto m = a_indicies_map(i, lhs_index_a);
      const auto k = a_indicies_map(i, rhs_index_a);
      ORT_RETURN_IF_NOT(k < lhs_right, "COO k index: ", k, " is out of bounds of lhs_right: ", lhs_right);
      ORT_RETURN_IF_NOT(m < out_left, "COO m index: ", m, " is out of bounds of out_left: ", out_left);
      const T a_value = a_values[i];
      for (int64_t n = 0; n < rhs_right; ++n) {
        const T b_value = (ctx.trans_B) ? map_b(n, k) : map_b(k, n);
        output_map(m, n) += Mul(a_value, ctx.alpha, b_value);
      }
    }

    return Status::OK();
  }
};

}  // namespace

Status SparseToDenseMatMul::Compute(OpKernelContext* ctx) const {
  // We currently do not support batching, but may do so in the future
  const auto* A = ctx->Input<SparseTensor>(0);
  const auto* B = ctx->Input<Tensor>(1);

  const auto& A_shape = A->DenseShape();
  const auto& B_shape = B->Shape();

  ORT_RETURN_IF_NOT(A_shape.NumDimensions() == 2, "Currently supporting only 2-D matrices");
  ORT_RETURN_IF_NOT(B_shape.NumDimensions() == 2, "Currently supporting only 2-D matrices");

  const auto& a_dims = A_shape.GetDims();
  const auto& b_dims = B_shape.GetDims();

  const auto outer_A = (trans_a_attr_) ? a_dims[1] : a_dims[0];
  const auto inner_A = (trans_a_attr_) ? a_dims[0] : a_dims[1];
  const auto inner_B = (trans_b_attr_) ? b_dims[1] : b_dims[0];
  const auto outer_B = (trans_b_attr_) ? b_dims[0] : b_dims[1];

  ORT_RETURN_IF_NOT(inner_A == inner_B, "Can not multiply A and B as inner dimension does not match. inner_A: ",
                    inner_A, " vs inner_B: ", inner_B);

  TensorShape output_shape{outer_A, outer_B};
  auto* output = ctx->Output(0, output_shape);

  utils::MLTypeCallDispatcherFromTypeList<SparseToDenseSupportedTypes> t_disp(A->GetElementType());
  // I am not expecting to do the below in every kernel but this is a reference
  // implementation to show the expectations.
  ComputeCtx compute_ctx{trans_a_attr_ != 0, trans_b_attr_ != 0, alpha_attr_};
  if (A->Format() == SparseFormat::kCoo) {
    auto coo_view = A->AsCoo();
    const auto num_dims = coo_view.Indices().Shape().NumDimensions();
    ORT_RETURN_IF_NOT(num_dims == 2, "Expecting COO 2-D indices shape");
    ORT_RETURN_IF_NOT(A->Values().Shape().Size() * 2 == coo_view.Indices().Shape().Size(), "Expecting 2xValues == indices");
    auto status = t_disp.InvokeRet<Status, SparseToDenseCoo>(compute_ctx, *A, *B, *output);
    ORT_RETURN_IF_ERROR(status);
// Eigen has a bug in x86 where it calculates reallocation size as -1
// and throws bad_alloc
#if !defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)
  } else if (A->Format() == SparseFormat::kCsrc) {
    auto csr_view = A->AsCsr();
    ORT_RETURN_IF_NOT(A->Values().Shape().Size() == csr_view.Inner().Shape().Size(),
                      "Expecting the same number NNZ == size of Inner indices");
    ORT_RETURN_IF_NOT((A_shape.GetDims()[0] + 1) == csr_view.Outer().Shape().Size(), "Outer size must be M + 1");
    t_disp.Invoke<SparseToDenseCsr>(compute_ctx, *A, *B, *output);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Currently support only COO and CSR(x64) formats");
  }
#else
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "WASM and 32-bit builds support only COO format");
  }
#endif  //!defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)

  return Status::OK();
}

namespace {
using SparseGemmSupportedTypes = TypeList<float, int64_t>;
}

class SparseToSparseMatMul final : public OpKernel {
 public:
  explicit SparseToSparseMatMul(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
    int64_t trans_a_attr, trans_b_attr;
    info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr, 0);
    info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr, 0);
    trans_a_attr_ = trans_a_attr != 0;
    trans_b_attr_ = trans_b_attr != 0;
  }

  Status Compute(OpKernelContext*) const override;

 private:
  Status EigenCompute(const SparseTensor& input_A,
                      const SparseTensor& input_B,
                      std::vector<int64_t> a_dims,
                      std::vector<int64_t> b_dims,
                      SparseTensor& output_tensor) const;

  Status ComputeImpl(const SparseTensor& input_A,
                     const SparseTensor& input_B,
                     std::vector<int64_t> a_dims,
                     std::vector<int64_t> b_dims,
                     SparseTensor& output_tensor) const;

  float alpha_attr_;
  bool trans_a_attr_;
  bool trans_b_attr_;
};

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefSparseConstraintsFromTypeList<SparseGemmSupportedTypes>())
        .TypeConstraint("T1", BuildKernelDefConstraintsFromTypeList<SparseGemmSupportedTypes>()),
    SparseToSparseMatMul);

namespace {
Status CheckOutputDims(int64_t result_rows, int64_t result_cols, SparseTensor& output_tensor) {
  const auto output_rows = (output_tensor.DenseShape().NumDimensions() == 1)
                               ? 1
                               : output_tensor.DenseShape().GetDims()[0];

  ORT_RETURN_IF_NOT(output_rows == result_rows, "Result rows does not match output tensor rows");

  const auto output_cols = (output_tensor.DenseShape().NumDimensions() == 1)
                               ? output_tensor.DenseShape().GetDims()[0]
                               : output_tensor.DenseShape().GetDims()[1];

  ORT_RETURN_IF_NOT(output_cols == result_cols, "Result cols does not match output tensor cols");

  return Status::OK();
}

Status CheckDimsAndCopyResult(size_t nnz, int64_t result_rows, int64_t result_cols,
                              void* result_values_ptr, gsl::span<const int64_t> inner_span,
                              gsl::span<const int64_t> outer_span,
                              SparseTensor& output_tensor) {
  if (nnz == 0) {
    ORT_IGNORE_RETURN_VALUE(output_tensor.MakeCooData(0, 0));
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(CheckOutputDims(result_rows, result_cols, output_tensor));

  TensorShape result_values_shape{static_cast<int64_t>(nnz)};
  Tensor result_values(output_tensor.DataType(), result_values_shape, result_values_ptr, output_tensor.Location());
  auto coo_mutator = output_tensor.MakeCooData(nnz, nnz);
  sparse_utils::CopyCpuTensor(result_values, coo_mutator.Values());

  ORT_RETURN_IF_ERROR(sparse_utils::ConvertCsrIndicesToCooIndices(result_cols, inner_span, outer_span,
                                                                  coo_mutator.Indices().template MutableDataAsSpan<int64_t>()));

  return Status::OK();
}

inline bool IsVector(const std::vector<int64_t>& dims) {
  return (dims.size() == 1 || (dims.size() == 2 && (dims[0] == 1 || dims[1] == 1)));
}

inline bool IsRowVector(const std::vector<int64_t>& computed_dims, bool transpose) {
  return (computed_dims.size() == 2 && (transpose) ? computed_dims[1] == 1 : computed_dims[0] == 1);
}

inline bool IsColVector(const std::vector<int64_t>& computed_dims, bool transpose) {
  return (computed_dims.size() == 2 && (transpose) ? computed_dims[0] == 1 : computed_dims[1] == 1);
}

}  // namespace

#if !defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)
namespace {

struct SparseToSparseComputeCtx {
  bool transA_;
  bool transB_;
  float alpha_;
  const Tensor& values_A_;
  const sparse_utils::CsrIndicesSpan& csr_A_;
  const Tensor& values_B_;
  const sparse_utils::CsrIndicesSpan& csr_B_;
  SparseTensor& output_;
};

template <typename T>
inline SparseMatrix<T> SparseToSparseMatMulImpl(const SparseToSparseComputeCtx& ctx,
                                                const ConstSparseMatrixMap<T>& map_A,
                                                const ConstSparseMatrixMap<T>& map_B) {
  if (ctx.transA_ && ctx.transB_) {
    return map_A.transpose() * map_B.transpose();
  } else if (ctx.transA_ && !ctx.transB_) {
    return map_A.transpose() * map_B;
  } else if (!ctx.transA_ && ctx.transB_) {
    return map_A * map_B.transpose();
  }
  return map_A * map_B;
}

template <>
inline SparseMatrix<float> SparseToSparseMatMulImpl<float>(const SparseToSparseComputeCtx& ctx,
                                                           const ConstSparseMatrixMap<float>& map_A,
                                                           const ConstSparseMatrixMap<float>& map_B) {
  if (ctx.transA_ && ctx.transB_) {
    return map_A.transpose() * ctx.alpha_ * map_B.transpose();
  } else if (ctx.transA_ && !ctx.transB_) {
    return map_A.transpose() * ctx.alpha_ * map_B;
  } else if (!ctx.transA_ && ctx.transB_) {
    return map_A * ctx.alpha_ * map_B.transpose();
  }
  return map_A * ctx.alpha_ * map_B;
}

template <class T>
struct SparseToSparseEigenMatrixB {
  Status operator()(const SparseToSparseComputeCtx& ctx,
                    const std::vector<int64_t>& computed_a_dims,
                    const std::vector<int64_t>& computed_b_dims) const {
    ConstSparseMatrixMap<T> map_A(computed_a_dims[0], computed_a_dims[1],
                                  ctx.values_A_.Shape().Size(),
                                  ctx.csr_A_.Outer().data(),
                                  ctx.csr_A_.Inner().data(),
                                  ctx.values_A_.template Data<T>());

    ConstSparseMatrixMap<T> map_B(computed_b_dims[0], computed_b_dims[1],
                                  ctx.values_B_.Shape().Size(),
                                  ctx.csr_B_.Outer().data(),
                                  ctx.csr_B_.Inner().data(),
                                  ctx.values_B_.template Data<T>());

    SparseMatrix<T> result = SparseToSparseMatMulImpl(ctx, map_A, map_B);
    const auto nnz = gsl::narrow<size_t>(result.nonZeros());
    const auto rows = gsl::narrow<size_t>(result.rows());
    gsl::span<const int64_t> inner_span = gsl::make_span(result.innerIndexPtr(), nnz);
    gsl::span<const int64_t> outer_span = gsl::make_span(result.outerIndexPtr(), rows + 1);
    return CheckDimsAndCopyResult(nnz, result.rows(), result.cols(), result.valuePtr(), inner_span, outer_span, ctx.output_);
  }
};

}  // namespace

Status SparseToSparseMatMul::EigenCompute(const SparseTensor& input_A,
                                          const SparseTensor& input_B,
                                          std::vector<int64_t> a_dims,
                                          std::vector<int64_t> b_dims,
                                          SparseTensor& output_tensor) const {
  sparse_utils::CsrIndicesSpan csr_A;
  ORT_RETURN_IF_ERROR(sparse_utils::GetCsrIndicesAndMaybeConvert(a_dims, input_A, csr_A));

  const bool a_is_vector = IsVector(a_dims);
  bool a_transpose = trans_a_attr_;
  if (a_is_vector) {
    // For vectors conversion to CSR takes place as if it was a row vector
    // therefore, we need to flip transpose flag and swap dims
    if (IsColVector(a_dims, false)) {
      a_transpose = !a_transpose;
      std::swap(a_dims[0], a_dims[1]);
    }
  }

  sparse_utils::CsrIndicesSpan csr_B;
  ORT_RETURN_IF_ERROR(sparse_utils::GetCsrIndicesAndMaybeConvert(b_dims, input_B, csr_B));

  const bool b_is_vector = IsVector(b_dims);
  bool b_transpose = trans_b_attr_;
  if (b_is_vector) {
    // For vectors conversion to CSR takes place as if it was a row vector
    // therefore, we need to flip transpose flag and swap dims
    if (IsColVector(b_dims, false)) {
      b_transpose = !b_transpose;
      std::swap(b_dims[0], b_dims[1]);
    }
  }

  utils::MLTypeCallDispatcherFromTypeList<SparseGemmSupportedTypes> t_disp(input_A.GetElementType());

  SparseToSparseComputeCtx compute_ctx{a_transpose, b_transpose, alpha_attr_,
                                       input_A.Values(), csr_A,
                                       input_B.Values(), csr_B,
                                       output_tensor};

  return t_disp.InvokeRet<Status, SparseToSparseEigenMatrixB>(compute_ctx, a_dims, b_dims);
}

#else

Status SparseToSparseMatMul::EigenCompute(const SparseTensor&,
                                          const SparseTensor&,
                                          std::vector<int64_t>,
                                          std::vector<int64_t>,
                                          SparseTensor&) const {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Eigen Gemm sparse not supported on 32-bit builds");
}

#endif  // !defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)

namespace {

// Used for RowVector to ColVector Mul
// The functor will multiply all non-zeros in A to all non-zeros in B
// and will produce appropriate indices for each of the products.
template <typename T>
struct ScalarAccumulate {
  void operator()(const Tensor& A_values, const Tensor& B_values, float alpha,
                  const sparse_utils::IndicesSpan& a_indices, const sparse_utils::IndicesSpan& b_indices,
                  Tensor& output_values) const {
    const T* a_data = A_values.Data<T>();
    const T* b_data = B_values.Data<T>();
    T accum = 0;

    auto match_cb = [&accum, a_data, alpha, b_data](size_t a_offset, size_t b_offset) {
      accum += Mul(a_data[a_offset], alpha, b_data[b_offset]);
    };

    sparse_utils::ScanForSparseMatches(a_indices.Get(), b_indices.Get(), match_cb);
    *output_values.MutableData<T>() = accum;
  }
};

size_t CountNonEmptyRows(const gsl::span<const int64_t>& outer_indices) {
  const size_t outer_limit = outer_indices.size() - 1;
  size_t non_empty_rows = 0;
  for (size_t row = 0; row < outer_limit; ++row) {
    const auto row_indices_start = gsl::narrow<gsl::index>(outer_indices[row]);
    const auto row_nnz_len = gsl::narrow<gsl::index>(outer_indices[row + 1]) - row_indices_start;
    if (row_nnz_len > 0) {
      ++non_empty_rows;
    }
  }
  return non_empty_rows;
}

// Used for ColVector to RowVector Mul
struct VectorOuterProductCtx {
  const SparseTensor& input_A_;
  const SparseTensor& input_B_;
  // 1d coo indices
  const sparse_utils::IndicesSpan& a_indices_;
  const sparse_utils::IndicesSpan& b_indices_;
  SparseTensor& output_tensor_;
};

template <typename T>
struct VectorOuterProduct {
  Status operator()(const VectorOuterProductCtx& ctx, float alpha) const {
    const auto a_values = ctx.input_A_.Values().DataAsSpan<T>();
    const auto b_values = ctx.input_B_.Values().DataAsSpan<T>();
    const auto& a_indices = ctx.a_indices_.Get();
    const auto& b_indices = ctx.b_indices_.Get();
    ORT_RETURN_IF_NOT(a_values.size() == a_indices.size(), "A values size does not match A indices size");
    ORT_RETURN_IF_NOT(b_values.size() == b_indices.size(), "B values size does not match B indices size");

    const auto cols = ctx.output_tensor_.DenseShape().GetDims()[1];
    // We know output is going to be m*n products of non-zeros (including app defined zeros)
    const size_t output_size = a_values.size() * b_values.size();
    auto coo_mutator = ctx.output_tensor_.MakeCooData(output_size, output_size);
    T* output_data = coo_mutator.Values().MutableData<T>();
    int64_t* output_indices = coo_mutator.Indices().MutableData<int64_t>();
    const size_t b_limit = b_indices.size();
    for (size_t a_i = 0, a_limit = a_indices.size(); a_i < a_limit; ++a_i) {
      const T a_val = a_values[a_i];
      const auto dest_row = a_indices[a_i];
      const auto dest_row_offset = dest_row * cols;
      for (size_t b_i = 0; b_i < b_limit; ++b_i) {
        const auto dest_col = b_indices[b_i];
        *output_indices++ = dest_row_offset + dest_col;
        *output_data++ = Mul(a_val, alpha, b_values[b_i]);
      }
    }
    return Status::OK();
  }
};

struct MatrixToVectorCtx {
  const Tensor& matrix_values_;
  const sparse_utils::CsrIndicesSpan& mat_csr_;
  const Tensor& vector_values_;
  const sparse_utils::IndicesSpan& b_1d_coo_;
  SparseTensor& output_tensor_;
};

template <typename T>
class MatrixToVectorCallback {
 protected:
  gsl::span<const T> matrix_values_;
  gsl::span<const T> vector_values_;
  float alpha_ = 1.f;
  size_t row_offset_ = 0;
  size_t row_ind_size_ = 0;
  T accum_ = 0;

 public:
  MatrixToVectorCallback() = default;

  MatrixToVectorCallback(const MatrixToVectorCtx& ctx, float alpha)
      : matrix_values_(ctx.matrix_values_.DataAsSpan<T>()),
        vector_values_(ctx.vector_values_.DataAsSpan<T>()),
        alpha_(alpha) {
  }

  // Must be set on on each row
  void SetRow(size_t offset, size_t row_ind_size) noexcept {
    row_offset_ = offset;
    row_ind_size_ = row_ind_size;
    accum_ = 0;
  }

  void operator()(size_t a_offset, size_t b_offset) {
    ORT_ENFORCE(a_offset < row_ind_size_, "a_offset is out of bounds");
    auto m_offset = row_offset_ + a_offset;
    accum_ += Mul(matrix_values_[m_offset], alpha_, vector_values_[b_offset]);
  }

  T GetRowSum() const noexcept { return accum_; }
};

template <typename T>
class MatrixTransposeToVectorCallback : public MatrixToVectorCallback<T> {
 private:
  gsl::span<const size_t> transposed_value_offsets_;

 public:
  MatrixTransposeToVectorCallback() = default;
  MatrixTransposeToVectorCallback(const MatrixToVectorCtx& ctx, float alpha)
      : MatrixToVectorCallback(ctx, alpha),
        transposed_value_offsets_(ctx.mat_csr_.TransposedOffsets()) {}
  void operator()(size_t a_offset, size_t b_offset) {
    ORT_ENFORCE(a_offset < row_ind_size_, "a_offset is out of bounds");
    const auto m_offset = transposed_value_offsets_[row_offset_ + a_offset];
    accum_ += Mul(matrix_values_[m_offset], alpha_, vector_values_[b_offset]);
  }
};

// Performs multiplication of matrix to a column vector (or row vector to a matrix)
// A(m, n) B(n, 1) = C(m, 1) output is a column vector
// Providing the matrix is properly transposed, we can also do the reverse
// with the same code
// A(1, m) B(m, n) = C(1, n) => row vector
template <typename T>
struct MatrixToVector {
  Status operator()(const MatrixToVectorCtx& ctx, float alpha) const {
    const auto& mat_inner = ctx.mat_csr_.Inner();
    const auto& mat_outer = ctx.mat_csr_.Outer();
    // We know that A is not a fully sparse matrix at this point
    // we must have at least 1 row (actually more since it is not a vector and we handle vectors separately)
    // and outer indices must be at least 2 (for 1 row) or more in size.
    ORT_RETURN_IF_NOT(mat_outer.size() > 1, "A outer indices must have at least 2 in size");

    std::function<void(size_t, size_t)> cb_func;
    MatrixToVectorCallback<T> cb;
    MatrixTransposeToVectorCallback<T> cb_trans;
    MatrixToVectorCallback<T>* cb_in_use;
    if (ctx.mat_csr_.IsTransposed()) {
      cb_trans = MatrixTransposeToVectorCallback<T>(ctx, alpha);
      cb_func = [&cb_trans](size_t a_offset, size_t b_offset) {
        cb_trans(a_offset, b_offset);
      };
      cb_in_use = &cb_trans;
    } else {
      cb = MatrixToVectorCallback<T>(ctx, alpha);
      cb_func = [&cb](size_t a_offset, size_t b_offset) {
        cb(a_offset, b_offset);
      };
      cb_in_use = &cb;
    }

    // Scan all the rows and count those that contain any nnz
    const size_t output_size = CountNonEmptyRows(mat_outer);

    auto coo_mutator = ctx.output_tensor_.MakeCooData(output_size, output_size);
    T* output_data = coo_mutator.Values().MutableData<T>();
    int64_t* output_indices = coo_mutator.Indices().MutableData<int64_t>();

    // XXX: This can be parallelized
    // We multiply each row nnz of the of the A matrix to vector B nnz.
    const size_t outer_limit = mat_outer.size() - 1;
    for (size_t row = 0; row < outer_limit; ++row) {
      const auto row_indices_start = gsl::narrow<gsl::index>(mat_outer[row]);
      const auto row_nnz_len = gsl::narrow<gsl::index>(mat_outer[row + 1]) - row_indices_start;
      // We skip rows with no nnz, assuming that application defined zeros still show up in the nnz
      if (row_nnz_len > 0) {
        cb_in_use->SetRow(row_indices_start, row_nnz_len);
        // Each row of indices has row relative offsets (column number)
        auto row_indices_span = mat_inner.subspan(row_indices_start, row_nnz_len);
        sparse_utils::ScanForSparseMatches(row_indices_span, ctx.b_1d_coo_.Get(), cb_func);
        *output_data++ = cb_in_use->GetRowSum();
        *output_indices++ = gsl::narrow<int64_t>(row);
      }
    }
    return Status::OK();
  }
};

struct MatrixToMatrixCtx {
  const Tensor& a_values_;
  const sparse_utils::CsrIndicesSpan& a_csr_;
  const Tensor& b_values_;
  const sparse_utils::CsrIndicesSpan& b_csr_;
  SparseTensor& output_tensor_;
};

template <typename T>
struct MatrixToMatrix {
  Status operator()(const MatrixToMatrixCtx& ctx, float alphat) const {
    std::function<void(size_t, size_t)> cb_func;
    const auto& a_inner = ctx.a_csr_.Inner();
    const auto& a_outer = ctx.a_csr.Outer();
    const auto& b_inner = ctx.b_csr_.Inner();
    const auto& b_outer = ctx.b_csr_.Outer();

    const size_t a_outer_limit = a_outer.size() - 1;
    const size_t b_outer_limit = b_outer.size() - 1;
    for (size_t row_a = 0; row_a < a_outer_limit; ++row_a) {
      const auto a_row_start = gsl::narrow<gsl::index>(a_outer[row_a]);
      const auto a_row_nnz_len = gsl::narrow<gsl::index>(a_outer[row_a + 1]) - a_row_start;

      if (a_row_nnz_len > 0) {
        auto a_idx_span = a_inner.subspan(a_row_start, a_row_nnz_len);
        for (size_t col_b = 0; col_b < b_outer_limit; ++col_b) {
          const auto b_col_start = gsl::narrow<gsl::index>(b_outer[col_b]);
          const auto b_col_nnz_len = gsl::narrow<gsl::index>(b_outer[col_b + 1]) - b_col_start;
          // We skip rows with no nnz, assuming that application defined zeros still show up in the nnz
          if (b_col_nnz_len > 0) {
            // cb_in_use->SetRow(row_indices_start, row_nnz_len);
            auto b_idx_span = b_inner.subspan(b_col_start, b_col_nnz_len);
            sparse_utils::ScanForSparseMatches(a_idx_span, b_idx_span, cb_func);
            // *output_data++ = cb_in_use->GetRowSum();
            *output_indices++ = gsl::narrow<int64_t>(row_a);
          }
        }
      }
    }
  }
};

}  // namespace

Status SparseToSparseMatMul::ComputeImpl(const SparseTensor& input_A, const SparseTensor& input_B,
                                         std::vector<int64_t> a_dims, std::vector<int64_t> b_dims,
                                         SparseTensor& output_tensor) const {
  const bool a_is_vector = IsVector(a_dims);
  const bool b_is_vector = IsVector(b_dims);

  if (!a_is_vector && !b_is_vector) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Non-Eigen impl does not support this combination yet");
  }

  sparse_utils::CsrIndicesSpan csr_A;
  sparse_utils::IndicesSpan a_1d_coo;
  if (a_is_vector) {
    ORT_RETURN_IF_ERROR(sparse_utils::GetCoo1DIndicesAndMaybeConvert(input_A, a_1d_coo));
  } else {
    if (b_is_vector) {
      // A Matrix to B vector
      if (!trans_a_attr_) {
        ORT_RETURN_IF_ERROR(sparse_utils::GetCsrIndicesAndMaybeConvert(a_dims, input_A, csr_A));
      } else {
        // Transpose indices
        ORT_RETURN_IF_ERROR(sparse_utils::GetCsrIndicesAndTranspose(a_dims, input_A, csr_A));
      }
    }
  }

  sparse_utils::CsrIndicesSpan csr_B;
  sparse_utils::IndicesSpan b_1d_coo;
  if (b_is_vector) {
    ORT_RETURN_IF_ERROR(sparse_utils::GetCoo1DIndicesAndMaybeConvert(input_B, b_1d_coo));
  } else {
    if (a_is_vector) {
      // We do reverse of the transpose flag bc we switch operands places, no need to transpose the vector
      // or the product since the product is a vector
      if (trans_b_attr_) {
        ORT_RETURN_IF_ERROR(sparse_utils::GetCsrIndicesAndMaybeConvert(b_dims, input_B, csr_B));
      } else {
        // Transpose indices
        ORT_RETURN_IF_ERROR(sparse_utils::GetCsrIndicesAndTranspose(b_dims, input_B, csr_B));
      }
    }
  }

  utils::MLTypeCallDispatcherFromTypeList<SparseGemmSupportedTypes> t_disp(input_A.GetElementType());
  if (a_is_vector && b_is_vector) {
    if (IsRowVector(a_dims, trans_a_attr_) && IsColVector(b_dims, trans_b_attr_)) {
      // Result is a scalar with dense shape [1, 1] and coordinates (0, 0)
      // need to multiply corresponding elements of the vectors
      assert(!a_1d_coo.Get().empty());
      assert(!b_1d_coo.Get().empty());
      auto coo_mutator = output_tensor.MakeCooData(1, 1);
      int64_t* output_indices = coo_mutator.Indices().MutableData<int64_t>();
      *output_indices = 0;
      t_disp.Invoke<ScalarAccumulate>(input_A.Values(), input_B.Values(), alpha_attr_,
                                      a_1d_coo, b_1d_coo,
                                      coo_mutator.Values());
      return Status::OK();
    } else if (IsColVector(a_dims, trans_a_attr_) && IsRowVector(b_dims, trans_b_attr_)) {
      // Result is a matrix [m, n] of everything non-zero in A to everything non-zero in B
      assert(!a_1d_coo.Get().empty());
      assert(!b_1d_coo.Get().empty());
      VectorOuterProductCtx compute_ctx{input_A, input_B, a_1d_coo, b_1d_coo, output_tensor};
      return t_disp.InvokeRet<Status, VectorOuterProduct>(compute_ctx, alpha_attr_);
    }
  } else if (!a_is_vector && b_is_vector) {
    // Result is a column vector
    assert(!csr_A.Inner().empty());
    assert(!b_1d_coo.Get().empty());
    MatrixToVectorCtx compute_ctx{input_A.Values(), csr_A,
                                  input_B.Values(), b_1d_coo,
                                  output_tensor};
    return t_disp.InvokeRet<Status, MatrixToVector>(compute_ctx, alpha_attr_);
  } else if (a_is_vector && !b_is_vector) {
    // This requires B indices to be transposed
    // because we use B matrix as the first operand.
    // A vector remains the same.
    assert(csr_B.IsTransposed());
    assert(!csr_B.Inner().empty());
    assert(!csr_B.Outer().empty());
    assert(!a_1d_coo.Get().empty());
    MatrixToVectorCtx compute_ctx{input_B.Values(), csr_B,
                                  input_A.Values(), a_1d_coo,
                                  output_tensor};
    return t_disp.InvokeRet<Status, MatrixToVector>(compute_ctx, alpha_attr_);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported operand combination");
}

Status SparseToSparseMatMul::Compute(OpKernelContext* ctx) const {
  const SparseTensor& input_A = *ctx->Input<SparseTensor>(0);
  const SparseTensor& input_B = *ctx->Input<SparseTensor>(1);

  ORT_RETURN_IF_NOT(input_A.Format() == SparseFormat::kCoo && input_B.Format() == SparseFormat::kCoo,
                    "Currently support only COO format");

  const auto& A_shape = input_A.DenseShape();
  const auto& B_shape = input_B.DenseShape();

  ORT_RETURN_IF_NOT(A_shape.NumDimensions() > 0 && A_shape.NumDimensions() <= 2, "Currently supporting only 1 and 2-D matrices");
  ORT_RETURN_IF_NOT(B_shape.NumDimensions() > 0 && B_shape.NumDimensions() <= 2, "Currently supporting only 1 and 2-D matrices");

  auto a_dims = A_shape.GetDims();
  auto b_dims = B_shape.GetDims();
  if (a_dims.size() == 1) {
    a_dims.insert(a_dims.begin(), 1);
  }

  if (b_dims.size() == 1) {
    b_dims.insert(b_dims.end(), 1);
  }

  const auto outer_A = (trans_a_attr_) ? a_dims[1] : a_dims[0];
  const auto inner_A = (trans_a_attr_) ? a_dims[0] : a_dims[1];
  const auto inner_B = (trans_b_attr_) ? b_dims[1] : b_dims[0];
  const auto outer_B = (trans_b_attr_) ? b_dims[0] : b_dims[1];

  ORT_RETURN_IF_NOT(inner_A == inner_B, "Can not multiply A and B as inner dimension does not match. inner_A: ",
                    inner_A, " vs inner_B: ", inner_B);

  std::vector<int64_t> output_dims{outer_A, outer_B};
  SparseTensor& output_tensor = *ctx->OutputSparse(0, output_dims);
  if (input_A.NumValues() == 0 || input_B.NumValues() == 0) {
    // All zeros as a result.
    ORT_IGNORE_RETURN_VALUE(output_tensor.MakeCooData(0, 0));
    return Status::OK();
  }

  // return EigenCompute(input_A, input_B, a_dims, b_dims, output_tensor);
  return ComputeImpl(input_A, input_B, a_dims, b_dims, output_tensor);
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  //!defined(DISABLE_SPARSE_TENSORS)