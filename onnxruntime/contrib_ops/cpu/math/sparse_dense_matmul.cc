// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_SPARSE_TENSORS)

#include "core/framework/sparse_tensor.h"
#include "core/common/narrow.h"
#include "core/providers/cpu/math/gemm_matmul_common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

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
        .TypeConstraint("T", BuildKernelDefSparseConstraints<float, double, int32_t, int64_t, uint32_t, uint64_t>())
        .TypeConstraint("T1", BuildKernelDefConstraints<float, double, int32_t, int64_t, uint32_t, uint64_t>()),
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
                                         const ConstEigenMatrixMapRowMajor<float>& map_B,
                                         EigenMatrixMapRowMajor<float>& output_map) {
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
    const Eigen::Index* inner_index_pointer = nullptr;
    const Eigen::Index* outer_index_pointer = nullptr;
    std::unique_ptr<Eigen::Index[]> buffer_holder_inner, buffer_holder_outer;
    if constexpr (std::is_integral<Eigen::Index>::value &&
                  std::is_signed<Eigen::Index>::value &&
                  (sizeof(Eigen::Index) == sizeof(int64_t))) {
      // On macOS the following reinterpret_cast is necessary because Eigen::Index is an alias of `long` but int64_t is
      // `long long`. Though they have the same size, compilers still do not allow an implicit casting between them.
      inner_index_pointer = reinterpret_cast<const Eigen::Index*>(csr_view.Inner().Data<int64_t>());
      outer_index_pointer = reinterpret_cast<const Eigen::Index*>(csr_view.Outer().Data<int64_t>());
    } else {
      const size_t inner_size = narrow<size_t>(csr_view.Inner().Shape().Size());
      const size_t outer_size = narrow<size_t>(csr_view.Outer().Shape().Size());
      buffer_holder_inner.reset(new Eigen::Index[inner_size]);
      buffer_holder_outer.reset(new Eigen::Index[outer_size]);
      inner_index_pointer = buffer_holder_inner.get();
      outer_index_pointer = buffer_holder_outer.get();
      const int64_t* inner_begin = csr_view.Inner().Data<int64_t>();
      const int64_t* outer_begin = csr_view.Outer().Data<int64_t>();
      std::transform<const int64_t*, Eigen::Index*>(inner_begin, inner_begin + inner_size,
                                                    buffer_holder_inner.get(), [](int64_t v) -> Eigen::Index {
                                                      return narrow<Eigen::Index>(v);
                                                    });
      std::transform<const int64_t*, Eigen::Index*>(outer_begin, outer_begin + outer_size,
                                                    buffer_holder_outer.get(), [](int64_t v) -> Eigen::Index {
                                                      return narrow<Eigen::Index>(v);
                                                    });
    }
    ConstSparseMatrixMap<T> map_A(narrow<Eigen::Index>(a_dims[0]), narrow<Eigen::Index>(a_dims[1]),
                                  narrow<Eigen::Index>(A.NumValues()), outer_index_pointer, inner_index_pointer,
                                  A.Values().Data<T>());
    ConstEigenMatrixMapRowMajor<T> map_B(B.Data<T>(), narrow<Eigen::Index>(b_dims[0]), narrow<Eigen::Index>(b_dims[1]));
    EigenMatrixMapRowMajor<T> output_map(output.MutableData<T>(), narrow<Eigen::Index>(out_dims[0]),
                                         narrow<Eigen::Index>(out_dims[1]));
    // XXX: Consider re-writing it as a parallel loop as Eigen requires it to use OpenMP
    // XXX: Consider vectorization
    SparseDenseMatMulImpl(ctx, map_A, map_B, output_map);
  }
};

#endif  //! defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)

template <typename T>
inline T Mul(T a_value, float, T b_value) {
  return a_value * b_value;
}

template <>
inline constexpr float Mul<float>(float a_value, float alpha, float b_value) {
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
    ConstEigenMatrixMapRowMajor<int64_t> a_indicies_map(coo_view.Indices().Data<int64_t>(), narrow<size_t>(ind_dims[0]),
                                                        narrow<size_t>(ind_dims[1]));
    ConstEigenMatrixMapRowMajor<T> map_b(B.Data<T>(), narrow<size_t>(b_dims[0]), narrow<size_t>(b_dims[1]));
    EigenMatrixMapRowMajor<T> output_map(output.MutableData<T>(), narrow<size_t>(out_dims[0]),
                                         narrow<size_t>(out_dims[1]));
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
        const T b_value =
            (ctx.trans_B) ? map_b(narrow<size_t>(n), narrow<size_t>(k)) : map_b(narrow<size_t>(k), narrow<size_t>(n));
        output_map(narrow<size_t>(m), narrow<size_t>(n)) += Mul(a_value, ctx.alpha, b_value);
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

  ORT_RETURN_IF_NOT(inner_A == inner_B,
                    "Can not multiply A and B as inner dimension does not match. inner_A: ", inner_A,
                    " vs inner_B: ", inner_B);

  TensorShape output_shape{outer_A, outer_B};
  auto* output = ctx->Output(0, output_shape);

  utils::MLTypeCallDispatcher<float, double, int32_t, uint32_t, int64_t, uint64_t> t_disp(A->GetElementType());
  // I am not expecting to do the below in every kernel but this is a reference
  // implementation to show the expectations.
  ComputeCtx compute_ctx{trans_a_attr_ != 0, trans_b_attr_ != 0, alpha_attr_};
  if (A->Format() == SparseFormat::kCoo) {
    auto coo_view = A->AsCoo();
    const auto num_dims = coo_view.Indices().Shape().NumDimensions();
    ORT_RETURN_IF_NOT(num_dims == 2, "Expecting COO 2-D indices shape");
    ORT_RETURN_IF_NOT(A->Values().Shape().Size() * 2 == coo_view.Indices().Shape().Size(),
                      "Expecting 2xValues == indices");
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
#endif  //! defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__) && !defined(__ANDROID__)

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif  //! defined(DISABLE_SPARSE_TENSORS)
