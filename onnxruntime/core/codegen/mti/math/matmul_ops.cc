// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/matmul_ops.h"

#include "core/codegen/mti/common.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor MatMul2D(const tvm::Tensor& A, const tvm::Tensor& B, bool trans_a, bool trans_b, const std::string& name) {
  return topi::matmul(A, B, trans_a, trans_b, name);
}

/*
 * Generic Matrix Multiplication
 *
 * If both arguments are 2-D, they are multiplied like conventional matrices.
 *
 * If either argument is N-D and N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
 *
 * If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
 * After matrix multiplication the prepended 1 is removed.
 *
 * If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions.
 * After matrix multiplication the appended 1 is removed.
 */
tvm::Tensor MatMul(const tvm::Tensor& A, const tvm::Tensor& B, const std::string& name) {
  int64_t a_rank = static_cast<int64_t>(A->shape.size());
  int64_t b_rank = static_cast<int64_t>(B->shape.size());
  const auto& A_shape = A->shape;
  const auto& B_shape = B->shape;
  if (a_rank == 2 && b_rank == 2) {
    // 2-D X 2-D
    return MatMul2D(A, B);
  } else if (a_rank == 1 && b_rank == 1) {
    // 1-D X 1-D
    auto k = tvm::reduce_axis(tvm::Range(0, A_shape[0]), "k");

    return tvm::compute(
        {},
        [&](const tvm::Array<tvm::Var>& /*indices*/) {
          return tvm::sum(A[k] * B[k], {k});
        },
        name);
  } else if (a_rank == 1) {
    // 1-D X n-D
    auto k = tvm::reduce_axis(tvm::Range(0, A_shape[0]), "k");

    auto l = [&](const tvm::Array<tvm::Var>& indices) {
      auto ndims = indices.size();
      MTI_ASSERT(ndims >= 1);
      tvm::Array<tvm::Expr> b_indices;
      for (size_t bi = 0; bi < ndims - 1; ++bi) {
        b_indices.push_back(indices[bi]);
      }
      b_indices.push_back(k);
      b_indices.push_back(indices[ndims - 1]);
      return tvm::sum(A({k}) * B(b_indices), {k});
    };
    return tvm::compute(ConcatShapes(SliceShapeToDimension(B_shape, -2), SliceShapeFromDimension(B_shape, -1)), l, name);
  } else if (b_rank == 1) {
    // n-D X 1-D
    auto k = tvm::reduce_axis(tvm::Range(0, B_shape[0]), "k");

    auto l = [&](const tvm::Array<tvm::Var>& indices) {
      tvm::Array<tvm::Expr> a_indices(indices.begin(), indices.end());
      a_indices.push_back(k);
      return tvm::sum(A(a_indices) * B({k}), {k});
    };
    return tvm::compute(SliceShapeToDimension(A->shape, -1), l, name);
  } else {
    // n-D X m-D
    MTI_ASSERT(a_rank >= 2 && b_rank >= 2);
    auto k = tvm::reduce_axis(tvm::Range(0, A_shape[a_rank - 1]), "k");

    auto l = [&](const tvm::Array<tvm::Var>& indices) {
      auto ndims = static_cast<int>(indices.size());
      MTI_ASSERT(ndims > 2);
      tvm::Array<tvm::Expr> a_indices, b_indices;

      // handle broadcasting
      int i = 0, a_idx = 0, b_idx = 0;
      bool a_greater = a_rank > b_rank;
      for (; i < std::abs(a_rank - b_rank); ++i) {
        if (a_greater) {
          a_indices.push_back(indices[i]);
          a_idx++;
        } else {
          b_indices.push_back(indices[i]);
          b_idx++;
        }
      }
      for (; i < ndims - 2; ++i, ++a_idx, ++b_idx) {
        auto tp = indices[i].type();
        if (IsOne(A_shape, a_idx)) {
          a_indices.push_back(tvm::make_zero(tp));
          b_indices.push_back(indices[i]);
        } else if (IsOne(B_shape, b_idx)) {
          b_indices.push_back(tvm::make_zero(tp));
          a_indices.push_back(indices[i]);
        } else {
          a_indices.push_back(indices[i]);
          b_indices.push_back(indices[i]);
        }
      }

      MTI_ASSERT(a_idx == a_rank - 2 && b_idx == b_rank - 2);
      a_indices.push_back(indices[ndims - 2]);
      a_indices.push_back(k);

      b_indices.push_back(k);
      b_indices.push_back(indices[ndims - 1]);

      return tvm::sum(A(a_indices) * B(b_indices), {k});
    };

    return tvm::compute(ComputeMatMulShape(A_shape, B_shape), l, name);
  }
}

tvm::Array<tvm::Expr>
ComputeMatMulShape(
    const tvm::Array<tvm::Expr>& A_shape,
    const tvm::Array<tvm::Expr>& B_shape,
    bool trans_a,
    bool trans_b) {
  auto a_rank = A_shape.size();
  auto b_rank = B_shape.size();
  tvm::Array<tvm::Expr> output_shape;
  int64_t output_rank = std::max(a_rank, b_rank);
  MTI_ASSERT(a_rank > 0 && b_rank > 0);
  if (a_rank == 1 && b_rank == 1) {
    MTI_ASSERT(!trans_a && !trans_b);
    // reduction, output shape is empty
  } else if (a_rank == 1) {
    MTI_ASSERT(!trans_a && !trans_b);
    output_shape = SliceShapeToDimension(B_shape, b_rank - 2);
    output_shape.push_back(B_shape[b_rank - 1]);
  } else if (b_rank == 1) {
    MTI_ASSERT(!trans_a && !trans_b);
    output_shape = SliceShapeToDimension(A_shape, a_rank - 1);
  } else {
    for (int64_t i = 0; i < output_rank - 2; i++) {
      tvm::Expr broadcasted_dim = tvm::make_const(HalideIR::Int(32), 1);
      bool broadcasted =
          BroadcastDim(A_shape, i, output_rank, broadcasted_dim) &&
          BroadcastDim(B_shape, i, output_rank, broadcasted_dim);
      MTI_ASSERT(broadcasted);
      output_shape.push_back(broadcasted_dim);
    }
    output_shape.push_back(A_shape[a_rank - (trans_a ? 1 : 2)]);
    output_shape.push_back(B_shape[b_rank - (trans_b ? 2 : 1)]);
  }
  return output_shape;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
