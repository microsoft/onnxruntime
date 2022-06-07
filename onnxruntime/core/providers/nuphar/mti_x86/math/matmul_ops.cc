// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/math/matmul_ops.h"

#include "core/codegen/common/profile.h"
#include "core/codegen/mti/math/matmul_ops.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include <topi/detail/extern.h>
#include <topi/transform.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor MatMul2D(const tvm::Tensor& A, const tvm::Tensor& B, bool trans_a, bool trans_b, const std::string& name) {
  tvm::Tensor Y;
  if (GemmExternCpu(A, B, Y, trans_a, trans_b))
    return Y;

  return topi::matmul(A, B, trans_a, trans_b, name);
}

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.sgemm_cpu")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      CODEGEN_PROFILER_EVENT("math_sgemm");
      // Explicitly construct TVMArgValue instead of calling operator[] on args for saving some cycles.
      DLTensor* A = tvm::runtime::TVMArgValue(args.values[0], args.type_codes[0]);
      DLTensor* B = tvm::runtime::TVMArgValue(args.values[1], args.type_codes[1]);
      DLTensor* C = tvm::runtime::TVMArgValue(args.values[2], args.type_codes[2]);
      bool trans_a = tvm::runtime::TVMArgValue(args.values[3], args.type_codes[3]);
      bool trans_b = tvm::runtime::TVMArgValue(args.values[4], args.type_codes[4]);
      float alpha = 1.0f;
      float beta = 0.0f;

      DCHECK(C->strides == nullptr);
      DCHECK(B->strides == nullptr);
      DCHECK(A->strides == nullptr);
      DCHECK(tvm::runtime::TypeMatch(A->dtype, kDLFloat, 32));
      DCHECK(tvm::runtime::TypeMatch(B->dtype, kDLFloat, 32));
      DCHECK(tvm::runtime::TypeMatch(C->dtype, kDLFloat, 32));

      int64_t M, N, K;

      // compute default M by flatten A dims
      M = 1;
      for (int d = 0; d < A->ndim - 1; ++d)
        M *= A->shape[d];

      if (A->ndim == 1) {
        DCHECK(!trans_a);
        DCHECK_GT(B->ndim, 1);
        M = 1;
        N = B->shape[trans_b ? 0 : B->ndim - 1];
        K = A->shape[0];
      } else if (B->ndim == 1) {
        // N-D x 1-D
        DCHECK(!trans_a);
        DCHECK(!trans_b);
        DCHECK_GT(A->ndim, 1);
        N = 1;
        K = A->shape[A->ndim - 1];
      } else {
        // N-D x N-D
        DCHECK(!trans_a || A->ndim == 2);  // only allow trans_a for 2D
        if (trans_a) {
          M = A->shape[1];
          K = A->shape[0];
        } else {
          K = A->shape[A->ndim - 1];
        }

        // B is essentially 2D, allowing >2D here to reduce flatten at extern input
        N = B->shape[trans_b ? B->ndim - 2 : B->ndim - 1];
      }

      // for empty tensor, don't do anything
      if (M == 0 || N == 0 || K == 0)
        return;

      math::Gemm<float, concurrency::ThreadPool>(
          trans_a ? CblasTrans : CblasNoTrans,
          trans_b ? CblasTrans : CblasNoTrans,
          M,
          N,
          K,
          alpha,
          reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset),
          reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset),
          beta,
          reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset),
          nullptr);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.batched_matmul_cpu")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      CODEGEN_PROFILER_EVENT("math_batched_sgemm");
      DLTensor* A = tvm::runtime::TVMArgValue(args.values[0], args.type_codes[0]);
      DLTensor* B = tvm::runtime::TVMArgValue(args.values[1], args.type_codes[1]);
      DLTensor* C = tvm::runtime::TVMArgValue(args.values[2], args.type_codes[2]);

      DCHECK(C->strides == nullptr);
      DCHECK(B->strides == nullptr);
      DCHECK(A->strides == nullptr);
      DCHECK(tvm::runtime::TypeMatch(A->dtype, kDLFloat, 32));
      DCHECK(tvm::runtime::TypeMatch(B->dtype, kDLFloat, 32));
      DCHECK(tvm::runtime::TypeMatch(C->dtype, kDLFloat, 32));

      if (args.num_args == 3) {
        MatMulComputeHelper helper;
        TensorShape A_shape(A->shape, A->ndim);
        TensorShape B_shape(B->shape, B->ndim);
        helper.Compute(A_shape, B_shape);

        size_t max_len = helper.OutputOffsets().size();
        for (size_t i = 0; i < max_len; i++) {
          math::MatMul<float>(
              static_cast<int>(helper.M()),
              static_cast<int>(helper.N()),
              static_cast<int>(helper.K()),
              (float*)A->data + helper.LeftOffsets()[i],
              (float*)B->data + helper.RightOffsets()[i],
              (float*)C->data + helper.OutputOffsets()[i],
              nullptr);  // TODO: use thread pool from OpContext
        }
      } else {
        // matmul fused with transpose, modify lda/ldb and step_a/step_b for the zero-cost transpose
        DCHECK(A->ndim == B->ndim);
        DCHECK(args.num_args - 3 == A->ndim + B->ndim);
        std::vector<int32_t> permute_A(A->ndim);
        std::vector<int64_t> stride_A(A->ndim);
        std::vector<int32_t> permute_B(B->ndim);
        std::vector<int64_t> stride_B(B->ndim);
        int arg_idx = 3;
        int num_matmuls = 1;
        for (int i = 0; i < A->ndim; ++i) {
          permute_A[i] = tvm::runtime::TVMArgValue(args.values[arg_idx + i], args.type_codes[arg_idx + i]);
          if (i < A->ndim - 2) {
            num_matmuls *= A->shape[permute_A[i]];
          }
          stride_A[A->ndim - 1 - i] = (i == 0) ? 1 : stride_A[A->ndim - i] * A->shape[A->ndim - i];
        }
        arg_idx += A->ndim;
        for (int i = 0; i < B->ndim; ++i) {
          permute_B[i] = tvm::runtime::TVMArgValue(args.values[arg_idx + i], args.type_codes[arg_idx + i]);
          stride_B[B->ndim - 1 - i] = (i == 0) ? 1 : stride_B[B->ndim - i] * B->shape[B->ndim - i];
        }

        float alpha = 1.0f;
        float beta = 0.0f;
        int64_t M = A->shape[permute_A[A->ndim - 2]];
        int64_t K = A->shape[permute_A[A->ndim - 1]];
        int64_t N = B->shape[permute_B[B->ndim - 1]];
        bool trans_a = (permute_A[A->ndim - 2] == A->ndim - 1);
        bool trans_b = (permute_B[B->ndim - 2] == B->ndim - 1);
        int64_t step_a = num_matmuls > 1 ? stride_A[permute_A[A->ndim - 3]] : 0;
        int64_t lda = stride_A[permute_A[A->ndim - (trans_a ? 1 : 2)]];
        int64_t step_b = num_matmuls > 1 ? stride_B[permute_B[B->ndim - 3]] : 0;
        int64_t ldb = stride_B[permute_B[B->ndim - (trans_b ? 1 : 2)]];

        for (int i = 0; i < num_matmuls; i++) {
          math::GemmEx<float, concurrency::ThreadPool>(
              trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans,
              M,
              N,
              K,
              alpha,
              (float*)A->data + i * step_a,
              lda,
              (float*)B->data + i * step_b,
              ldb,
              beta,
              (float*)C->data + i * M * N,
              N,
              nullptr);  // TODO: use thread pool from OpContext
        }
      }
    });

static bool ShouldUseMatMulExtern() {
  // Note: currently default behavior is always prefer extern
  const codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  if (settings.HasOption(kNupharMatmulExec)) {
    bool prefer_extern = settings.OptionMatches(
        kNupharMatmulExec,
        kNupharMatMulExec_ExternCpu);
    if (!prefer_extern)
      return false;
  }
  return true;
}

bool CanPermuteBeFusedInMatMul(const std::vector<int32_t>& perm) {
  auto rank = gsl::narrow<int32_t>(perm.size());
  if (rank < 2) return true;

  // only fusable if inner-most dim could be transposed
  return (perm[rank - 1] == rank - 1) ||
         (perm[rank - 2] == rank - 1);
};

bool GemmExternCpu(
    const tvm::Tensor& A,
    const tvm::Tensor& B,
    tvm::Tensor& Y,
    bool trans_a,
    bool trans_b,
    const std::string& name) {
  if (!ShouldUseMatMulExtern())
    return false;

  if (A->shape.size() == 1 && B->shape.size() == 1)
    return false;  // TVM extern cannot have output shape being empty

  // TODO: add support for mixed precisions
  if (A->dtype != B->dtype ||
      !A->dtype.is_float() ||
      A->dtype.bits() != 32)
    return false;

  tvm::Array<tvm::Expr> out_shape = tvm_codegen::ComputeMatMulShape(A->shape, B->shape, trans_a, trans_b);

  Y = topi::detail::make_extern(
      {out_shape}, {A->dtype}, {A, B},
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        return topi::detail::call_packed(
            {tvm::Expr("tvm.contrib.onnxruntime.sgemm_cpu"),
             topi::detail::pack_buffer(ins[0]),
             topi::detail::pack_buffer(ins[1]),
             topi::detail::pack_buffer(outs[0]),
             trans_a,
             trans_b});
      },
      name + "_sgemm_cpu", "", {})[0];

  return true;
}

bool MatMulExternCpu(
    const tvm::Tensor& A,
    const tvm::Tensor& B,
    tvm::Tensor& Y,
    const std::vector<int32_t>* permute_A,
    const std::vector<int32_t>* permute_B,
    const std::string& name) {
  if (permute_A != nullptr) {
    ORT_ENFORCE(permute_B != nullptr);
    ORT_ENFORCE(CanPermuteBeFusedInMatMul(*permute_A));
    ORT_ENFORCE(CanPermuteBeFusedInMatMul(*permute_B));
    ORT_ENFORCE(permute_A->size() == permute_B->size());
    ORT_ENFORCE(permute_A->size() == A->shape.size());
    ORT_ENFORCE(permute_B->size() == B->shape.size());
  }

  // TODO: add support for mixed precisions
  if (A->dtype != B->dtype ||
      !A->dtype.is_float() ||
      A->dtype.bits() != 32)
    return false;

  // inputs need to be at least 1D
  auto rank_A = gsl::narrow<int32_t>(A->shape.size());
  auto rank_B = gsl::narrow<int32_t>(B->shape.size());

  if (rank_A < 1 || rank_B < 1)
    return false;

  // do not support 1-D x 1-D as tvm extern require buffer size > 0
  if (rank_A == 1 && rank_B == 1)
    return false;

  tvm::Array<tvm::Expr> matmul_A_shape, matmul_B_shape;
  for (int32_t d = 0; d < rank_A; ++d) {
    matmul_A_shape.push_back(A->shape[permute_A != nullptr ? permute_A->at(d) : d]);
  }
  for (int32_t d = 0; d < rank_B; ++d) {
    matmul_B_shape.push_back(B->shape[permute_B != nullptr ? permute_B->at(d) : d]);
  }

  tvm::Array<tvm::Expr> out_shape;
  out_shape = tvm_codegen::ComputeMatMulShape(matmul_A_shape, matmul_B_shape);

  Y = topi::detail::make_extern(
      {out_shape}, {A->dtype}, {A, B},
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        tvm::Array<tvm::Expr> extern_args = {
            tvm::Expr("tvm.contrib.onnxruntime.batched_matmul_cpu"),
            topi::detail::pack_buffer(ins[0]),
            topi::detail::pack_buffer(ins[1]),
            topi::detail::pack_buffer(outs[0])};
        if (permute_A != nullptr && permute_B != nullptr) {
          for (const auto& perm_A : *permute_A) {
            extern_args.push_back(perm_A);
          }
          for (const auto& perm_B : *permute_B) {
            extern_args.push_back(perm_B);
          }
        }
        return topi::detail::call_packed(extern_args);
      },
      name + "_batched_matmul_cpu", "", {})[0];

  return true;
}

tvm::Tensor MatMul(const tvm::Tensor& A, const tvm::Tensor& B, const std::string& name) {
  tvm::Tensor Y;
  if (GemmExternCpu(A, B, Y))
    return Y;
  // go through generic case otherwise
  return tvm_codegen::MatMul(A, B, name);
}

}  // namespace nuphar
}  // namespace onnxruntime
