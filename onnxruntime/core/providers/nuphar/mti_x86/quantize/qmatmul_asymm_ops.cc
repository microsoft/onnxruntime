// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/quantize/qmatmul_asymm_ops.h"

#include "core/providers/nuphar/nblas/nblas_igemv_avx2.h"
#include "core/providers/nuphar/nblas/nblas_igemv_mkl.h"

// TODO: refactor the headers
#include "core/codegen/common/common.h"
#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/reduce_ops.h"
#include "core/codegen/mti/math/unary_ops.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/codegen/mti/tensor/cast_ops.h"
#include "core/codegen/mti/tensor/concat_ops.h"
#include "core/codegen/mti/tensor/split.h"
#include "core/codegen/mti/tensor/transpose.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/target/generic/weight_layout/transpose_2d.h"
#include "core/common/cpuid_info.h"
#include "core/util/math_quantization.h"
#include <topi/detail/extern.h>
#include <topi/elemwise.h>

namespace onnxruntime {
namespace nuphar_codegen {

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.qmatmulasymmetric.mkl")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      DLTensor* transposed_quantized_param = args[0];
      DLTensor* Q_X = args[1];
      DLTensor* batch_tensor = args[2];
      DLTensor* Q_Y = args[3];
      int input_dim = args[4];
      int embed_dim = args[5];

      DCHECK(transposed_quantized_param->strides == nullptr);
      DCHECK(Q_X->strides == nullptr);
      DCHECK(Q_Y->strides == nullptr);

      auto quantized_param = reinterpret_cast<int8_t*>(static_cast<char*>(transposed_quantized_param->data) + transposed_quantized_param->byte_offset);
      auto quantized_X = reinterpret_cast<uint8_t*>(static_cast<char*>(Q_X->data) + Q_X->byte_offset);
      auto quantized_Y = reinterpret_cast<int32_t*>(static_cast<char*>(Q_Y->data) + Q_Y->byte_offset);
      auto batch = *reinterpret_cast<int*>(static_cast<char*>(batch_tensor->data) + batch_tensor->byte_offset);

      MKLIntGemvS8U8S32R(quantized_param, quantized_X, embed_dim, batch, input_dim, quantized_Y);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.qmatmulasymmetric.avx2")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      DLTensor* transposed_quantized_param = args[0];
      DLTensor* Q_X = args[1];
      DLTensor* batch_tensor = args[2];
      DLTensor* Q_Y = args[3];
      int input_dim = args[4];
      int embed_dim = args[5];

      DCHECK(transposed_quantized_param->strides == nullptr);
      DCHECK(Q_X->strides == nullptr);
      DCHECK(Q_Y->strides == nullptr);

      auto quantized_param = reinterpret_cast<int8_t*>(static_cast<char*>(transposed_quantized_param->data) + transposed_quantized_param->byte_offset);
      auto quantized_X = reinterpret_cast<uint8_t*>(static_cast<char*>(Q_X->data) + Q_X->byte_offset);
      auto quantized_Y = reinterpret_cast<int32_t*>(static_cast<char*>(Q_Y->data) + Q_Y->byte_offset);
      auto batch = *reinterpret_cast<int*>(static_cast<char*>(batch_tensor->data) + batch_tensor->byte_offset);

      if (batch == 1) {
        if (input_dim % 32 == 0)
          AVX2IntGemvS8U8S32R(quantized_param, quantized_X, input_dim, input_dim, embed_dim, quantized_Y);
        else
          AVX2IntGemvS8U8S32REx(quantized_param, quantized_X, input_dim, embed_dim, quantized_Y);
      } else {
        MKLIntGemvS8U8S32R(quantized_param, quantized_X, embed_dim, batch, input_dim, quantized_Y);
      }
    });

tvm::Array<tvm::Tensor>
QMatMulAsymmetricMKL(const tvm::Tensor& transposed_quantized_param,
                     const tvm::Tensor& Q_X,
                     const tvm::Expr& batch_seq_dim,
                     int input_dim,
                     int embed_dim,
                     const std::string& name) {
  std::string func_str;
#ifdef NUPHAR_USE_MKL
  func_str = "tvm.contrib.onnxruntime.qmatmulasymmetric.mkl";
#else
  ORT_NOT_IMPLEMENTED("Not implemented. Please set NUPHAR_USE_MKL!")
#endif

  return topi::detail::make_extern(
      {{batch_seq_dim, embed_dim}}, {tvm::Int(32)},
      {transposed_quantized_param, Q_X, tvm_codegen::Promote(batch_seq_dim, {16}, name + "_batch_asymm")},
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        return topi::detail::call_packed({tvm::Expr(func_str),
                                          topi::detail::pack_buffer(ins[0]),
                                          topi::detail::pack_buffer(ins[1]),
                                          topi::detail::pack_buffer(ins[2]),
                                          topi::detail::pack_buffer(outs[0]),
                                          input_dim,
                                          embed_dim});
      },
      name, "", {});
}

tvm::Array<tvm::Tensor>
QMatMulAsymmetricAVX2(const tvm::Tensor& transposed_quantized_param,
                      const tvm::Tensor& Q_X,
                      const tvm::Expr& batch_seq_dim,
                      int input_dim,
                      int embed_dim,
                      const std::string& name) {
#ifdef NUPHAR_USE_TENSORIZE
  // Tensorization support Gemv with input_dim aligned for now
  // TODO: extend to handle Gemm and unaligned cases
  const int64_t* batch_seq_dim_ptr = tvm::as_const_int(batch_seq_dim);
  if (batch_seq_dim_ptr != nullptr && *batch_seq_dim_ptr == 1  // check Gemv
      && input_dim % 32 == 0) {                                // check align
    tvm::Tensor X;
    int rank = gsl::narrow_cast<int>(Q_X->shape.size());
    ORT_ENFORCE(rank >= 2);
    auto* Q_X_dim0 = tvm::as_const_int(Q_X->shape[0]);
    auto* Q_X_dim1 = tvm::as_const_int(Q_X->shape[1]);
    if (rank == 2 && Q_X_dim0 && Q_X_dim1 &&
        *Q_X_dim0 == *batch_seq_dim_ptr &&
        gsl::narrow_cast<int>(*Q_X_dim1) == input_dim) {
      X = Q_X;
    } else {
      X = Reshape(Q_X, {batch_seq_dim, input_dim}, name + "_reshape_X");
    }
    auto Y = tvm::compute(
        {batch_seq_dim, embed_dim},
        [&](const tvm::Array<tvm::Var>& indices) {
          auto k = tvm::reduce_axis({0, input_dim});
          return tvm::sum(tvm::cast(HalideIR::Int(32), X(indices[0], k)) * tvm::cast(HalideIR::Int(32), transposed_quantized_param(indices[1], k)), {k});
        },
        name + "_8bit_tensorization");

    return {Y};
  }
#endif

  std::string func_str;
#ifdef NUPHAR_USE_AVX2
  func_str = "tvm.contrib.onnxruntime.qmatmulasymmetric.avx2";
#else
  ORT_NOT_IMPLEMENTED("Not implemented. Please set NUPHAR_USE_AVX2!");
#endif

  return topi::detail::make_extern(
      {{batch_seq_dim, embed_dim}}, {tvm::Int(32)},
      {transposed_quantized_param, Q_X, tvm_codegen::Promote(batch_seq_dim, {16}, name + "_batch_asymm")},
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        return topi::detail::call_packed({tvm::Expr(func_str),
                                          topi::detail::pack_buffer(ins[0]),
                                          topi::detail::pack_buffer(ins[1]),
                                          topi::detail::pack_buffer(ins[2]),
                                          topi::detail::pack_buffer(outs[0]),
                                          input_dim,
                                          embed_dim});
      },
      name, "", {});
}

}  // namespace nuphar_codegen
}  // namespace onnxruntime
