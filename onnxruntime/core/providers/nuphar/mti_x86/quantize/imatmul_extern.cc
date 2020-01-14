// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/quantize/imatmul_extern.h"

#include "core/common/common.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/providers/nuphar/extern/igemv_mkl.h"
#include "core/providers/nuphar/extern/igemv_avx2.h"
#include <topi/detail/extern.h>

namespace onnxruntime {
namespace nuphar {

#ifdef NUPHAR_USE_MKL
TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.imatmul.extern.mkl")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      DLTensor* A = args[0];
      DLTensor* B = args[1];
      DLTensor* batch_seq_tensor = args[2];
      DLTensor* Y = args[3];
      int input_dim = args[4];
      int embed_dim = args[5];

      DCHECK(A->strides == nullptr);
      DCHECK(B->strides == nullptr);
      DCHECK(Y->strides == nullptr);

      auto A_data = reinterpret_cast<uint8_t*>(static_cast<char*>(A->data) + A->byte_offset);
      auto B_data = reinterpret_cast<int8_t*>(static_cast<char*>(B->data) + B->byte_offset);
      auto Y_data = reinterpret_cast<int32_t*>(static_cast<char*>(Y->data) + Y->byte_offset);
      auto batch_seq = *reinterpret_cast<int*>(static_cast<char*>(batch_seq_tensor->data) + batch_seq_tensor->byte_offset);

      MKLIntGemvS8U8S32R(B_data, A_data, embed_dim, batch_seq, input_dim, Y_data);
    });
#endif

#ifdef NUPHAR_USE_AVX2
TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.imatmul.extern.avx2")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* /*ret*/) {
      DLTensor* A = args[0];
      DLTensor* B = args[1];
      DLTensor* batch_seq_tensor = args[2];
      DLTensor* Y = args[3];
      int input_dim = args[4];
      int embed_dim = args[5];

      DCHECK(B->strides == nullptr);
      DCHECK(A->strides == nullptr);
      DCHECK(Y->strides == nullptr);

      auto A_data = reinterpret_cast<uint8_t*>(static_cast<char*>(A->data) + A->byte_offset);
      auto B_data = reinterpret_cast<int8_t*>(static_cast<char*>(B->data) + B->byte_offset);
      auto Y_data = reinterpret_cast<int32_t*>(static_cast<char*>(Y->data) + Y->byte_offset);
      auto batch_seq = *reinterpret_cast<int*>(static_cast<char*>(batch_seq_tensor->data) + batch_seq_tensor->byte_offset);

      if (batch_seq == 1) {
        if (input_dim % 32 == 0)
          AVX2IntGemvS8U8S32R(B_data, A_data, input_dim, input_dim, embed_dim, Y_data);
        else
          AVX2IntGemvS8U8S32REx(B_data, A_data, input_dim, embed_dim, Y_data);
      } else {
#ifdef NUPHAR_USE_MKL
        MKLIntGemvS8U8S32R(B_data, A_data, embed_dim, batch_seq, input_dim, Y_data);
#else
        if (input_dim % 32 == 0) {
          for (int i = 0; i < batch_seq; i++)
            AVX2IntGemvS8U8S32R(B_data, A_data + i * input_dim, input_dim, input_dim, embed_dim, Y_data + i * embed_dim);
        } else {
          for (int i = 0; i < batch_seq; i++)
            AVX2IntGemvS8U8S32REx(B_data, A_data + i * input_dim, input_dim, embed_dim, Y_data + i * embed_dim);
        }
#endif
      }
    });
#endif

tvm::Tensor
IMatMulExternMKL(const tvm::Tensor& A,
                 const tvm::Tensor& B,
                 const tvm::Array<tvm::Expr>& output_shape,
                 int input_dim,
                 int embed_dim,
                 const std::string& name) {
  tvm::Expr batch_seq_dim = tvm_codegen::SizeToDimension(output_shape, -1);

  std::string func_str;
#ifdef NUPHAR_USE_MKL
  func_str = "tvm.contrib.onnxruntime.imatmul.extern.mkl";

  return topi::detail::make_extern(
      {output_shape}, {tvm::Int(32)},
      tvm_codegen::MakeInputsForExtern(
          {A, B, tvm_codegen::Promote(batch_seq_dim, {16}, name + "_batch_seq")}),
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        return topi::detail::call_packed({tvm::Expr(func_str),
                                          topi::detail::pack_buffer(ins[0]),
                                          topi::detail::pack_buffer(ins[1]),
                                          topi::detail::pack_buffer(ins[2]),
                                          topi::detail::pack_buffer(outs[0]),
                                          input_dim,
                                          embed_dim});
      },
      name, "", {})[0];
#else
  ORT_NOT_IMPLEMENTED("Not implemented. Please set NUPHAR_USE_MKL!");
#endif
}

tvm::Tensor
IMatMulExternAVX2(const tvm::Tensor& A,
                  const tvm::Tensor& B,
                  const tvm::Array<tvm::Expr>& output_shape,
                  int input_dim,
                  int embed_dim,
                  const std::string& name) {
  tvm::Expr batch_seq_dim = tvm_codegen::SizeToDimension(output_shape, -1);

  std::string func_str;
#ifdef NUPHAR_USE_AVX2
  func_str = "tvm.contrib.onnxruntime.imatmul.extern.avx2";

  return topi::detail::make_extern(
      {output_shape}, {tvm::Int(32)},
      tvm_codegen::MakeInputsForExtern(
          {A, B, tvm_codegen::Promote(batch_seq_dim, {16}, name + "_batch_seq")}),
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        return topi::detail::call_packed({tvm::Expr(func_str),
                                          topi::detail::pack_buffer(ins[0]),
                                          topi::detail::pack_buffer(ins[1]),
                                          topi::detail::pack_buffer(ins[2]),
                                          topi::detail::pack_buffer(outs[0]),
                                          input_dim,
                                          embed_dim});
      },
      name, "", {})[0];
#else
  ORT_NOT_IMPLEMENTED("Not implemented. Please set NUPHAR_USE_AVX2!");
#endif
}

}  // namespace nuphar
}  // namespace onnxruntime
