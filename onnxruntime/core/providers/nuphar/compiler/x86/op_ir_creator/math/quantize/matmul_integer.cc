// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/cast_ops.h"
#include "core/codegen/mti/tensor/pad_ops.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/codegen/mti/tensor/transpose.h"
#include "core/codegen/passes/weight_layout/tiling_2d.h"
#include "core/codegen/passes/weight_layout/transpose_2d.h"
#include "core/common/cpuid_info.h"  // TODO: refactor to control through config
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/providers/nuphar/mti_x86/quantize/imatmul_extern.h"
#include "core/providers/nuphar/mti_x86/quantize/imatmul16_extern.h"

namespace onnxruntime {
namespace nuphar {

tvm::Tensor IMatMulTensorize(const tvm::Tensor& A,
                             const tvm::Tensor& B,
                             const tvm::Expr& batchseq_dim,
                             int input_dim,
                             int embed_dim,
                             int vector_width,
                             const std::string& name) {
  tvm::Tensor A_reshape;
  int A_rank = gsl::narrow_cast<int>(A->shape.size());
  const int64_t* A_dim0 = tvm::as_const_int(A->shape[0]);
  const int64_t* A_dim1 = tvm::as_const_int(A->shape[1]);
  ORT_ENFORCE(A_rank >= 2);
  const int64_t* B_dim0 = tvm::as_const_int(B->shape[0]);
  const int64_t* B_dim1 = tvm::as_const_int(B->shape[1]);
  ORT_ENFORCE(B_dim1 != nullptr && B_dim0 != nullptr);
  int input_padded = gsl::narrow_cast<int>(*B_dim1);
  int embed_padded = gsl::narrow_cast<int>(*B_dim0);
  const int64_t* p_batchseq_dim = tvm::as_const_int(batchseq_dim);
  if (p_batchseq_dim != nullptr && A_rank == 2 &&
      A_dim0 != nullptr && *A_dim0 == *p_batchseq_dim &&
      A_dim1 != nullptr && gsl::narrow_cast<int>(*A_dim1) == input_dim) {
    A_reshape = A;

  } else {
    A_reshape = tvm_codegen::Reshape(A, {batchseq_dim, input_dim}, name + "_reshape_X");
    if (input_dim != input_padded) {
      tvm::Expr pad_value = tvm::make_const(A->dtype, 0);
      A_reshape = tvm_codegen::PadLastDim(A_reshape, vector_width, pad_value);
    }
  }
  tvm::Tensor Y = tvm::compute(
      {batchseq_dim, embed_padded},
      [&](const tvm::Array<tvm::Var>& indices) {
        auto k = tvm::reduce_axis({0, input_padded});
        return tvm::sum(tvm::cast(HalideIR::Int(32), A_reshape(indices[0], k)) * tvm::cast(HalideIR::Int(32), B(indices[1], k)), {k});
      },
      name + "_tensorize_Y");
  return Y;
}

// Evaluate of MatMulInteger or MatMulInteger16
static Status EvaluateMatMulInteger(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  const auto& A = inputs[0];
  const auto& B = inputs[1];
  auto& name = node.Name();

  if (B->shape.size() == 2) {
    const int64_t* p_input_dim = tvm::as_const_int(B->shape[0]);
    const int64_t* p_embed_dim = tvm::as_const_int(B->shape[1]);

    if (p_input_dim != nullptr && p_embed_dim != nullptr) {
      int64_t input_dim = *p_input_dim;
      int64_t embed_dim = *p_embed_dim;

      bool is8bit = (A->dtype == HalideIR::type_of<uint8_t>() &&
                     B->dtype == HalideIR::type_of<int8_t>());
      bool is16bit = (A->dtype == HalideIR::type_of<int16_t>() &&
                      B->dtype == HalideIR::type_of<int16_t>());

      if (is16bit || is8bit) {
        // batchseq_dim: batch * seq
        auto A_rank = gsl::narrow_cast<int>(A->shape.size());
        auto batchseq_dim = tvm_codegen::SizeToDimension(A->shape, A_rank - 1);

        const int64_t* p_batch_seq_dim = tvm::as_const_int(batchseq_dim);
        bool isGEMV = (p_batch_seq_dim != nullptr && *p_batch_seq_dim == 1);

        // tvm has known issue when handling tensorization of matmul: [1x1] = [1xK]x[Kx1]
        // and this case is not likely happen in real model
        // so add option to fall back to a general reduction
        bool is_scalar = isGEMV && (embed_dim == 1);

        tvm::Array<tvm::Expr> output_shape;
        for (int i = 0; i < A_rank - 1; ++i) {
          output_shape.push_back(A->shape[i]);
        }
        output_shape.push_back(tvm::Expr(gsl::narrow_cast<int>(embed_dim)));

        tvm::Tensor B_marshalled;
        auto B_NodeArg = node.InputDefs()[1];
        const std::string& B_name = B_NodeArg->Name();

        // vector width determined from target hardware
        // AVX2:   vector width 32 = 256bits / 8bit; 16 = 256 bits / 16bits;
        // AVX512: vector width 64 = 512bits / 8bit; 32 = 512 bits / 16bits;
        int vector_width = 32;
        CodeGenTargetX86* target = dynamic_cast<CodeGenTargetX86*>(ctx_codegen.GetCodeGenHandle()->codegen_target);
        if (target != nullptr) {
          vector_width = target->NaturalVectorWidth(B->dtype.bits()) / 2;
        }
        const codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
        bool force_mkl = false;
        if (settings.HasOption(kNupharIMatMulForceMkl)) {
          force_mkl = true;
        }
        // Tensorization: AVX2: 8bit GEMM
        //                AVX512: 8bit GEMV and GEMM
        bool use_tensorization = !force_mkl && is8bit &&
                                 ((CPUIDInfo::GetCPUIDInfo().HasAVX2() && !isGEMV) ||
                                  CPUIDInfo::GetCPUIDInfo().HasAVX512Skylake());
        if (settings.HasOption(kNupharForceNoTensorize)) {
          if (settings.OptionMatches(kNupharForceNoTensorize, kNupharTensorize_Int8Gemm)) {
            use_tensorization = false;
          }
        }
        // Apply layout for tensorization
        // TODO: when build decision tree, add check for the weight shape if worth doing layout packing
        //       if size is too small then not worth, e.g. embed_dim < 1/2 * layout_dim_y = 8 (avx2) / 16 (avx512)
        if (ctx_nuphar->IsInitializer(B_name)) {
          auto layout_key = (use_tensorization && (!is_scalar)) ? tvm_codegen::WeightLayoutTiling2D::GetKey(TensorProtoDataType(B_NodeArg), vector_width)
                                                                : tvm_codegen::WeightLayoutTranspose2D::GetKey(TensorProtoDataType(B_NodeArg));
          B_marshalled = ctx_nuphar->ApplyWeightLayout(layout_key, B_name, B, true);
        } else {
          B_marshalled = tvm_codegen::Transpose(B, {1, 0});
        }

        // TODO: add reserved_bits attribute
        tvm::Tensor output_tensor;
        if (is8bit) {
          if (!force_mkl && CPUIDInfo::GetCPUIDInfo().HasAVX512Skylake()) {
            output_tensor = use_tensorization ? IMatMulTensorize(A, B_marshalled, batchseq_dim, input_dim, embed_dim, vector_width, name + "_IMatMulTensorizeAVX512")
                                              : IMatMulExternMKL(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMulExternMKL");
          } else if (!force_mkl && CPUIDInfo::GetCPUIDInfo().HasAVX2()) {
            // TODO: remove the branch after test the tensorize AVX2 GEMV performance
            output_tensor = (use_tensorization && (!is_scalar)) ? IMatMulTensorize(A, B_marshalled, batchseq_dim, input_dim, embed_dim, vector_width, name + "_IMatMulTensorizeAVX2")
                                                                : IMatMulExternAVX2(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMulExternAVX2");
          } else {
            output_tensor = IMatMulExternMKL(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMulExternMKL");
          }
        } else if (is16bit) {
          if (!force_mkl && CPUIDInfo::GetCPUIDInfo().HasAVX2()) {
            output_tensor = IMatMul16ExternAVX2(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMul16ExternAVX2");
          } else {
            output_tensor = IMatMul16ExternMKL(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMul16ExternMKL");
          }
        }

        if (use_tensorization) {
          tvm::Array<tvm::Expr> Y_shape;
          for (int i = 0; i < A_rank - 1; ++i) {
            Y_shape.push_back(A->shape[i]);
          }

          tvm::Expr embed_padded = B_marshalled->shape[0];
          Y_shape.push_back(embed_padded);
          tvm::Tensor Y = tvm_codegen::Reshape(output_tensor, Y_shape, name + "_reshape");

          const int64_t* p_embed_padded = tvm::as_const_int(embed_padded);
          ORT_ENFORCE(p_embed_padded != nullptr);
          if (*p_embed_padded != embed_dim) {
            tvm::Tensor Y_unpad = tvm::compute(output_shape,
                                               [&](const tvm::Array<tvm::Var>& indices) {
                                                 return Y(indices);
                                               },
                                               name + "_unpad_shape");
            outputs.push_back(Y_unpad);
          } else {
            outputs.push_back(Y);
          }
        } else {
          outputs.push_back(output_tensor);
        }

        return Status::OK();
      }
    }
  }
  // A generic path, cast to int32
  // Support skipped trailing inputs
  auto A_Int32 = (node.InputDefs().size() >= 3 && node.InputDefs()[2]->Exists())
                     ? tvm_codegen::Sub(tvm_codegen::Cast(A, HalideIR::Int(32)), tvm_codegen::Cast(inputs[2], HalideIR::Int(32)))
                     : tvm_codegen::Cast(A, HalideIR::Int(32));
  auto B_Int32 = (node.InputDefs().size() >= 4 && node.InputDefs()[3]->Exists())
                     ? tvm_codegen::Sub(tvm_codegen::Cast(B, HalideIR::Int(32)), tvm_codegen::Cast(inputs[3], HalideIR::Int(32)))
                     : tvm_codegen::Cast(B, HalideIR::Int(32));
  tvm::Tensor Y = tvm_codegen::MatMul(A_Int32, B_Int32, name);
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of MatMulInteger OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(MatMulInteger)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  return EvaluateMatMulInteger(inputs, node, ctx_codegen, outputs);
}

// Evaluate of MatMulInteger16 OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(MatMulInteger16)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  return EvaluateMatMulInteger(inputs, node, ctx_codegen, outputs);
}

}  // namespace nuphar
}  // namespace onnxruntime
