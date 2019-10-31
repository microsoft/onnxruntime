// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/common/utils.h"
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
  ORT_ENFORCE(A_rank >= 2);

  const int64_t* A_dim0 = tvm::as_const_int(A->shape[0]);
  const int64_t* A_dim1 = tvm::as_const_int(A->shape[1]);
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
  }

  if (input_dim != input_padded) {
    tvm::Expr pad_value = tvm::make_const(A->dtype, 0);
    A_reshape = tvm_codegen::PadLastDim(A_reshape, vector_width, pad_value);
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

// A generic path, cast to int32
// Support skipped trailing inputs
tvm::Tensor GenericMatMulInteger(const tvm::Array<tvm::Tensor>& inputs, const Node& node) {
  auto A_Int32 = (node.InputDefs().size() >= 3 && node.InputDefs()[2]->Exists())
                     ? tvm_codegen::Sub(tvm_codegen::Cast(inputs[0], HalideIR::Int(32)), tvm_codegen::Cast(inputs[2], HalideIR::Int(32)))
                     : tvm_codegen::Cast(inputs[0], HalideIR::Int(32));
  auto B_Int32 = (node.InputDefs().size() >= 4 && node.InputDefs()[3]->Exists())
                     ? tvm_codegen::Sub(tvm_codegen::Cast(inputs[1], HalideIR::Int(32)), tvm_codegen::Cast(inputs[3], HalideIR::Int(32)))
                     : tvm_codegen::Cast(inputs[1], HalideIR::Int(32));
  return tvm_codegen::MatMul(A_Int32, B_Int32, node.Name() + "_Generic");
}

// Evaluate of MatMulInteger
static Status EvaluateMatMulInteger(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  const auto& A = inputs[0];
  const auto& B = inputs[1];
  auto& name = node.Name();

  bool is8bit = (A->dtype == HalideIR::type_of<uint8_t>() &&
                 B->dtype == HalideIR::type_of<int8_t>());

  if (B->shape.size() == 2 && is8bit) {
    const int64_t* p_input_dim = tvm::as_const_int(B->shape[0]);
    const int64_t* p_embed_dim = tvm::as_const_int(B->shape[1]);

    if (p_input_dim != nullptr && p_embed_dim != nullptr) {
      int64_t input_dim = *p_input_dim;
      int64_t embed_dim = *p_embed_dim;

      // batchseq_dim: batch * seq
      auto A_rank = gsl::narrow_cast<int>(A->shape.size());
      auto batchseq_dim = tvm_codegen::SizeToDimension(A->shape, A_rank - 1);
      const int64_t* p_batch_seq_dim = tvm::as_const_int(batchseq_dim);

      tvm::Array<tvm::Expr> output_shape;
      for (int i = 0; i < A_rank - 1; ++i) {
        output_shape.push_back(A->shape[i]);
      }
      output_shape.push_back(tvm::Expr(gsl::narrow_cast<int>(embed_dim)));

      // Enviornment variables option
      const codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
      TargetFeature feature = GetTargetInfo(settings);

      bool force_mkl = false;
      if (settings.HasOption(kNupharIMatMulForceMkl)) {
        force_mkl = true;
      }
      bool force_no_tensorize = false;
      if (settings.HasOption(kNupharForceNoTensorize)) {
        force_no_tensorize = true;
      }

      // Tensorization: AVX2: 8bit GEMM AVX512: 8bit GEMV and GEMM
      bool isGEMV = (p_batch_seq_dim != nullptr && *p_batch_seq_dim == 1);
      bool use_tensorization = !force_mkl && !force_no_tensorize && (feature.hasAVX512 || (feature.hasAVX2 && !isGEMV) || (!feature.hasAVX2 && feature.hasAVX));

      // Model input option
      auto B_NodeArg = node.InputDefs()[1];
      const std::string& B_name = B_NodeArg->Name();
      bool hasInitializer = ctx_nuphar->IsInitializer(B_name);

      if (!hasInitializer) {
        //TODO: change to use MLAS when no layout could apply
        tvm::Tensor B_marshalled = tvm_codegen::Transpose(B, {1, 0});

        bool use_extern_MKL = (force_mkl || !feature.hasAVX2);
        tvm::Tensor output_tensor = use_extern_MKL ? IMatMulExternMKL(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMulExternMKL")
                                                   : IMatMulExternAVX2(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMulExternAVX2");

        outputs.push_back(output_tensor);
      } else if (use_tensorization) {
        // vector width determined from target hardware
        // AVX:    vector width 16 = 128 bits / 8 bits;
        // AVX2:   vector width 32 = 256 bits / 8 bits;
        // AVX512: vector width 64 = 512 bits / 8 bits;
        CodeGenTargetX86* target = dynamic_cast<CodeGenTargetX86*>(ctx_codegen.GetCodeGenHandle()->codegen_target);
        ORT_ENFORCE(target != nullptr, "CodeGen target unknown: not AVX/AVX2/AVX512 !");
        int vector_width = target->NaturalVectorWidth(B->dtype.bits()) / 2;

        // TVM has known issue when handling tensorization of matmul: [1x1] = [1xK]x[Kx1]
        // and this case is not likely happen in real model
        // so add option to fall back to a general reduction
        bool isScalar = isGEMV && (embed_dim == 1);

        // Tensorization has two layout options: 1) Transpose or 2) Tiling
        auto layout_key = !isScalar ? tvm_codegen::WeightLayoutTiling2D::GetKey(TensorProtoDataType(B_NodeArg), vector_width)
                                    : tvm_codegen::WeightLayoutTranspose2D::GetKey(TensorProtoDataType(B_NodeArg));
        tvm::Tensor B_marshalled = ctx_nuphar->ApplyWeightLayout(layout_key, B_name, B, true);

        tvm::Tensor output_tensor = IMatMulTensorize(A, B_marshalled, batchseq_dim, input_dim, embed_dim, vector_width, name + "_IMatMulTensorizeAVX512");

        // Post processing output tensor
        tvm::Expr embed_padded = B_marshalled->shape[0];
        const int64_t* p_embed_padded = tvm::as_const_int(embed_padded);
        ORT_ENFORCE(p_embed_padded != nullptr);

        tvm::Array<tvm::Expr> Y_shape;
        for (int i = 0; i < A_rank - 1; ++i) {
          Y_shape.push_back(A->shape[i]);
        }
        Y_shape.push_back(embed_padded);

        tvm::Tensor Y = tvm_codegen::Reshape(output_tensor, Y_shape, name + "_reshape");

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
        auto layout_key = tvm_codegen::WeightLayoutTranspose2D::GetKey(TensorProtoDataType(B_NodeArg));
        tvm::Tensor B_marshalled = ctx_nuphar->ApplyWeightLayout(layout_key, B_name, B, true);

        bool use_extern_AVX2 = (!force_mkl && feature.hasAVX2);
        tvm::Tensor output_tensor = use_extern_AVX2 ? IMatMulExternAVX2(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMulExternAVX2")
                                                    : IMatMulExternMKL(A, B_marshalled, output_shape, input_dim, embed_dim, name + "_IMatMulExternMKL");

        outputs.push_back(output_tensor);
      }

      return Status::OK();
    }
  }

  tvm::Tensor Y = GenericMatMulInteger(inputs, node);
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of MatMulInteger16
static Status EvaluateMatMulInteger16(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  const auto& A = inputs[0];
  const auto& B = inputs[1];

  bool is16bit = (A->dtype == HalideIR::type_of<int16_t>() &&
                  B->dtype == HalideIR::type_of<int16_t>());

  const codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  TargetFeature feature = GetTargetInfo(settings);

  // 16bit on AVX fall back to 32bit
  bool AVXonly = feature.hasAVX && !feature.hasAVX2;

  if (!AVXonly && (B->shape.size() == 2 && is16bit)) {
    const int64_t* p_input_dim = tvm::as_const_int(B->shape[0]);
    const int64_t* p_embed_dim = tvm::as_const_int(B->shape[1]);

    if (p_input_dim != nullptr && p_embed_dim != nullptr) {
      int64_t input_dim = *p_input_dim;
      int64_t embed_dim = *p_embed_dim;

      auto A_rank = gsl::narrow_cast<int>(A->shape.size());
      tvm::Array<tvm::Expr> output_shape;
      for (int i = 0; i < A_rank - 1; ++i) {
        output_shape.push_back(A->shape[i]);
      }
      output_shape.push_back(tvm::Expr(gsl::narrow_cast<int>(embed_dim)));

      bool force_mkl = false;
      if (settings.HasOption(kNupharIMatMulForceMkl)) {
        force_mkl = true;
      }

      // Model input option
      auto B_NodeArg = node.InputDefs()[1];
      const std::string& B_name = B_NodeArg->Name();
      bool hasInitializer = ctx_nuphar->IsInitializer(B_name);

      if (!hasInitializer) {
        //TODO: change to use MLAS when no layout could apply
        tvm::Tensor B_marshalled = tvm_codegen::Transpose(B, {1, 0});

        bool use_extern_MKL = (force_mkl || !feature.hasAVX2);
        tvm::Tensor output_tensor = use_extern_MKL ? IMatMul16ExternMKL(A, B_marshalled, output_shape, input_dim, embed_dim, node.Name() + "_IMatMulExternMKL")
                                                   : IMatMul16ExternAVX2(A, B_marshalled, output_shape, input_dim, embed_dim, node.Name() + "_IMatMulExternAVX2");
        outputs.push_back(output_tensor);
      } else {
        auto layout_key = tvm_codegen::WeightLayoutTranspose2D::GetKey(TensorProtoDataType(B_NodeArg));
        tvm::Tensor B_marshalled = ctx_nuphar->ApplyWeightLayout(layout_key, B_name, B, true);

        bool use_extern_AVX2 = (!force_mkl && feature.hasAVX2);
        tvm::Tensor output_tensor = use_extern_AVX2 ? IMatMul16ExternAVX2(A, B_marshalled, output_shape, input_dim, embed_dim, node.Name() + "_IMatMulExternAVX2")
                                                    : IMatMul16ExternMKL(A, B_marshalled, output_shape, input_dim, embed_dim, node.Name() + "_IMatMulExternMKL");
        outputs.push_back(output_tensor);
      }

      return Status::OK();
    }
  }

  tvm::Tensor Y = GenericMatMulInteger(inputs, node);
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
  return EvaluateMatMulInteger16(inputs, node, ctx_codegen, outputs);
}

}  // namespace nuphar
}  // namespace onnxruntime
