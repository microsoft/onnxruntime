// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/cast_ops.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/codegen/mti/tensor/transpose.h"
#include "core/codegen/passes/weight_layout/transpose_2d.h"
#include "core/common/cpuid_info.h"  // TODO: refactor to control through config
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/mti_x86/quantize/imatmul_extern.h"
#include "core/providers/nuphar/mti_x86/quantize/imatmul16_extern.h"

namespace onnxruntime {
namespace nuphar {

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

      bool is16bitSymm = (A->dtype == HalideIR::type_of<int16_t>() &&
                          B->dtype == HalideIR::type_of<int16_t>());
      bool is8bitAsymm = (A->dtype == HalideIR::type_of<uint8_t>() &&
                          B->dtype == HalideIR::type_of<int8_t>());

      if (is16bitSymm || is8bitAsymm) {
        auto A_rank = gsl::narrow_cast<int>(A->shape.size());

        tvm::Array<tvm::Expr> output_shape;
        for (int i = 0; i < A_rank - 1; ++i) {
          output_shape.push_back(A->shape[i]);
        }
        output_shape.push_back(tvm::Expr(gsl::narrow_cast<int>(embed_dim)));

        tvm::Tensor B_marshalled;
        auto B_NodeArg = node.InputDefs()[1];
        const std::string& B_name = B_NodeArg->Name();

        if (ctx_nuphar->IsInitializer(B_name)) {
          auto layout_key = tvm_codegen::WeightLayoutTranspose2D::GetKey(TensorProtoDataType(B_NodeArg));
          B_marshalled = ctx_nuphar->ApplyWeightLayout(layout_key, B_name, B, true);
        } else {
          B_marshalled = tvm_codegen::Transpose(B, {1, 0});
        }

        // TODO: add reserved_bits attribute
        bool use_AVX2;
        const codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
        if (settings.HasOption(kNupharIMatMulForceMkl)) {
          use_AVX2 = false;
        } else {
          use_AVX2 = CPUIDInfo::GetCPUIDInfo().HasAVX2();
        }
        auto output_tensor =
            is16bitSymm ? use_AVX2 ? IMatMul16ExternAVX2(B_marshalled, A,
                                                         output_shape, input_dim, embed_dim,
                                                         name + "_IMatMul16ExternAVX2")
                                   : IMatMul16ExternMKL(B_marshalled, A,
                                                        output_shape, input_dim, embed_dim,
                                                        name + "_IMatMul16ExternMKL")
                        : use_AVX2 ? IMatMulExternAVX2(B_marshalled, A,
                                                       output_shape, input_dim, embed_dim,
                                                       name + "_IMatMulExternAVX2")
                                   : IMatMulExternMKL(B_marshalled, A,
                                                      output_shape, input_dim, embed_dim,
                                                      name + "_IMatMulExternMKL");

        outputs.push_back(output_tensor);
        return Status::OK();
      }
    }
  }
  // slow path, cast to int32 for now
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
