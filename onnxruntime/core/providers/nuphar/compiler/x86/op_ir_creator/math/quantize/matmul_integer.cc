// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/codegen/mti/math/binary_ops.h"                          // remove this after removing tvm core code out
#include "core/codegen/mti/math/matmul_ops.h"                          // remove this after removing tvm core code out
#include "core/providers/nuphar/mti_x86/quantize/qmatmul_symm_ops.h"   // remove this after removing tvm core code out
#include "core/providers/nuphar/mti_x86/quantize/qmatmul_asymm_ops.h"  // remove this after removing tvm core code out
#include "core/codegen/mti/tensor/cast_ops.h"                          // remove this after removing tvm core code out
#include "core/codegen/mti/tensor/reshape_ops.h"                       // remove this after removing tvm core code out
#include "core/codegen/mti/tensor/transpose.h"                         // remove this after removing tvm core code out
#include "core/codegen/passes/weight_layout/transpose_2d.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/common/cpuid_info.h"  // refactor this after move NUPHAR_USE_AVX2 common place

namespace onnxruntime {
namespace nuphar {

// Evaluate of MatMulInteger or MatMulInteger16
static Status EvaluateMatMul(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  const auto& lhs_tensor = inputs[0];
  const auto& rhs_tensor = inputs[1];

  auto& name = node.Name();

  if (rhs_tensor->shape.size() == 2) {
    const int64_t* p_input_dim = tvm::as_const_int(rhs_tensor->shape[0]);
    const int64_t* p_embed_dim = tvm::as_const_int(rhs_tensor->shape[1]);

    if (p_input_dim && p_embed_dim) {
      int64_t input_dim = *p_input_dim;
      int64_t embed_dim = *p_embed_dim;

      // special case for MatMulInteger in quantized RNN
      bool is16bitSymm = (lhs_tensor->dtype == HalideIR::type_of<int16_t>() &&
                          rhs_tensor->dtype == HalideIR::type_of<int16_t>());
      bool is8bitAsymm = (lhs_tensor->dtype == HalideIR::type_of<uint8_t>() &&
                          rhs_tensor->dtype == HalideIR::type_of<int8_t>());

      if (is16bitSymm || is8bitAsymm) {
        auto input_shape = lhs_tensor->shape;
        auto input_rank = gsl::narrow_cast<int>(input_shape.size());

        tvm::Array<tvm::Expr> output_shape;
        for (int i = 0; i < input_rank - 1; ++i) {
          output_shape.push_back(input_shape[i]);
        }
        output_shape.push_back(tvm::Expr(gsl::narrow_cast<int>(embed_dim)));

        auto quantized_param = rhs_tensor;
        tvm::Tensor quantized_marshalled;
        const std::string& quantized_param_name = node.InputDefs()[1]->Name();

        if (ctx_nuphar->IsInitializer(quantized_param_name)) {
          auto layout_key = tvm_codegen::WeightLayoutTranspose2D::GetKey(TensorProtoDataType(node.InputDefs()[1]));
          quantized_marshalled = ctx_nuphar->ApplyWeightLayout(layout_key, quantized_param_name, quantized_param, true);
        } else {
          quantized_marshalled = tvm_codegen::Transpose(quantized_param, {1, 0});
        }

        // reserved_bits should be checked somewhere
        // int reserved_bits = 1;  // force it to use AVX2 when possible
        bool use_AVX2 = CPUIDInfo::GetCPUIDInfo().HasAVX2();
        auto output_tensor =
            is16bitSymm ? use_AVX2 ? QMatMulSymmetricAVX2(quantized_marshalled, lhs_tensor,
                                                          output_shape, input_dim, embed_dim,
                                                          name + "_QMatMulSymmetricAVX2")
                                   : QMatMulSymmetricMKL(quantized_marshalled, lhs_tensor,
                                                         output_shape, input_dim, embed_dim,
                                                         name + "_QMatMulSymmetricMKL")
                        : use_AVX2 ? QMatMulAsymmetricAVX2(quantized_marshalled, lhs_tensor,
                                                           output_shape, input_dim, embed_dim,
                                                           name + "_QMatMulAsymmetricAVX2")
                                   : QMatMulAsymmetricMKL(quantized_marshalled, lhs_tensor,
                                                          output_shape, input_dim, embed_dim,
                                                          name + "_QMatMulAsymmetricMKL");

        outputs.push_back(output_tensor);
        return Status::OK();
      }
    }
  }
  // slow path, cast to int32 for now
  // Support skipped trailing inputs
  auto lhs = (node.InputDefs().size() >= 3 && node.InputDefs()[2]->Exists())
                 ? tvm_codegen::Sub(tvm_codegen::Cast(lhs_tensor, HalideIR::Int(32)), tvm_codegen::Cast(inputs[2], HalideIR::Int(32)))
                 : tvm_codegen::Cast(lhs_tensor, HalideIR::Int(32));
  auto rhs = (node.InputDefs().size() >= 4 && node.InputDefs()[3]->Exists())
                 ? tvm_codegen::Sub(tvm_codegen::Cast(rhs_tensor, HalideIR::Int(32)), tvm_codegen::Cast(inputs[3], HalideIR::Int(32)))
                 : tvm_codegen::Cast(rhs_tensor, HalideIR::Int(32));
  tvm::Tensor Y = tvm_codegen::MatMul(lhs, rhs, name);
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of MatMulInteger OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(MatMulInteger)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  return EvaluateMatMul(inputs, node, ctx_codegen, outputs);
}

// Evaluate of MatMulInteger16 OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(MatMulInteger16)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  return EvaluateMatMul(inputs, node, ctx_codegen, outputs);
}

}  // namespace nuphar
}  // namespace onnxruntime
