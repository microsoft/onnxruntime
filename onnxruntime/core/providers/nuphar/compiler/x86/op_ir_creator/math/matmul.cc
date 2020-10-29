// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/passes/weight_layout/transpose_2d.h"
#include "core/codegen/passes/weight_layout/vertical_stripes_2d.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"
#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/providers/nuphar/mti_x86/math/matmul_ops.h"

#include <tvm/ir_pass.h>

namespace onnxruntime {
namespace nuphar {

// TODO: remove tvm core function

// local helper functions

static bool MatMul_weights2D(
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    const tvm::Tensor& A,
    const tvm::Tensor& B,
    const std::string& initializer_name,
    NupharCodeGenCtx& ctx_codegen,
    tvm::Tensor& Y,
    const std::string& name = "matmul_weights2d") {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  // optimizations for B being 2D weights

  // The 2D weight is marshalled with stripe_width.
  // This should be 2x nature vector width
  int stripe_width = 8;
  int block_size = 32;

  onnxruntime::CodeGenTargetX86* target =
      dynamic_cast<onnxruntime::CodeGenTargetX86*>(ctx_codegen.GetCodeGenHandle()->codegen_target);
  if (nullptr != target) {
    stripe_width = 2 * target->NaturalVectorWidth(B->dtype.bits());
  }

  // align A, B to multiple of block size
  const auto& A_shape = A->shape;
  tvm::Expr A0_size = tvm_codegen::SizeToDimension(A_shape, -1);
  auto A0_roundup = tvm_codegen::RoundUp(A0_size, block_size);
  tvm::Expr A1_size = tvm_codegen::SizeFromDimension(A_shape, -1);
  auto A1_roundup = tvm_codegen::RoundUp(A1_size, block_size);
  bool A0_need_pad = !tvm::ir::Equal(A0_roundup, A0_size);
  bool A1_need_pad = !tvm::ir::Equal(A1_roundup, A1_size);

  const auto& B_shape = B->shape;
  tvm::Expr B0_size = tvm_codegen::SizeToDimension(B_shape, 1);
  auto B0_roundup = tvm_codegen::RoundUp(B0_size, block_size);
  tvm::Expr B1_size = tvm_codegen::SizeFromDimension(B_shape, 1);
  auto B1_roundup = tvm_codegen::RoundUp(B1_size, block_size);
  bool B1_need_pad = !tvm::ir::Equal(B1_roundup, B1_size);

  ORT_ENFORCE(tvm::ir::Equal(A1_roundup, B0_roundup));

  // Currently only support padding in B1, as it's free with memory marshalling
  if (A0_need_pad || A1_need_pad || B1_need_pad)
    return false;

  auto layout_key = tvm_codegen::WeightLayoutVerticalStripe2D::GetKey(proto_type, stripe_width);
  auto B_unmarshalled = ctx_nuphar->ApplyWeightLayout(layout_key, initializer_name, B, false);

  ORT_ENFORCE(B_unmarshalled->op.as<tvm::ComputeOpNode>());

  tvm::Array<tvm::Expr> Y_shape;
  for (size_t d = 0; d < A->shape.size() - 1; ++d)
    Y_shape.push_back(A->shape[d]);
  Y_shape.push_back(B->shape[1]);

  auto k = tvm::reduce_axis(tvm::Range(0, A1_size), "k");
  Y = tvm::compute(
      Y_shape,
      [&](const tvm::Array<tvm::Var>& idx) {
        tvm::Array<tvm::Expr> A_indices;
        for (size_t d = 0; d < idx.size() - 1; ++d)
          A_indices.push_back(idx[d]);
        A_indices.push_back(k);
        return tvm::sum(A(A_indices) * B_unmarshalled(k, idx[idx.size() - 1]), {k});
      },
      name);

  return true;
}

static bool MatMulF32ExternCPU(
    tvm::Tensor A,
    tvm::Tensor B,
    tvm::Tensor& Y,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  // try to fuse tranpose in MatMul input with MatMul
  auto find_transposed_input = [&ctx_nuphar](const tvm::Tensor& t, std::vector<int32_t>& cumulated_permute) {
    tvm::Tensor out = t;
    int64_t rank = gsl::narrow<int64_t>(t->shape.size());
    std::vector<int64_t> default_node_perm(rank);
    cumulated_permute.resize(rank);
    for (int64_t i = 0; i < rank; ++i) {
      cumulated_permute[i] = gsl::narrow<int32_t>(i);
      default_node_perm[i] = rank - i - 1;
    }
    for (const Node* root_node = ctx_nuphar->FindNode(out);
         root_node != nullptr && root_node->OpType() == "Transpose";
         root_node = ctx_nuphar->FindNode(out)) {
      ProtoHelperNodeContext ctx(*root_node);
      OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
      auto perm = info.GetAttrsOrDefault("perm", default_node_perm);
      std::vector<int32_t> updated_cumulated_permute = cumulated_permute;
      for (int64_t dst_dim = 0; dst_dim < rank; ++dst_dim) {
        auto src_dim = tvm_codegen::HandleNegativeAxis(perm[cumulated_permute[dst_dim]], rank);
        updated_cumulated_permute[dst_dim] = gsl::narrow<int32_t>(src_dim);
      }
      cumulated_permute = updated_cumulated_permute;
      // op corresponding to node should be Transpose
      auto op = out->op.as<tvm::ComputeOpNode>();
      ORT_ENFORCE(op != nullptr);
      ORT_ENFORCE(op->InputTensors().size() == 1);
      out = op->InputTensors()[0];
    }
    return out;
  };

  std::vector<int32_t> permute_A;
  std::vector<int32_t> permute_B;
  const std::vector<int32_t>* p_permute_A = nullptr;
  const std::vector<int32_t>* p_permute_B = nullptr;
  tvm::Tensor root_A = find_transposed_input(A, permute_A);
  tvm::Tensor root_B = find_transposed_input(B, permute_B);
  bool transA = false;
  if (A->shape.size() == B->shape.size() && A->shape.size() >= 2) {
    // currently only fuse Transpose into MatMul when rank(A) == rank(B)
    // make sure no broadcasting in MatMul
    bool no_broadcast = true;
    for (size_t i = 0; i < A->shape.size() - 2; ++i) {
      if (!tvm::ir::Equal(A->shape[i], B->shape[i])) {
        no_broadcast = false;
        break;
      }
    }
    if (no_broadcast) {
      if (CanPermuteBeFusedInMatMul(permute_A)) {
        if (A != root_A)
          transA = true;
        A = root_A;
        p_permute_A = &permute_A;
      }
      if (CanPermuteBeFusedInMatMul(permute_B)) {
        B = root_B;
        p_permute_B = &permute_B;
      }
    }
  }

  const auto& B_name = node.InputDefs()[1]->Name();
  if (ctx_nuphar->IsInitializer(B_name) && B->shape.size() == 2) {
    // matmul with initializer, using transpose weights
    auto layout_key = tvm_codegen::WeightLayoutTranspose2D::GetKey(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto actual_B = ctx_nuphar->ApplyWeightLayout(layout_key, B_name, B, true);
    return nuphar::GemmExternCpu(A, actual_B, Y, transA, true, B_name);
  } else {
    return nuphar::MatMulExternCpu(A, B, Y, p_permute_A, p_permute_B, node.Name() + "_matmul_extern");
  }
}

Status
NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(MatMul)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

  auto proto_type = TensorProtoDataType(node.InputDefs()[1]);

  tvm::Tensor Y;
  auto& A = inputs[0];
  auto& B = inputs[1];

  // float MatMul, try use extern
  if (A->dtype == HalideIR::Float(32) &&
      B->dtype == HalideIR::Float(32) &&
      MatMulF32ExternCPU(A, B, Y, node, ctx_codegen)) {
    outputs.push_back(Y);
    return Status::OK();
  }

  // if B is 2D initializer, use vertical stripe layout
  const std::string& input_1_name = node.InputDefs()[1]->Name();
  if (ShapeRank(node.InputDefs()[1]) == 2 && ctx_nuphar->IsInitializer(input_1_name)) {
    if (MatMul_weights2D(proto_type, A, B, input_1_name, *ctx_nuphar, Y)) {
      outputs.push_back(Y);
      return Status::OK();
    }
  }

  Y = nuphar::MatMul(A, B, node.Name());
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
