// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"

#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"

namespace onnxruntime {
namespace nuphar {

tvm_codegen::Scheduler* SCHEDULE_DISPATCHER_CLASS(NupharX86UseCount)::
    Find(const tvm::Tensor&, const Node* node, tvm_codegen::CodeGenContext& ctx) {
  if (nullptr == node)
    return nullptr;

  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx);
  bool reused = Promote<CodeGenUnitStats>(ctx_nuphar->GetGraphStats())->NodeUseCount(node) > 1;
  bool cheap_node_reused = Promote<CodeGenUnitStats>(ctx_nuphar->GetGraphStats())->IsCheapNodeReuse(node);

  if (reused && cheap_node_reused) {
    return DispatcherBase::Get("True");
  }
  return DispatcherBase::Get("False");
}

tvm_codegen::Scheduler* SCHEDULE_DISPATCHER_CLASS(NupharX86PartialResult)::
    Find(const tvm::Tensor&, const Node* node, tvm_codegen::CodeGenContext&) {
  if (nullptr == node)
    return DispatcherBase::Get("True");
  return nullptr;
}

tvm_codegen::Scheduler* SCHEDULE_DISPATCHER_CLASS(NupharX86Tensorize)::
    Find(const tvm::Tensor& tensor, const Node* node, tvm_codegen::CodeGenContext&) {
  if (nullptr == node)
    return nullptr;

  // Special checking to bypass tensorization
  // when fall back to extern function call
  if (tensor->op->InputTensors().size() == 2) {
    auto extern_op = tensor->op.as<tvm::ExternOpNode>();
    // Extern function call
    if (nullptr != extern_op)
      return nullptr;
  }

  return DispatcherBase::Get(node->OpType());
}

}  // namespace nuphar
}  // namespace onnxruntime
