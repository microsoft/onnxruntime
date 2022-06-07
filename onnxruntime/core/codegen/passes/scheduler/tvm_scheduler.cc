// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/scheduler/tvm_scheduler.h"

#include "core/codegen/common/common.h"
#include "core/codegen/common/dispatcher.h"
#include "core/codegen/passes/utils/codegen_context.h"

namespace onnxruntime {
namespace codegen {
// explicit instantiation
template class CreatorBase<const tvm::Tensor&,
                           const Node*,
                           tvm_codegen::CodeGenContext&,
                           tvm_codegen::ScheduleContext&,
                           bool>;

template class DispatcherBase<tvm_codegen::Scheduler*>;

}  // namespace codegen

namespace tvm_codegen {

static const std::string TMVOpRuleKey_Extern("TVMOpRule_Extern");
static const std::string TMVOpRuleKey_ComputeReduce("TVMOpRule_ComputeReduce");
static const std::string TMVOpRuleKey_ComputeRegular("TVMOpRule_ComputeRegular");
static const std::string TMVOpRuleKey_AlwaysRoot("TMVOpRuleKey_AlwaysRoot");
static const std::string TMVOpRuleKey_NoRule("TVMOpRule_NoRule");

const std::string& GetTVMOpRule(TVMOpRuleType rule) {
  if (rule == TVMOpRuleType::Extern) {
    return TMVOpRuleKey_Extern;
  } else if (rule == TVMOpRuleType::ComputeReduce) {
    return TMVOpRuleKey_ComputeReduce;
  } else if (rule == TVMOpRuleType::AlwaysRoot) {
    return TMVOpRuleKey_AlwaysRoot;
  }
  return TMVOpRuleKey_NoRule;
}

const std::string& GetTVMOpRule(const tvm::Tensor& tensor) {
  auto extern_op = tensor->op.as<tvm::ExternOpNode>();

  if (nullptr != extern_op) {
    return TMVOpRuleKey_Extern;
  }

  auto compute_op = tensor->op.as<tvm::ComputeOpNode>();
  if (nullptr != compute_op) {
    if (compute_op->reduce_axis.size() > 0) {
      return TMVOpRuleKey_ComputeReduce;
    }
  }

  return TMVOpRuleKey_NoRule;
}

Scheduler* SCHEDULE_DISPATCHER_CLASS(OrtOpType)::
    Find(const tvm::Tensor&, const Node* node, tvm_codegen::CodeGenContext&) {
  if (nullptr == node)
    return nullptr;
  return DispatcherBase::Get(node->OpType());
}

Scheduler* SCHEDULE_DISPATCHER_CLASS(TVMOpRule)::
    Find(const tvm::Tensor& tensor, const Node*, tvm_codegen::CodeGenContext&) {
  return DispatcherBase::Get(GetTVMOpRule(tensor));
}

Scheduler* SCHEDULE_DISPATCHER_CLASS(OrtOpName)::
    Find(const tvm::Tensor&, const Node* node, tvm_codegen::CodeGenContext&) {
  if (nullptr == node)
    return nullptr;
  return DispatcherBase::Get(GetKey(node));
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
