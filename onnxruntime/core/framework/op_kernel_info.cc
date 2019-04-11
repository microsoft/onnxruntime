// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"

namespace onnxruntime {

OpKernelInfo::OpKernelInfo(const onnxruntime::Node& node,
                           const KernelDef& kernel_def,
                           const IExecutionProvider& execution_provider,
                           const std::unordered_map<int, MLValue>& initialized_tensors,
                           const SessionState& session_state)
    : OpNodeProtoHelper(&proto_helper_context_),
      node_(node),
      kernel_def_(kernel_def),
      execution_provider_(&execution_provider),
      initialized_tensors_(initialized_tensors),
      session_state_(session_state),
      proto_helper_context_(node) {}

OpKernelInfo::OpKernelInfo(const OpKernelInfo& other)
    : OpKernelInfo(other.node_,
                   other.kernel_def_,
                   *other.execution_provider_,
                   other.initialized_tensors_,
                   other.session_state_) {}

const OrtAllocatorInfo& OpKernelInfo::GetAllocatorInfo(int device_id, OrtMemType mem_type) const {
  AllocatorPtr alloc = GetAllocator(device_id, mem_type);
  if (alloc == nullptr) ORT_THROW("cannot find allocator");
  return alloc->Info();
}

const AllocatorPtr OpKernelInfo::GetAllocator(int device_id, OrtMemType mem_type) const {
  return execution_provider_->GetAllocator(device_id, mem_type);
}

const KernelDef& OpKernelInfo::GetKernelDef() const {
  return kernel_def_;
}

const IExecutionProvider* OpKernelInfo::GetExecutionProvider() const noexcept {
  return execution_provider_;
}

const onnxruntime::Node& OpKernelInfo::node() const noexcept {
  return node_;
}

bool OpKernelInfo::TryGetConstantInput(int input_index, const Tensor** constant_input_value) const {
  if (input_index < 0 || input_index >= gsl::narrow_cast<int>(node_.InputDefs().size())) {
    return false;
  }
  auto& input_arg_name = node_.InputDefs()[input_index]->Name();
  int input_arg_index = -1;
  if (!session_state_.GetMLValueNameIdxMap().GetIdx(input_arg_name, input_arg_index).IsOK()) {
    return false;
  }

  auto iter = initialized_tensors_.find(input_arg_index);
  if (initialized_tensors_.end() == iter) {
    return false;
  }
  if (!iter->second.IsTensor()) {
    // Only constant Tensor input is support right now, since we're using initializers to store the data.
    return false;
  }
  *constant_input_value = &iter->second.Get<Tensor>();
  return true;
}

common::Status OpKernelInfo::GetFusedFuncs(ComputeFunc* compute, CreateFunctionStateFunc* create, DestroyFunctionStateFunc* release) const {
  return session_state_.GetFuncMgr().GetFuncs(node_.Name(), compute, create, release);
}

common::Status OpKernelInfo::GetOutputTensorAllocator(int output_id, AllocatorPtr& allocator) const{
  ORT_ENFORCE(output_id >= 0 && output_id < node_.OutputDefs().size());
  auto* arg = node_.OutputDefs()[output_id];
  int ml_index;
  ORT_RETURN_IF_ERROR(session_state_.GetMLValueNameIdxMap().GetIdx(arg->Name(), ml_index));
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  ORT_ENFORCE(ml_index >= 0 && ml_index < alloc_plan.size());
  const auto& per_alloc_plan = alloc_plan[ml_index];

  auto alloc_info = per_alloc_plan.location;
  auto ml_type = per_alloc_plan.value_type;
  if (ml_type == nullptr)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Tried to allocate without valid type information, mlvalue index=" + std::to_string(ml_index));

  if (!ml_type->IsTensorType()) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "No allocator for non-tensor type value, mlvalue index=" + std::to_string(ml_index));
  }

  ORT_ENFORCE(per_alloc_plan.alloc_kind == AllocKind::kAllocate || per_alloc_plan.alloc_kind == AllocKind::kAllocateOutput);
  allocator = utils::GetAllocator(session_state_, per_alloc_plan.location);
  return Status::OK();
}
}  // namespace onnxruntime
