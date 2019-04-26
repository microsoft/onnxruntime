// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_viewer.h"
#include "gsl/span"
#include "gsl/gsl_util"

namespace onnxruntime {

class MLValueNameIdxMap;
class FuncManager;

// A very light-weight class, which works as an aggregated
// view of all data needed for constructing a Kernel instance.
// NOTE: it does not own/hold any objects.
class OpKernelInfo : public OpNodeProtoHelper<ProtoHelperNodeContext> {
 public:
  explicit OpKernelInfo(const onnxruntime::Node& node,
                        const KernelDef& kernel_def,
                        const IExecutionProvider& execution_provider,
                        const std::unordered_map<int, MLValue>& initialized_tensors,
                        const MLValueNameIdxMap& mlvalue_name_idx_map,
                        const FuncManager& funcs_mgr);

  OpKernelInfo(const OpKernelInfo& other);

  const OrtAllocatorInfo& GetAllocatorInfo(int device_id, OrtMemType mem_type) const;

  const AllocatorPtr GetAllocator(int device_id, OrtMemType mem_type) const;

  const KernelDef& GetKernelDef() const;

  const IExecutionProvider* GetExecutionProvider() const noexcept;

  const onnxruntime::Node& node() const noexcept;

  bool TryGetConstantInput(int input_index, const Tensor** constant_input_value) const;

  common::Status GetFusedFuncs(ComputeFunc* compute,
                               CreateFunctionStateFunc* create,
                               DestroyFunctionStateFunc* release) const;

 private:
  ORT_DISALLOW_MOVE(OpKernelInfo);
  ORT_DISALLOW_ASSIGNMENT(OpKernelInfo);

  const onnxruntime::Node& node_;
  const KernelDef& kernel_def_;
  // For non cpu/cuda case, this pointer should be set so that function kernel
  // will delegate kernel compute call to <execution_provider> compute call.
  gsl::not_null<const ::onnxruntime::IExecutionProvider*> execution_provider_;
  const std::unordered_map<int, MLValue>& initialized_tensors_;
  const MLValueNameIdxMap& mlvalue_name_idx_map_;
  const FuncManager& funcs_mgr_;
  ProtoHelperNodeContext proto_helper_context_;
};

}  // namespace onnxruntime
