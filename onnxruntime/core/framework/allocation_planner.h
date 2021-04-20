// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/alloc_kind.h"
#include "core/framework/allocator.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/session_options.h"

namespace ONNX_NAMESPACE {
class TensorShapeProto;
}
namespace onnxruntime {

class ExecutionProviders;
struct KernelCreateInfo;
class KernelRegistryManager;
class OrtValueNameIdxMap;

// ISequentialPlannerContext abstracts how the planner accesses information (such as inferred shape)
// to do the planning.
class ISequentialPlannerContext {
 public:
  virtual const ONNX_NAMESPACE::TensorShapeProto* GetShape(const onnxruntime::NodeArg& arg) const = 0;
  // If it returns true, planner won't reuse output tensors
  // see PlannerImpl::ComputeReusePlan
  virtual bool IsParallelExecutionEnabled() const { return false; }

  virtual ExecutionOrder GetExecutionOrder() const { return ExecutionOrder::DEFAULT; }

  virtual bool GetEnableMemoryReuse() const { return true; }
};

class SequentialPlannerContext : public ISequentialPlannerContext {
 public:
  SequentialPlannerContext(ExecutionMode execution_mode, ExecutionOrder execution_order, bool enable_memory_reuse)
      : execution_mode_(execution_mode),
        exection_order_(execution_order),
        enable_memory_reuse_(enable_memory_reuse) {
  }

  const ONNX_NAMESPACE::TensorShapeProto* GetShape(const onnxruntime::NodeArg& arg) const override {
    return arg.Shape();
  }

  bool IsParallelExecutionEnabled() const override { return execution_mode_ == ExecutionMode::ORT_PARALLEL; }

  ExecutionOrder GetExecutionOrder() const override { return exection_order_; }

  bool GetEnableMemoryReuse() const override { return enable_memory_reuse_; }

 private:
  ExecutionMode execution_mode_ = ExecutionMode::ORT_SEQUENTIAL;
  ExecutionOrder exection_order_ = ExecutionOrder::DEFAULT;
  bool enable_memory_reuse_ = true;
};

class SequentialPlanner {
 public:
  // This API allows user to provide a custom planner context.
  static Status CreatePlan(
      const Node* parent_node, const onnxruntime::GraphViewer& graph,
      const std::vector<const NodeArg*>& outer_scope_node_args,
      const ExecutionProviders& providers,
      const std::unordered_map<NodeIndex, gsl::not_null<const KernelCreateInfo*>>& kernel_create_info_map,
      const OrtValueNameIdxMap& ort_value_name_idx_map,
      const ISequentialPlannerContext& context,
      std::unique_ptr<SequentialExecutionPlan>& plan);
};

}  // namespace onnxruntime
