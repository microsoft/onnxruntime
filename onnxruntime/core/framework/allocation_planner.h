// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/platform/path_lib.h"
#include "core/framework/alloc_kind.h"
#include "core/framework/allocator.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/session_options.h"

namespace ONNX_NAMESPACE {
class TensorShapeProto;
}
namespace onnxruntime {

namespace NestedSubgraphInfoDetails {

// Used to compose a unique key to identify a nested subgraph
// relative to a current graph level (which in turn is identified using a "base")
std::string ComposeNestedSubgraphInfoKeyHelper(const std::string& base, size_t graph_depth,
                                               NodeIndex node_index, const std::string& attr_name);

}  // namespace NestedSubgraphInfoDetails

class ExecutionProviders;
struct KernelCreateInfo;
class KernelRegistryManager;
class OrtValueNameIdxMap;
class IStreamCommandHandleRegistry;

using KernelCreateInfoMap = std::unordered_map<onnxruntime::NodeIndex, gsl::not_null<const KernelCreateInfo*>>;
using SubgraphsKernelCreateInfoMaps = std::unordered_map<std::string, KernelCreateInfoMap>;

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
  virtual ~ISequentialPlannerContext() = default;
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

#ifdef ORT_ENABLE_STREAM
// Given a graph with node placement information, partition the nodes into multiple sequence.
// Each sequence can be executed in-dependently. The nodes in each sequence are executed in order,
// but we can't assume any execution order between sequences, unless there is a data dependency.
class IGraphPartitioner {
 public:
  // DeviceBasedPartitioner is the default, who partitions a graph based off device information.
  // i.e., given a graph which has CPU EP nodes, Cuda EP nodes and TRT EP nodes,
  // it will be partitioned as two sequences, one is for CPU EP nodes, another is for TRT and Cuda nodes.
  // We will add more optimized partitioner later.
  enum GraphPartitioningStrategy {
    DeviceBasedPartition = 0,
    Unknown,
  };
  virtual ~IGraphPartitioner() = default;
  // create the partition based on the partition type.
  // perform partition based on the user input when provided.
  static std::unique_ptr<IGraphPartitioner> CreateGraphPartitioner(const logging::Logger& logger,
                                                                   const PathString& config_file);
  virtual Status PartitionGraph(const onnxruntime::GraphViewer& graph_viewer,
                                const ExecutionProviders& execution_providers,
                                std::vector<InlinedVector<NodeIndex>>& stream_nodes,
                                ExecutionOrder execution_order) = 0;
  virtual const char* Type() const = 0;
  // return total number of streams
  virtual size_t Streams() const = 0;

 protected:
  IGraphPartitioner(const logging::Logger& logger,
                    const PathString& config_file) : logger_(logger),
                                                     config_file_(config_file) {}
  const logging::Logger& logger_;
  PathString config_file_;
};
#endif

class SequentialPlanner {
 public:
  // This API allows user to provide a custom planner context.
  // TODO - remove duplicated ExecutionProvider argument
  static Status CreatePlan(
      const Node* parent_node, const onnxruntime::GraphViewer& graph,
      gsl::span<const NodeArg* const> outer_scope_node_args,
      const ExecutionProviders& providers,
      const KernelCreateInfoMap& kernel_create_info_map,
      const SubgraphsKernelCreateInfoMaps& subgraphs_kernel_create_info_maps,
      const InlinedHashMap<OrtValueName, OrtDevice>& outer_scope_arg_to_location_map,
      const OrtValueNameIdxMap& ort_value_name_idx_map,
      const ISequentialPlannerContext& context,
#ifdef ORT_ENABLE_STREAM
      const IStreamCommandHandleRegistry& stream_handle_registry,
#endif
      const std::basic_string<PATH_CHAR_TYPE>& partition_config_file,
      const logging::Logger& logger,
      std::optional<SequentialExecutionPlan>& plan);
};

}  // namespace onnxruntime
