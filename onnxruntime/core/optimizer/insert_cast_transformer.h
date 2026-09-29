// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/graph_partitioner.h"

namespace onnxruntime {

/**
@Class InsertCastTransformer

Transformer to insert cast node that casts float16 to float for cpu nodes
*/
class InsertCastTransformer : public onnxruntime::GraphTransformer {
 public:
  // Every kernel registry a node assigned to the CPU EP can draw a kernel from, in the priority order
  // KernelRegistryManager uses: custom registries first, the CPU EP's own registry last.
  using KernelRegistryList = InlinedVector<gsl::not_null<const KernelRegistry*>>;

  /**
   * @brief Initializer
   * @param name                    for logging purpose
   * @param cpu_kernel_registries   used to query whether an op node can be safely created. Pass the full list
   *                                from KernelRegistryManager::GetKernelRegistriesByProviderType so that kernels
   *                                registered through a custom registry are seen here as well; leaving them out
   *                                would make a node with a working custom fp16 kernel look kernel-less and get
   *                                rewritten to fp32, so the custom kernel would never run.
   */
  InsertCastTransformer(const std::string& name, KernelRegistryList cpu_kernel_registries,
                        OnPartitionAssignmentFunction on_partition_assignment_fn = {})
      : onnxruntime::GraphTransformer(name),
        cpu_kernel_registries_(std::move(cpu_kernel_registries)),
        force_cpu_fp32_(!cpu_kernel_registries_.empty()),
        on_partition_assignment_fn_(std::move(on_partition_assignment_fn)) {}

  /**
   * @brief Convenience initializer for callers that only have the CPU EP's own registry (e.g. tests).
   */
  InsertCastTransformer(const std::string& name, const KernelRegistry* cpu_kernel_registry,
                        OnPartitionAssignmentFunction on_partition_assignment_fn = {})
      : InsertCastTransformer(name,
                              cpu_kernel_registry != nullptr ? KernelRegistryList{cpu_kernel_registry}
                                                             : KernelRegistryList{},
                              std::move(on_partition_assignment_fn)) {}

 private:
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  bool NeedInsertCast(const onnxruntime::Node* node, const onnxruntime::NodeArg* input) const;

  const KernelRegistryList cpu_kernel_registries_;

  // Currently because we only have very few cpu kernels support float16, place those nodes on float16
  // will introduce many cast between fp32 and fp16, which will slow the execution.
  // A better solution is to have a cost model to evaluate does it works to place the node on float16.
  // Here for simplify, we only force the single-node-float16 sub-graph to float32
  const bool force_cpu_fp32_;

  // Optional callback to record when nodes are assigned to CPU EP by this transformer.
  // Reuses the same callback type as GraphPartitioner to maintain consistent EP assignment tracking.
  OnPartitionAssignmentFunction on_partition_assignment_fn_;
};
}  // namespace onnxruntime
