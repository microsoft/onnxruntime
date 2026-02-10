// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "telum_common.h"

namespace onnxruntime {
namespace telum {

/**
 * @brief Telum Execution Provider for IBM z16 (Telum) hardware acceleration
 *
 * This execution provider leverages zDNN to accelerate neural network operations
 * on IBM z16 processors with Neural Network Processing Assist (NNPA) facility.
 *
 * Key characteristics:
 * - Synchronous execution model
 * - Static shape requirements
 * - Optimized for transformer inference
 * - Strict validation with explicit error reporting
 */
class TelumExecutionProvider : public IExecutionProvider {
 public:
  explicit TelumExecutionProvider(const TelumExecutionProviderInfo& info);
  virtual ~TelumExecutionProvider();

  // IExecutionProvider interface implementation
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  /**
   * @brief Get execution provider's capability for the specified graph
   *
   * Analyzes the graph and returns subgraphs that can be executed on Telum.
   * Only claims nodes/subgraphs that meet all hardware constraints.
   *
   * @param graph The graph to analyze
   * @param kernel_lookup Interface for kernel lookup
   * @param graph_optimizer_registry Registry of graph optimizers (optional; unused for now)
   * @return Vector of compute capabilities (subgraphs this EP can handle)
   */
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph,
                const IKernelLookup& kernel_lookup,
                const GraphOptimizerRegistry& graph_optimizer_registry,
                IResourceAccountant* resource_accountant = nullptr) const override;

  /**
   * @brief Compile fused nodes into executable compute functions
   *
   * @param fused_nodes Nodes that have been fused together
   * @param node_compute_funcs Output vector of compute functions
   * @return Status indicating success or failure
   */
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                        std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  /**
   * @brief Check if a node is supported by Telum EP
   *
   * @param node Node to check
   * @return true if node is supported, false otherwise
   */
  bool IsNodeSupported(const Node& node) const;

  /**
   * @brief Validate that node has static shapes
   *
   * @param node Node to validate
   * @return true if all shapes are static, false otherwise
   */
  bool ValidateStaticShapes(const Node& node) const;

  /**
   * @brief Check if node's data types are supported
   *
   * @param node Node to check
   * @return true if data types are supported, false otherwise
   */
  bool ValidateDataTypes(const Node& node) const;

  /**
   * @brief Get reason why a node is not supported
   *
   * @param node Node to analyze
   * @return String describing why node is not supported
   */
  std::string GetRejectionReason(const Node& node) const;

  /**
   * @brief Register graph transformers for fusion optimizations
   */
  void RegisterGraphTransformers();

  /**
   * @brief Check if operator is in supported list
   *
   * @param op_type ONNX operator type
   * @return true if operator is supported, false otherwise
   */
  bool IsOperatorSupported(const std::string& op_type) const;

  // Configuration
  TelumExecutionProviderInfo info_;

  // Supported operator types
  std::unordered_set<std::string> supported_ops_;
};

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
