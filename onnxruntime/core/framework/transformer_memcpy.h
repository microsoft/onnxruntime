// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/graph/graph_transformer.h"

namespace onnxruntime {

class MemcpyTransformer : public GraphTransformer {
 public:
  MemcpyTransformer(const ExecutionProviders& providers, const KernelRegistryManager& registry_manager)
      : GraphTransformer("MemcpyTransformer", "Insert nodes to copy memory between devices when needed"),
        providers_{providers},
        registry_manager_{registry_manager} {}

 private:
  common::Status ApplyImpl(Graph& graph, bool& modified) const override;

  const ExecutionProviders& providers_;
  const KernelRegistryManager& registry_manager_;
};

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(onnxruntime::Graph& graph, const std::string& provider)
      : graph_(graph), provider_(provider) {}

  bool ModifyGraph(const KernelRegistryManager& schema_registries);

 private:
  void ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries);
  void BuildDefsMapping(const onnxruntime::NodeArg* arg, const KernelRegistryManager& kernel_registries);
  void AddCopyNode(onnxruntime::NodeArg* arg, bool is_input);
  void ProcessInitializers();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransformerMemcpyImpl);

  // use value-based compare to make sure transformer output order is consistent
  struct NodeCompare {
    bool operator()(const onnxruntime::Node* lhs, const onnxruntime::Node* rhs) const {
      return lhs->Index() < rhs->Index();
    }
  };

  // use value-based compare to make sure transformer output order is consistent
  struct NodeArgCompare {
    bool operator()(const onnxruntime::NodeArg* lhs, const onnxruntime::NodeArg* rhs) const {
      return lhs->Name() < rhs->Name();
    }
  };

  std::set<onnxruntime::Node*, NodeCompare> provider_nodes_;
  std::set<const onnxruntime::NodeArg*, NodeArgCompare> non_provider_input_defs_;  // all input defs of non-provider nodes
  std::set<onnxruntime::NodeArg*, NodeArgCompare> non_provider_output_defs_;       // all output defs of non-provider nodes
  std::set<const onnxruntime::NodeArg*, NodeArgCompare> provider_input_defs_;      // all input defs of provider nodes that should be in provider allocator
  std::set<onnxruntime::NodeArg*, NodeArgCompare> provider_output_defs_;           // all output defs of provider nodes that should be in provider allocator
  std::map<const onnxruntime::NodeArg*, std::set<onnxruntime::Node*, NodeCompare>> provider_input_nodes_;
  std::map<const onnxruntime::NodeArg*, std::set<onnxruntime::Node*, NodeCompare>> provider_output_nodes_;

  onnxruntime::Graph& graph_;
  std::string provider_;
};

}  // namespace onnxruntime
