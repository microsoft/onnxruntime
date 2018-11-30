// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry_manager.h"

namespace onnxruntime {

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(onnxruntime::Graph& graph, const std::string& provider)
      : graph_(graph), provider_(provider) {}

  bool ModifyGraph(const KernelRegistryManager& schema_registries);

 private:
  void ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries);
  void AddCopyNode(onnxruntime::NodeArg* arg, bool is_input);
  void ProcessInitializers();

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransformerMemcpyImpl);

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
  std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacements_;
  onnxruntime::Graph& graph_;
  std::string provider_;
};

}  // namespace onnxruntime
