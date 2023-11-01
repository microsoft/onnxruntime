// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
#include <optional>
#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/framework/kernel_registry_manager.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/selectors_actions/actions.h"
#include "core/optimizer/selectors_actions/selector_action_transformer_apply_contexts.h"

namespace onnxruntime {

class Graph;
class GraphViewer;
class Node;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

// Base class for a selector which checks for a match and returns the set of nodes involved.
struct NodeSelector {
  // Select one or more nodes for an Action to process if the constraints are satisfied,
  // otherwise returns std::nullopt
  virtual std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const = 0;

  virtual ~NodeSelector() = default;

 protected:
  NodeSelector() = default;
};

#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

// class to manage a set of selector and associated actions
class SelectorActionRegistry {
 public:
  // The key is a string representing the op, optionally specifying the domain using ':' as the
  // separator with domain as the first part and operator as the second part, "<domain>:<operator>" or "<operator>".
  // For ops in kOnnxDomain, the domain should be left unspecified ("<operator>").
  // For ops in other domains, the domain should be specified ("<domain>:<operator>").
  // Ex: "Conv", "com.microsoft:Conv", "com.ms.internal.nhwc:Conv"
  using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

  // Helper function to create a key to OpVersionsMap using domain and op_type.
  static std::string OpVersionsMapKey(std::string_view op_type, std::string_view domain = kOnnxDomain) {
    return (domain == kOnnxDomain)
               ? std::string{op_type}
               : std::string{domain} + ":" + std::string{op_type};
  }

  struct Entry {
    Entry(const std::string& name_in,
#if !defined(ORT_MINIMAL_BUILD)
          const OpVersionsMap& ops_and_versions_in,
          std::unique_ptr<NodeSelector> selector_in,
#endif  // !defined(ORT_MINIMAL_BUILD)
          std::unique_ptr<Action> action_in)
        : name{name_in},
#if !defined(ORT_MINIMAL_BUILD)
          ops_and_versions{ops_and_versions_in},
          selector{std::move(selector_in)},
#endif  // !defined(ORT_MINIMAL_BUILD)
          action{std::move(action_in)} {
    }

    std::string name;

#if !defined(ORT_MINIMAL_BUILD)
    OpVersionsMap ops_and_versions;
    std::unique_ptr<NodeSelector> selector;
#endif  // !defined(ORT_MINIMAL_BUILD)

    std::unique_ptr<Action> action;
  };

  SelectorActionRegistry() noexcept = default;

  SelectorActionRegistry(SelectorActionRegistry&&) noexcept = default;
  SelectorActionRegistry& operator=(SelectorActionRegistry&&) noexcept = default;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SelectorActionRegistry);

#if !defined(ORT_MINIMAL_BUILD)

  // register a selector and action for the specified ops.
  // the name used in the registration is for matching the action when replaying the optimizations in a minimal build.
  // as it's stored in the ORT format model a shorter name is better. the name is scoped to this SelectorActionRegistry
  // instance (which is scoped to a single SelectorActionTransformer instance).
  void RegisterSelectorAndAction(const std::string& name,
                                 const OpVersionsMap& ops_and_versions,
                                 std::unique_ptr<NodeSelector> selector,
                                 std::unique_ptr<Action> action);

#else  // !defined(ORT_MINIMAL_BUILD)

  // register an action
  void RegisterAction(const std::string& name, std::unique_ptr<Action> action);

#endif  // !defined(ORT_MINIMAL_BUILD)

  // return registered Entry or nullptr if not found
  const Entry* LookUp(const std::string& name) const;

#if !defined(ORT_MINIMAL_BUILD)
  // return registered Entry or nullptr if not found
  auto LookUpByOpTypeAndDomain(const std::string& op_type,
                               const std::string& domain) const -> std::vector<gsl::not_null<const Entry*>>;
#endif  // !defined(ORT_MINIMAL_BUILD)

 private:
  std::unordered_map<std::string, const Entry> name_to_entry_;

#if !defined(ORT_MINIMAL_BUILD)
  // auxiliary mapping to enable lookup by op type or "domain:op type"
  std::unordered_multimap<std::string, const Entry*> op_type_to_entry_;
#endif  // !defined(ORT_MINIMAL_BUILD)
};

/**
Class that implements graph transformation via a set of Selector+Action pairs.
This setup allows optimizations to be captured and applied at runtime in a minimal build.
*/
class SelectorActionTransformer : public GraphTransformer {
 protected:
  SelectorActionTransformer(const std::string& name, SelectorActionRegistry&& selector_action_registry,
                            const SatApplyContextVariant& apply_context,
                            const InlinedHashSet<std::string_view>& compatible_execution_providers);

  // can't copy/assign selector_action_registry_
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SelectorActionTransformer);

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

#if !defined(ORT_MINIMAL_BUILD)

  // apply optimizations by selecting nodes from graph and running or saving the associated actions
  Status ApplySelectorsAndActions(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                                  const SatRuntimeOptimizationSaveContext* save_context) const;

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // apply optimizations by replaying saved runtime optimizations
  Status ApplySavedRuntimeOptimizations(Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger) const;
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  SelectorActionRegistry selector_action_registry_;

  SatApplyContextVariant apply_context_;
};

}  // namespace onnxruntime
