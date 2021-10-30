// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

class Graph;
class Node;

#if !defined(ORT_MINIMAL_BUILD)
// Base class for a selector which checks for a match and returns the set of nodes involved.
struct NodeSelector {
  // Select one or more nodes for an Action to process if the constraints are satisfied.
  // `selection` should not be set if this returns false
  virtual bool Select(Graph& graph, const Node& node, std::unique_ptr<NodesToOptimize>& selection) const = 0;
  virtual ~NodeSelector() = default;

 protected:
  NodeSelector() = default;
};

struct SelectorAndAction {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<ONNX_NAMESPACE::OperatorSetVersion>>;

  // ctor so we can use make_unique to construct this class
  SelectorAndAction(const std::string& name_in,
                    const OpVersionsMap& ops_and_versions_in,
                    std::unique_ptr<NodeSelector> selector_in,
                    std::unique_ptr<Action> action_in)
      : name{name_in},
        ops_and_versions{ops_and_versions_in},
        selector{std::move(selector_in)},
        action{std::move(action_in)} {}

  const std::string name;
  OpVersionsMap ops_and_versions;
  std::unique_ptr<NodeSelector> selector;
  std::unique_ptr<Action> action;

  // can't copy/assign our unique_ptr members
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SelectorAndAction);
};
#endif

// class to manage a set of selector and associated actions in a full build,
// or just the set of actions in a minimal build.
class SelectorsAndActions {
 public:
  SelectorsAndActions() = default;

#if !defined(ORT_MINIMAL_BUILD)
  SelectorsAndActions(SelectorsAndActions&& rhs) noexcept
      : selectors_and_actions_map_{std::move(rhs.selectors_and_actions_map_)} {}

  // register a selector and action for the specified ops.
  // the name used in the registration is for matching the action when replaying the optimizations in a minimal build.
  // as it's stored in the ORT format model a shorter name is better. the name is scoped to this SelectorsAndActions
  // instance (which is scoped to a single SelectorActionTransformer instance).
  void RegisterSelectorAndAction(const std::string& name,
                                 const SelectorAndAction::OpVersionsMap& ops_and_versions_in,
                                 std::unique_ptr<NodeSelector> selector_in,
                                 std::unique_ptr<Action> action_in);

  const std::unordered_map<std::string, std::unique_ptr<SelectorAndAction>>& SelectorsAndActionsMap() const {
    return selectors_and_actions_map_;
  }

#else
  SelectorsAndActions(SelectorsAndActions&& rhs) noexcept
      : actions_map_{std::move(rhs.actions_map_)} {}

  void RegisterAction(const std::string& name, std::unique_ptr<Action> action);

  const std::unordered_map<std::string, std::unique_ptr<Action>>& ActionsMap() const {
    return actions_map_;
  }
#endif

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SelectorsAndActions);

 private:
#if !defined(ORT_MINIMAL_BUILD)
  std::unordered_map<std::string, std::unique_ptr<SelectorAndAction>> selectors_and_actions_map_;
#else
  std::unordered_map<std::string, std::unique_ptr<Action>> actions_map_;
#endif
};

/**
Class that implements graph transformation via a set of Selector+Action pairs. 
This setup allows optimizations to be captured and applied at runtime in a minimal build.
*/
class SelectorActionTransformer : public GraphTransformer {
 protected:
  // set `save` to find matching node groups and save them for later replay. if `save` is true the matching Action
  // for the Selector will not be called, so the nodes in the Graph will be preserved.
  SelectorActionTransformer(const std::string& name, SelectorsAndActions&& selectors_and_actions, bool save = false);

  // can't copy/assign selectors_and_actions_
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(SelectorActionTransformer);

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  SelectorsAndActions selectors_and_actions_;

#if !defined(ORT_MINIMAL_BUILD)
  Status MatchAndProcess(Graph& graph, Node& node, bool& modified, const logging::Logger& logger) const;

  std::unordered_map<std::string, const SelectorAndAction*> op_type_to_selector_and_action_;
  bool save_;  // save the node groups for use in runtime optimization in a minimal build with an ORT format model
#else
  // apply any saved optimizations
  Status ApplySaved(Graph& graph, bool& modified, const logging::Logger& logger) const;
#endif
};

}  // namespace onnxruntime
