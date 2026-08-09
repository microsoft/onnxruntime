// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fusion.h"
#include <torch/extension.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace onnxruntime {
namespace lazytensor {

struct OrtFuser {
  using FusionCallback = std::function<bool(OrtFuser*, torch::jit::Node*)>;

  torch::jit::Block* block_;
  torch::jit::AliasDb* aliasDb_;
  std::shared_ptr<torch::jit::Graph> graph_;
  FusionCallback callback_;
  torch::jit::Symbol kind_;
  bool strict_fuser_check_ = false;

  // nvrtc has a limit on the number of arguments allowed in a CUDA kernel.
  // The specific limit is a function of constant memory size, amount available
  // to pass arguments, and some implementation dependence. Select a safe
  // limit here.
  // This limit is also applied to other devices in the fuser by default.
  // Change with setInputArgLimit
  size_t subgraph_arg_limit_;

  // Custom passes require kind to specified
  OrtFuser(
      torch::jit::AliasDb* aliasDb,
      torch::jit::Block* block,
      FusionCallback callback,
      torch::jit::Symbol kind,
      bool strict_fuser_check = false,
      size_t subgraph_arg_limit = 128)
      : block_(block),
        aliasDb_(aliasDb),
        callback_(std::move(callback)),
        kind_(kind),
        strict_fuser_check_(strict_fuser_check),
        subgraph_arg_limit_(subgraph_arg_limit) {}

  torch::jit::value_list tensorInputs(torch::jit::Node* node) {
    return filter(node->inputs(), [](torch::jit::Value* v) {
      return v->type()->isSubtypeOf(*c10::TensorType::get());
    });
  }

  bool isFusable(torch::jit::Node* node) {
    return callback_(this, node);
  }

  bool calculatesSize(torch::jit::Node* node) {
    return node->matches("aten::size(Tensor self) -> int[]");
  }

  bool allUsersAreThisConsumerOrCalcSizes(torch::jit::Node* consumer, torch::jit::Value* producer) {
    auto defining_node = producer->node();
    for (auto o : defining_node->outputs()) {
      for (auto u : o->uses()) {
        if (u.user != consumer && !calculatesSize(u.user))
          return false;
      }
    }
    return true;
  }

  torch::jit::Graph& getSubgraph(torch::jit::Node* n) {
    AT_ASSERT(n->kind() == kind_);
    return *n->g(torch::jit::attr::Subgraph);
  }

  void mergeFusionGroups(torch::jit::Node* consumer_group, torch::jit::Node* producer_group) {
    // Now we have two fusion groups!
    // Revert the fusion - place all inner nodes of producer back in the outer
    // graph.
    std::vector<torch::jit::Node*> temporary_nodes;
    auto producer_subgraph = &getSubgraph(producer_group);

    // Initialize a map of inner graph values to outer graph values
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> inner_to_outer;
    auto inner_inputs = producer_subgraph->inputs();
    auto outer_inputs = producer_group->inputs();
    for (const auto i : c10::irange(inner_inputs.size())) {
      inner_to_outer[inner_inputs[i]] = outer_inputs[i];
    }

    // Clone all nodes
    for (auto inner : producer_subgraph->nodes()) {
      torch::jit::Node* outer = block_->owningGraph()->createClone(
          inner, [&](torch::jit::Value* k) -> torch::jit::Value* { return inner_to_outer.at(k); });
      outer->insertBefore(producer_group);
      temporary_nodes.emplace_back(outer);
      auto inner_outputs = inner->outputs();
      auto outer_outputs = outer->outputs();
      for (const auto i : c10::irange(inner_outputs.size())) {
        inner_to_outer[inner_outputs[i]] = outer_outputs[i];
      }
    }

    // Replace uses of producer_group outputs and destroy the producer
    auto subgraph_outputs = producer_subgraph->outputs();
    for (const auto i : c10::irange(subgraph_outputs.size())) {
      auto outer_output = inner_to_outer.at(subgraph_outputs[i]);
      producer_group->outputs()[i]->replaceAllUsesWith(outer_output);
      // new producer outputs have same aliasing properties as outer_output
      aliasDb_->replaceWithNewValue(producer_group->outputs()[i], outer_output);
    }
    producer_group->destroy();
    producer_group =
        nullptr;  // Just to get a clear error in case someone uses it

    // Inline the temporary nodes into the first group
    auto consumer_subgraph = &getSubgraph(consumer_group);
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
         ++it) {
      torch::jit::Node* node = *it;
      torch::jit::Node* merged = mergeNodeIntoGroup(consumer_group, node);
      // If any of the outputs are still used then we need to add them
      auto outputs = node->outputs();
      for (const auto i : c10::irange(outputs.size())) {
        auto output = outputs[i];
        if (output->uses().size() == 0)
          continue;
        consumer_subgraph->registerOutput(merged->outputs()[i]);
        auto new_output = consumer_group->addOutput();
        new_output->setType(output->type());
        output->replaceAllUsesWith(new_output);
        aliasDb_->replaceWithNewValue(output, new_output);
      }
      node->destroy();
    }
  }

  // insert a producer node into a consuming fusion group.
  // DOES NOT WORK if n is a consumer of an output of the fusion group
  // returns the node _inside_ the group that represents the node
  torch::jit::Node* mergeNodeIntoGroup(torch::jit::Node* group, torch::jit::Node* n) {
    AT_ASSERT(n->kind() != kind_);
    auto& subgraph = getSubgraph(group);
    // map from nodes in the surrounding graph to parameters in the fusion
    // group's subgraph that correspond to them
    std::unordered_map<torch::jit::Value*, torch::jit::Value*> inputs_map;
    AT_ASSERT(group->inputs().size() == subgraph.inputs().size());
    for (size_t i = 0; i < group->inputs().size(); ++i) {
      // outer scope input -> inner scope (inside subgraph) input
      inputs_map[group->inputs().at(i)] = subgraph.inputs().at(i);
    }
    // add n's inputs to the fusion group's input list if we don't already have
    // them
    // we insert tensors first because the fuser assumes that to be the case
    // (as a legacy from tensors only)
    torch::jit::WithInsertPoint guard(*subgraph.nodes().begin());
    for (auto input : n->inputs()) {
      if (inputs_map.count(input) == 0) {
        if (input->type()->isSubtypeOf(*c10::TensorType::get())) {
          group->addInput(input);
          // Add the corresponding input to subgraph's input list.
          auto inner_input = subgraph.addInput();
          inner_input->setType(input->type());
          // Update outer-to-inner value mapping.
          inputs_map[input] = inner_input;
        } else if (
            (input->type()->isSubtypeOf(*c10::FloatType::get()) &&
             input->node()->kind() != torch::jit::prim::Constant) ||
            (n->kind() == torch::jit::aten::_grad_sum_to_size &&
             input->type()->isSubtypeOf(*c10::ListType::ofInts()))) {
          group->addInput(input);
          auto inner_input = subgraph.addInput();
          inner_input->setType(input->type());
          inputs_map[input] = inner_input;
        } else if (
            input->type()->isSubtypeOf(*c10::IntType::get()) &&
            input->node()->kind() != torch::jit::prim::Constant) {
          group->addInput(input);
          auto inner_input = subgraph.addInput();
          inner_input->setType(input->type());
          inputs_map[input] = inner_input;
        } else {
          // We don't support passing in scalars as arguments to fused kernels,
          // so we generally don't allow fusing tensor-scalar operations unless
          // the scalar is constant. In those cases we inline the constants
          // directly in the body of the fused group.
          AT_ASSERT(input->node()->kind() == torch::jit::prim::Constant);
          torch::jit::Node* in_const =
              subgraph.createClone(input->node(), [](torch::jit::Value*) -> torch::jit::Value* {
                throw std::runtime_error("unexpected input");
              });
          subgraph.insertNode(in_const);
          inputs_map[input] = in_const->output();
        }
      }
    }
    // copy n into the graph, remapping its inputs to internal nodes
    torch::jit::Node* n_in_graph = subgraph.createClone(
        n, [&](torch::jit::Value* k) -> torch::jit::Value* { return inputs_map[k]; });
    // if n's outputs are already inputs to the fusion group,
    // we need to remove them because n is now inside the fusion group.
    //
    // i.e.,
    // x = f(w); group(x, y, z) becomes group(w, y, z).
    // x, y, z = f(w); group(x, y, z) becomes group(w).
    //
    // remapping nodes that used the input to the newly-merged node
    // n is not an input when the fusion group is empty
    auto inputs = group->inputs();
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      auto it = std::find(inputs.begin(), inputs.end(), n->outputs()[i]);
      if (it != inputs.end()) {
        size_t p = it - inputs.begin();
        group->removeInput(p);
        subgraph.inputs()[p]->replaceAllUsesWith(n_in_graph->outputs()[i]);
        subgraph.eraseInput(p);
      }
    }
    return subgraph.insertNode(n_in_graph);
  }

  // turn consumer node n into a fusion group with just n inside
  // to prepare for fusion and replace uses of n with the new group
  torch::jit::Node* createSingletonFusionGroup(torch::jit::Node* n) {
    auto group = block_->owningGraph()->createWithSubgraph(kind_);
    // propagate position information for the new node so we can always
    // have a valid mapping
    group->insertBefore(n);
    torch::jit::Node* mergedNode = mergeNodeIntoGroup(group, n);
    // Now n's outputs should be generated by the new node (aka mergedNode)
    // in the fusion group. Let's connect mergedNode to the outer graph.
    for (size_t i = 0; i < mergedNode->outputs().size(); ++i) {
      // Connect the i-th inner output to outer graph.
      getSubgraph(group).registerOutput(mergedNode->output(i));
      auto new_outer_output = group->addOutput();
      // Copy metadata from old outer output to new outer output.
      new_outer_output->copyMetadata(n->output(i));
      aliasDb_->replaceWithNewValue(n->output(i), new_outer_output);
    }
    // Now group is a single-op subgraph containing the clone of n.
    AT_ASSERT(n->outputs().size() == group->outputs().size());
    n->replaceAllUsesWith(group);
    n->destroy();
    return group;
  }

  at::optional<torch::jit::Node*> tryFuse(torch::jit::Node* consumer, torch::jit::Value* producer) {
    // this handles cases where producer can be moved _into_ the fusion group of
    // consumer.
    // TODO: extend to fusion of consumer into _producer's_ fusion blob
    // if the consumer allInputsAreThisProducer(consumer,producer)
    // we can move the consumer up into the producer.
    // but this requires better handling of merging fusion groups so it is not
    // done now
    bool shouldFuse = isFusable(producer->node()) &&
                      // Rearrange nodes such that all uses of producer are after the
                      // consumer. Fusion will rewrite those later uses to use the version of
                      // producer generated by the fused blob. In this case, producer becomes
                      // an output of the fusion group.
                      aliasDb_->moveBeforeTopologicallyValid(producer->node(), consumer);

    if (!shouldFuse) {
      return at::nullopt;
    }

    if ((consumer->inputs().size() + consumer->outputs().size() +
         producer->node()->inputs().size() +
         producer->node()->outputs().size()) > subgraph_arg_limit_) {
      return at::nullopt;
    }

    auto group = consumer;
    if (consumer->kind() != kind_) {
      group = createSingletonFusionGroup(consumer);
    }

    if (producer->node()->kind() == kind_) {
      mergeFusionGroups(group, producer->node());
      return group;
    }
    // AT_ASSERT(producer->node()->outputs().size() == 1);
    torch::jit::Node* merged = mergeNodeIntoGroup(group, producer->node());
    // remaining uses of this producer can occur because we allow
    // fusion in cases where uses remain after the consumer
    // if these exist, re-route them to the version of producer
    // created in FusionGroup
    size_t i = -1;
    for (auto output : producer->node()->outputs()) {
      ++i;
      if (output->uses().size() == 0) {
        continue;
      }
      getSubgraph(group).registerOutput(merged->outputs()[i]);
      torch::jit::Value* new_output = group->addOutput();
      new_output->copyMetadata(new_output);
      aliasDb_->replaceWithNewValue(output, new_output);
      output->replaceAllUsesWith(new_output);
    }
    producer->node()->destroy();
    return group;
  }

  torch::jit::value_list sortReverseTopological(torch::jit::ArrayRef<torch::jit::Value*> inputs) {
    torch::jit::value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == block_) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](torch::jit::Value* a, torch::jit::Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  // returns where to continue scanning, and whether any fusion was made
  std::pair<torch::jit::graph_node_list::iterator, bool> scanNode(torch::jit::Node* consumer) {
    if (isFusable(consumer)) {
      // handle inputs in reverse topological order as well...
      // otherwise in f(a,a+b) it will appear a is used twice if we consider
      // the f-a fusion before the f-(a+b) fusion first.
      auto inputs = sortReverseTopological(consumer->inputs());
      for (auto producer : inputs) {
        auto fusion_group = tryFuse(consumer, producer);
        if (fusion_group) {
          // after fusion, consumer moves into a FusionGroup, so inputs is no
          // longer valid so we rescan the new FusionGroup for more fusions...
          return std::make_pair(fusion_group.value()->reverseIterator(), true);
        }
      }
    }
    return std::make_pair(++consumer->reverseIterator(), false);
  }

  void optimizeFusedGraphs() {
    for (torch::jit::Node* node : block_->nodes()) {
      if (node->kind() != torch::jit::prim::FusionGroup) {
        continue;
      }
      auto subgraph = node->g(torch::jit::attr::Subgraph);
      EliminateDeadCode(subgraph);
      EliminateCommonSubexpression(subgraph);
      ConstantPooling(subgraph);
    }
  }

  void run() {
    // Run the pass until no changes are made.
    // This is necessary, because the algorithm can miss out on certain fusion
    // opportunities if ran only once. Consider this graph:
    //
    // %1 = f(...)
    // %2 = g(%1)
    // %3 = h(%1)
    // %4 = l(%3)
    // return (%4, %2)
    //
    // where f, g, h, l are simple map ops.
    // The first iteration will fuse %4 and %3, and see that %1 is an input, but
    // can't be fused, because it has a different use before the fusion group
    // in our topological ordering. Then, %2 will be considered, and fused with
    // %1. If we do another iteration, the algorithm will consider the fusion of
    // these two groups and fix the situation.
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    optimizeFusedGraphs();

    for (torch::jit::Node* node : block_->nodes()) {
      for (torch::jit::Block* sub_block : node->blocks()) {
        OrtFuser(aliasDb_, sub_block, callback_, kind_, strict_fuser_check_)
            .run();
      }
    }
  }
};

void OrtFuseGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    const std::function<bool(torch::jit::Node*)>& fn,
    torch::jit::Symbol kind,
    size_t arg_limit) {
  torch::jit::AliasDb db(graph);
  auto g = OrtFuser(
      &db,
      graph->block(),
      [=](OrtFuser* gf, torch::jit::Node* n) { return fn(n) || n->kind() == kind; },
      kind, false, arg_limit);

  g.run();
  torch::jit::Lint(&db);
}

}  // namespace lazytensor
}  // namespace onnxruntime
