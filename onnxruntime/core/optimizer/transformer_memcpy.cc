// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transformer_memcpy.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/utils.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(onnxruntime::Graph& graph, const std::string& provider, const GraphInitializersLocationInfo& graph_initializers_location_info)
      : graph_(graph), provider_(provider), graph_initializers_location_info_(graph_initializers_location_info) {}

  bool ModifyGraph(const KernelRegistryManager& schema_registries);

 private:
  void ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries, InitializedTensorSet& initializers_consumed);
  void BuildDefsMapping(const onnxruntime::NodeArg* arg, const KernelRegistryManager& kernel_registries);
  void AddCopyNode(onnxruntime::NodeArg* arg, bool is_input);
  bool ProcessInitializers(const KernelRegistryManager& kernel_registries, const InitializedTensorSet& initializers_consumed);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransformerMemcpyImpl);

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
  const GraphInitializersLocationInfo graph_initializers_location_info_;

  onnxruntime::Graph& graph_;
  std::string provider_;
};

/** Helper that returns a pointer to the corresponding TensorProto for a name if it is an initializer.
@param check_outer_scope If true and the graph is a subgraph, check parent graph/s for 'name' if not found in 'graph'.
*/
static const onnx::TensorProto* GetInitializer(const Graph& graph, const std::string& name, bool check_outer_scope) {
  const onnx::TensorProto* initializer = nullptr;
  if (graph.GetInitializedTensor(name, initializer)) {
    return initializer;
  } else if (check_outer_scope && graph.IsSubgraph()) {
    return GetInitializer(*graph.ParentGraph(), name, check_outer_scope);
  }
  return initializer;
}

void GraphInitializersLocationInfo::AccumulateInitializerLocationsInSubgraphs(Graph& graph,
                                                                              const InitializedTensorSet& initializers,
                                                                              const KernelRegistryManager& kernel_registries,
                                                                              /*out*/ std::unordered_map<std::string, std::unordered_set<int>>& initializer_to_location_map) const {
  for (auto& node : graph.Nodes()) {
    int index = 0;
    for (auto input : node.InputDefs()) {
      if (input) {
        const auto& def_name = input->Name();
        if (initializers.count(def_name) > 0) {
          auto map_iter = initializer_to_location_map.find(def_name);

          if (map_iter == initializer_to_location_map.end()) {
            initializer_to_location_map.insert({def_name, {}});
          }

          const KernelCreateInfo* kci = nullptr;
          kernel_registries.SearchKernelRegistry(node, &kci);

          if (node.GetExecutionProviderType() == kCpuExecutionProvider || utils::IsInputOnCpu(node, kci, index)) {
            initializer_to_location_map[def_name].insert(0);  // 0 for non-providers
          } else {
            initializer_to_location_map[def_name].insert(1);  // 1 for providers
          }
        }
      }

      ++index;
    }

    // Recurse into subgraphs
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& subgraph = *entry.second;
      AccumulateInitializerLocationsInSubgraphs(subgraph, initializers,
                                                kernel_registries, initializer_to_location_map);
    }
  }
}

void GraphInitializersLocationInfo::MakeProviderInitializerDuplicates(Graph& graph, const KernelRegistryManager& kernel_registries) const {
  const auto& graph_initializers = graph.GetAllInitializedTensors();
  // TODO: Support shadow initializers
  // It is uncommon that we see shadow initializers (i.e.) an initializer re-defined in a subgraph
  // while the outer level graph(s) have different initializers defined for the same name.
  // To support that adds some amount of complexity, so we will support it if we see such a model.

  for (auto& pair : graph_initializers) {
    // We have already added an initializer with the same name while processing one of the outer graphs,
    // so break if we see shadow initializers
    if ((provider_initializer_names_.count(pair.first) > 0) || (non_provider_initializer_names_.count(pair.first) > 0)) {
      ORT_THROW(
          "Shadow initializers are currently unsupported in ORT. "
          "Consider using a new initializer (different name) for the shadow initializer");
    }
  }

  std::unordered_map<std::string, std::unordered_set<int>> initializer_to_location_map;
  AccumulateInitializerLocationsInSubgraphs(graph, graph_initializers,
                                            kernel_registries, initializer_to_location_map);

  for (auto& pair : initializer_to_location_map) {
    const auto& def_name = pair.first;
    const auto& usage_count_set = pair.second;

    bool is_used_by_non_provider_nodes = usage_count_set.count(0) > 0;

    bool is_used_by_provider_nodes = usage_count_set.count(1) > 0;

    // We make a copy of the initializer that is to be consumed by the provider Node so that
    // session state initializer can copy it over to the provider device during its operation
    // NOTE: We only need to make a copy if the initializer is consumed by both provider and
    // non-provider nodes, not if they are only consumed by one group of nodes
    if (is_used_by_non_provider_nodes && is_used_by_provider_nodes) {
      std::string dupe_name = graph.GenerateNodeArgName(pair.first);

      // No need to check outer scope
      // we are only processing initializers introduced in the current level of the graph
      const TensorProto* tensor_proto = GetInitializer(graph, def_name, false);
      ORT_ENFORCE(tensor_proto);  // Sanity check
      TensorProto new_tensor_proto = *tensor_proto;
      *(new_tensor_proto.mutable_name()) = dupe_name;
      graph.AddInitializedTensor(new_tensor_proto);

      non_provider_initializer_names_.insert(def_name);  // Original initializer for non-provider nodes
      provider_initializer_names_.insert(dupe_name);     // New initializer for provider nodes

      non_provider_initializer_names_to_provider_dupe_initializer_names_.insert({def_name, dupe_name});
    } else if (is_used_by_non_provider_nodes) {
      non_provider_initializer_names_.insert(def_name);
    } else {
      provider_initializer_names_.insert(def_name);
    }
  }
}

// very simple GraphTransformer that uses TransformerMemcpyImpl for each graph
// and mainly provides the subgraph recursion functionality
common::Status MemcpyTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  for (auto& provider : provider_types_) {
    if (!utils::ProviderIsCpuBased(provider)) {
      // Accumulate initializers' location info for all subgraphs
      graph_initializers_location_info_.MakeProviderInitializerDuplicates(graph, registry_manager_);
      TransformerMemcpyImpl copy_impl(graph, provider, graph_initializers_location_info_);
      auto current_modified = copy_impl.ModifyGraph(registry_manager_);
      modified = modified || current_modified;
      break;
    }
  }

  // TODO: We probably need to do the recursion inline when processing the main graph in order to maximize efficiency.
  // e.g. data on GPU prior to an 'If' node. The 'If' must run on CPU, but if the subgraph is GPU based it could
  // consume the data from GPU and we shouldn't insert a memcpy from GPU to CPU prior to the If node, and one from
  // CPU back to GPU when beginning execution of the subgraph. To do that requires inspecting the subgraph (and any
  // nested subgraphs) when deciding whether to insert a memcpy in the parent graph, and may need to be fully done
  // within TransformerMemcpyImpl instead of via this simple GraphTransformer.

  // handle any subgraphs in nodes
  for (auto& node : graph.Nodes()) {
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  return Status::OK();
}

/*

Overview: The transformer transforms the input graph as follows:

(1) For every initializer W that is referenced by both provider and non-provider nodes,
we create a duplicate initializer W2 and change all provider nodes to reference this
duplicate copy.

(2) For every ml-value X that is computed by a provider node and referenced by a
non-provider node, we introduce a new ml-value X2. We replace all references to X
in provider nodes by X2, and introduce a copy from X2 to X. (All graph outputs
are considered as non-provider references here.)

(3 For every ml-value X that is computed by a non-provider node and referenced by
a provider node, we introduce a new ml-value X2. We replace all references to X in
provider nodes by X2, and introduce a copy from X to X2. (All graph inputs are
considered to be non-provider here.)

Note that every ml-value is computed at a unique point (either provider or non-provider),
but it may be referenced and used at multiple points (by both provider and non-provider).

This transformer does not currently optimize copies between, e.g., two different GPU devices, etc.

*/

bool TransformerMemcpyImpl::ModifyGraph(const KernelRegistryManager& kernel_registries) {
  bool modified = false;
  InitializedTensorSet initializers_consumed;
  // find defs that require copy
  for (auto& node : graph_.Nodes()) {
    //as we process the defs, collect all the initializers consumed at the current graph level
    ProcessDefs(node, kernel_registries, initializers_consumed);
  }

  // for initializers shared by different providers, create dups
  if (ProcessInitializers(kernel_registries, initializers_consumed))
    modified = true;

  for (auto arg : graph_.GetInputs())
    BuildDefsMapping(arg, kernel_registries);

  for (auto arg : non_provider_input_defs_)
    BuildDefsMapping(arg, kernel_registries);

  for (auto arg : non_provider_output_defs_)
    BuildDefsMapping(arg, kernel_registries);

  for (auto arg : graph_.GetInputs())
    // For inputs we need to create a copy node only when the input is connected to both provider
    // and non-provider nodes. Otherwise utils::CopyInputsAcrossDevices() will do the job.
    if (provider_input_defs_.count(arg) && non_provider_input_defs_.count(arg)) {
      AddCopyNode(const_cast<onnxruntime::NodeArg*>(arg), true);
      modified = true;
    }

  for (auto arg : non_provider_output_defs_)
    if (provider_input_defs_.count(arg)) {
      AddCopyNode(arg, true);
      modified = true;
    }

  for (auto arg : provider_output_defs_)
    if (non_provider_input_defs_.count(arg)) {
      AddCopyNode(arg, false);
      modified = true;
    }

  return modified;
}

void TransformerMemcpyImpl::ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries,
                                        InitializedTensorSet& initializers_consumed) {
  auto node_provider_type = node.GetExecutionProviderType();
  if ((node_provider_type == provider_) ||
      (node_provider_type == kCudaExecutionProvider && kTensorrtExecutionProvider == provider_)) {
    provider_nodes_.insert(&node);
    // note KernelCreateInfo might be nullptr for custom kernel
    const KernelCreateInfo* kci = nullptr;
    kernel_registries.SearchKernelRegistry(node, &kci);

    bool is_implicit_input = false;
    auto process_inputs =
        [this, &node, &kci, &initializers_consumed, &is_implicit_input](const onnxruntime::NodeArg& arg, size_t index) {
          // check if this NodeArg is an initializer defined in current outer graph level

          const auto* initializer_tensor_proto = GetInitializer(graph_, arg.Name(), true);
          if (initializer_tensor_proto != nullptr) {
            initializers_consumed[arg.Name()] = initializer_tensor_proto;
          }

          // implicit inputs have no location info in the kernel def, so do nothing to them here, leaving the control
          // flow op (Loop, Scan, If) to do the necessary copy if the input crosses different provider.
          // PlannerImpl::ComputeUseCounts has matching logic so the allocation plan does the same thing
          if (!is_implicit_input) {
            if (utils::IsInputOnCpu(node, kci, index)) {
              non_provider_input_defs_.insert(&arg);
            } else {
              provider_input_defs_.insert(&arg);
            }
          }

          return Status::OK();
        };

    auto status = onnxruntime::Node::ForEachWithIndex(node.InputDefs(), process_inputs);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());

    is_implicit_input = true;
    status = onnxruntime::Node::ForEachWithIndex(node.ImplicitInputDefs(), process_inputs);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());

    auto& output_defs = node.MutableOutputDefs();
    for (size_t i = 0; i < output_defs.size(); ++i) {
      auto arg = output_defs[i];
      if (!arg->Exists())
        continue;

      if (kci && kci->kernel_def->IsOutputOnCpu(i))
        non_provider_output_defs_.insert(arg);
      else
        provider_output_defs_.insert(arg);
    }
  } else if (node_provider_type != kCudaExecutionProvider && node_provider_type != kTensorrtExecutionProvider) {
    // TODO: copy between devices? i.e. multiple GPUs
    if (node_provider_type != onnxruntime::kCpuExecutionProvider &&
        node_provider_type != onnxruntime::kVitisAIExecutionProvider &&
        !node_provider_type.empty()) {
      ORT_THROW("Execution type '", node_provider_type, "' doesn't support memcpy ");
    }

    for (const auto* arg : node.InputDefs()) {
      if (arg->Exists())
        non_provider_input_defs_.insert(arg);
    }

    for (const auto* arg : node.ImplicitInputDefs()) {
      if (arg->Exists())
        non_provider_input_defs_.insert(arg);
    }

    for (auto* arg : node.MutableOutputDefs()) {
      if (arg->Exists())
        non_provider_output_defs_.insert(arg);
    }
  }
}

//for non_provider defs, collect the nodes that expect it is provider tensor as input/output.
void TransformerMemcpyImpl::BuildDefsMapping(const onnxruntime::NodeArg* arg, const KernelRegistryManager& kernel_registries) {
  for (auto& it : graph_.Nodes()) {
    if (it.OpType() == "MemcpyFromHost" || it.OpType() == "MemcpyToHost") continue;
    auto input_it =
        std::find(it.MutableInputDefs().begin(), it.MutableInputDefs().end(), const_cast<onnxruntime::NodeArg*>(arg));
    auto output_it =
        std::find(it.MutableOutputDefs().begin(), it.MutableOutputDefs().end(), const_cast<onnxruntime::NodeArg*>(arg));
    int arg_input_index =
        input_it != it.MutableInputDefs().end() ? static_cast<int>(input_it - it.MutableInputDefs().begin()) : -1;
    int arg_output_index =
        output_it != it.MutableOutputDefs().end() ? static_cast<int>(output_it - it.MutableOutputDefs().begin()) : -1;
    if (arg_input_index == -1 && arg_output_index == -1)
      continue;
    auto node_provider_type = it.GetExecutionProviderType();
    if ((node_provider_type == provider_) || (node_provider_type == kCudaExecutionProvider && kTensorrtExecutionProvider == provider_)) {
      const KernelCreateInfo* kci = nullptr;
      kernel_registries.SearchKernelRegistry(it, &kci);
      if (arg_input_index != -1) {
        if (!kci || !utils::IsInputOnCpu(it, kci, arg_input_index)) provider_input_nodes_[arg].insert(&it);
      }
      if (arg_output_index != -1) {
        if (!kci || !kci->kernel_def->IsOutputOnCpu(arg_output_index)) provider_output_nodes_[arg].insert(&it);
      }
    }
  }
}

void TransformerMemcpyImpl::AddCopyNode(onnxruntime::NodeArg* arg, bool is_input) {
  // create unique name for new def
  std::string new_def_name = graph_.GenerateNodeArgName(arg->Name() + "_" + provider_);

  auto* new_arg = &graph_.GetOrCreateNodeArg(new_def_name, arg->TypeAsProto());
  auto* src_arg = is_input ? arg : new_arg;
  auto* dst_arg = is_input ? new_arg : arg;

  // create unique name for copy node
  std::string new_node_name = graph_.GenerateNodeName("Memcpy");

  const auto op_name = is_input ? "MemcpyFromHost" : "MemcpyToHost";
  auto& new_node = graph_.AddNode(new_node_name, op_name, "Copy from/to host memory",
                                  std::vector<onnxruntime::NodeArg*>{src_arg},
                                  std::vector<onnxruntime::NodeArg*>{dst_arg});
  new_node.SetExecutionProviderType(provider_);
  std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> map = {{arg, new_arg}};
  auto it = provider_input_nodes_.find(arg);
  if (it != provider_input_nodes_.end()) {
    for (auto* node : it->second)
      node->ReplaceDefs(map);
  }
  it = provider_output_nodes_.find(arg);
  if (it != provider_output_nodes_.end()) {
    for (auto* node : it->second)
      node->ReplaceDefs(map);
  }
}

template <typename NodeArgSetType>
static const onnxruntime::NodeArg* FindNodeArg(const NodeArgSetType& def_set, const std::string& name) {
  NodeArg def(name, nullptr);
  auto it = def_set.find(&def);  // this works since we use name to compare NodeArg
  if (it != def_set.end())
    return *it;
  return nullptr;
}

// We duplicate any initializer that is used by both provider nodes and non-provider nodes
// to ensure that provider nodes and non-provider nodes don't share initializers, as they
// need to stay in different memory locations.
bool TransformerMemcpyImpl::ProcessInitializers(const KernelRegistryManager& kernel_registries,
                                                const InitializedTensorSet& initializers_consumed) {
  std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacements;
  for (const auto& pair : initializers_consumed) {
    const auto& name = pair.first;
    if (name == "self.logits_processor.min_length") {
      float f = 1.2f;
      ORT_IGNORE_RETURN_VALUE(f);
    }
    const onnxruntime::NodeArg* provider_def = FindNodeArg(provider_input_defs_, name);
    const onnxruntime::NodeArg* non_provider_def = FindNodeArg(non_provider_input_defs_, name);
    if (provider_def != nullptr && non_provider_def != nullptr) {
      ORT_ENFORCE(graph_initializers_location_info_.non_provider_initializer_names_.find(name) !=
                      graph_initializers_location_info_.non_provider_initializer_names_.end(),
                  name);

      ORT_ENFORCE(graph_initializers_location_info_.non_provider_initializer_names_to_provider_dupe_initializer_names_.find(name) !=
                  graph_initializers_location_info_.non_provider_initializer_names_to_provider_dupe_initializer_names_.end());

      // Use the duplicate def name we have created just for this scenario
      const auto& new_def_name = graph_initializers_location_info_.non_provider_initializer_names_to_provider_dupe_initializer_names_[name];

      auto& new_def = graph_.GetOrCreateNodeArg(new_def_name, provider_def->TypeAsProto());
      replacements.insert(std::make_pair(provider_def, &new_def));
    } else if (provider_def != nullptr) {
      // This def is consumed by a provider node
      // If this initializer is ONLY consumed by provider nodes, then we don't need to use
      // the duplicate initializer. Look into the provider_initializer_names_ map to know
      // if this is the case.
      // Else, use the duplicate initializer
      auto iter = graph_initializers_location_info_.provider_initializer_names_.find(name);

      if (iter == graph_initializers_location_info_.provider_initializer_names_.end()) {
        // Use the duplicate def name we have created just for this scenario
        const auto& new_def_name = graph_initializers_location_info_.non_provider_initializer_names_to_provider_dupe_initializer_names_[name];
        auto& new_def = graph_.GetOrCreateNodeArg(new_def_name, provider_def->TypeAsProto());
        replacements.insert(std::make_pair(provider_def, &new_def));
      }
    }
    // For non_provider defs, there is no action - stick to using the original def name
  }

  for (auto p_node : provider_nodes_) {
    // make a copy of replacement map as the node may exclude mapping for InputDefs with MemTypeOnCpuExplicitly
    auto dup_replacements = replacements;

    const KernelCreateInfo* kci = nullptr;
    auto status = kernel_registries.SearchKernelRegistry(*p_node, &kci);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    if (kci == nullptr) continue;
    if (kci->kernel_def == nullptr) continue;
    onnxruntime::Node::ForEachWithIndex(p_node->InputDefs(),
                                        [kci, &p_node, &dup_replacements](const onnxruntime::NodeArg& arg, size_t index) {
                                          if (utils::IsInputOnCpu(*p_node, kci, index)) dup_replacements.erase(&arg);
                                          return Status::OK();
                                        });

    // normally initializers are only inputs, but things may change with ops like assign
    onnxruntime::Node::ForEachWithIndex(p_node->OutputDefs(),
                                        [kci, &dup_replacements](const onnxruntime::NodeArg& arg, size_t index) {
                                          if (kci->kernel_def->IsOutputOnCpu(index)) {
                                            ORT_ENFORCE(dup_replacements.find(&arg) == dup_replacements.end());
                                          }
                                          return Status::OK();
                                        });

    p_node->ReplaceDefs(dup_replacements);
  }

  // This denotes a modification to the graph
  return !replacements.empty();
}

}  // namespace onnxruntime
