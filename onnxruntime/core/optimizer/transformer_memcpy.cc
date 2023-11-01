// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transformer_memcpy.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/utils.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

// implements MemCpy node insertion in graph transform
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTranformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(onnxruntime::Graph& graph, const std::string& provider)
      : graph_(graph), provider_(provider) {}

  bool ModifyGraph(const KernelRegistryManager& schema_registries, const logging::Logger& logger, int& copy_node_counter);

 private:
  void ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries, InitializedTensorSet& initializers_consumed);
  void BuildDefsMapping(const onnxruntime::NodeArg* arg, const KernelRegistryManager& kernel_registries);
  void AddCopyNode(onnxruntime::NodeArg* arg, bool is_input, const logging::Logger& logger);
  bool ProcessInitializers(const KernelRegistryManager& kernel_registries, const InitializedTensorSet& initializers_consumed);

 private:
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

// very simple GraphTransformer that uses TransformerMemcpyImpl for each graph
// and mainly provides the subgraph recursion functionality
common::Status MemcpyTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  for (auto& provider : provider_types_) {
    if (!utils::ProviderIsCpuBased(provider)) {
      TransformerMemcpyImpl copy_impl(graph, provider);

      int copy_node_counter = 0;
      auto current_modified = copy_impl.ModifyGraph(registry_manager_, logger, copy_node_counter);
      if (copy_node_counter > 0 && provider == kCudaExecutionProvider) {
        LOGS(logger, WARNING) << copy_node_counter << " Memcpy nodes are added to the graph " << graph.Name()
                              << " for " << provider
                              << ". It might have negative impact on performance (including unable to run CUDA graph). "
                              << "Set session_options.log_severity_level=1 to see the detail logs before this message.";
      }

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

bool TransformerMemcpyImpl::ModifyGraph(const KernelRegistryManager& kernel_registries,
                                        const logging::Logger& logger,
                                        int& copy_node_counter) {
  bool modified = false;
  InitializedTensorSet initializers_consumed;
  // find defs that require copy
  for (auto& node : graph_.Nodes()) {
    // as we process the defs, collect all the initializers consumed at the current graph level
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
      AddCopyNode(const_cast<onnxruntime::NodeArg*>(arg), true, logger);
      copy_node_counter++;
      modified = true;
    }

  for (auto arg : non_provider_output_defs_)
    if (provider_input_defs_.count(arg)) {
      AddCopyNode(arg, true, logger);
      copy_node_counter++;
      modified = true;
    }

  for (auto arg : provider_output_defs_)
    if (non_provider_input_defs_.count(arg)) {
      AddCopyNode(arg, false, logger);
      copy_node_counter++;
      modified = true;
    }

  // Process implicit inputs in subgraphs that is explicitly consumed
  // on both provider and non-provider nodes. This is mimicking
  // logic for explicit graph inputs.
  if (graph_.IsSubgraph()) {
    for (auto arg : graph_.ParentNode()->ImplicitInputDefs()) {
      // Looking into `provider_input_defs_` and `non_provider_input_defs_`
      // using NodeArg pointers from the outer scope is okay because the
      // comparator is only name based (and doesn't compare raw pointers)
      if (provider_input_defs_.count(arg) && non_provider_input_defs_.count(arg)) {
        // There should be at-least one explicit consumer of the NodeArg
        // in both the provider node list and the non-provider node list.
        // If there are no explicit consumers in both lists, we don't want
        // to get into the business of adding copy nodes at this
        // level.
        // If there are explicit consumers in only one list (either provider
        // or non-provider node consumers), there isn't any point in adding
        // copy nodes in that case either as subgraph copy logic will take
        // it to the required device (i.e.) we don't need to care about it here.

        // Be sure to use the NodeArg* relevant to the current graph level
        // (the name will be the same as the parent node's implicit input)
        const auto* node_arg_in_current_graph_level = *provider_input_defs_.find(arg);

        AddCopyNode(const_cast<onnxruntime::NodeArg*>(node_arg_in_current_graph_level), true, logger);
        copy_node_counter++;
        modified = true;
      }
    }
  }

  return modified;
}

void TransformerMemcpyImpl::ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries,
                                        InitializedTensorSet& initializers_consumed) {
  auto node_provider_type = node.GetExecutionProviderType();
  if ((node_provider_type == provider_) ||
      (node_provider_type == kCudaExecutionProvider && kTensorrtExecutionProvider == provider_) ||
      (node_provider_type == kRocmExecutionProvider && kMIGraphXExecutionProvider == provider_)) {
    provider_nodes_.insert(&node);
    // note KernelCreateInfo might be nullptr for custom kernel
    const KernelCreateInfo* kci = nullptr;
    ORT_IGNORE_RETURN_VALUE(kernel_registries.SearchKernelRegistry(node, &kci));

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

      if (utils::IsOutputOnCpu(node, kci, i))
        non_provider_output_defs_.insert(arg);
      else
        provider_output_defs_.insert(arg);
    }
  } else if (node_provider_type != kCudaExecutionProvider && node_provider_type != kTensorrtExecutionProvider &&
             node_provider_type != kRocmExecutionProvider && node_provider_type != kMIGraphXExecutionProvider) {
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

    // Never add an implicit def to provider_input_defs_ or non_provider_input_defs_.
    // This is because we don't want to add copy nodes on account of implicit
    // inputs to nodes.
    // We will rely on utils::CopyInputsAcrossDevices() to do the job.
    // for (const auto* arg : node.ImplicitInputDefs()) {
    //  if (arg->Exists())
    //    non_provider_input_defs_.insert(arg);
    //}

    for (auto* arg : node.MutableOutputDefs()) {
      if (arg->Exists())
        non_provider_output_defs_.insert(arg);
    }
  }
}

// for non_provider defs, collect the nodes that expect it is provider tensor as input/output.
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
    if ((node_provider_type == provider_) ||
        (node_provider_type == kCudaExecutionProvider && kTensorrtExecutionProvider == provider_) ||
        (node_provider_type == kRocmExecutionProvider && kMIGraphXExecutionProvider == provider_)) {
      const KernelCreateInfo* kci = nullptr;
      ORT_IGNORE_RETURN_VALUE(kernel_registries.SearchKernelRegistry(it, &kci));
      if (arg_input_index != -1) {
        if (!kci || !utils::IsInputOnCpu(it, kci, arg_input_index)) provider_input_nodes_[arg].insert(&it);
      }
      if (arg_output_index != -1) {
        if (!kci || !utils::IsOutputOnCpu(it, kci, arg_output_index)) provider_output_nodes_[arg].insert(&it);
      }
    }
  }
}

void TransformerMemcpyImpl::AddCopyNode(onnxruntime::NodeArg* arg, bool is_input, const logging::Logger& logger) {
  // create unique name for new def
  std::string new_def_name = graph_.GenerateNodeArgName(arg->Name() + "_" + provider_);

  auto* new_arg = &graph_.GetOrCreateNodeArg(new_def_name, arg->TypeAsProto());
  auto* src_arg = is_input ? arg : new_arg;
  auto* dst_arg = is_input ? new_arg : arg;

  // create unique name for copy node
  std::string new_node_name = graph_.GenerateNodeName("Memcpy");

  const auto op_name = is_input ? "MemcpyFromHost" : "MemcpyToHost";
  LOGS(logger, INFO) << "Add " << op_name << (is_input ? " after " : " before ") << arg->Name()
                     << " for " << provider_;

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
bool TransformerMemcpyImpl::ProcessInitializers(const KernelRegistryManager& kernel_registries, const InitializedTensorSet& initializers_consumed) {
  std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacements;
  for (const auto& pair : initializers_consumed) {
    const auto& name = pair.first;
    const onnxruntime::NodeArg* provider_def = FindNodeArg(provider_input_defs_, name);
    const onnxruntime::NodeArg* non_provider_def = FindNodeArg(non_provider_input_defs_, name);
    if (provider_def != nullptr && non_provider_def != nullptr) {
      std::string new_def_name = graph_.GenerateNodeArgName(name);
      auto& new_def = graph_.GetOrCreateNodeArg(new_def_name, provider_def->TypeAsProto());

      // We make a copy of the initializer that is to be consumed by the provider Node so that
      // session state initializer can copy it over to the provider device during its operation
      // TODO: The copy being made is possibly redundant if this occurs in a subgraph
      // When multiple subgraphs consume the same initializer as an implicit input,
      // multiple copies of the initializer will be made into the provider device
      // This should not directly affect runtime performance as the copies occur during initialization
      // but overuse of the provider device's memory is definitely inefficient
      // In future, we need to "statefully" make the copy only once and use it in all subgraphs referencing the initializer
      const TensorProto* tensor_proto = pair.second;
      TensorProto new_tensor_proto = *tensor_proto;
      *(new_tensor_proto.mutable_name()) = new_def_name;
      graph_.AddInitializedTensor(new_tensor_proto);

      replacements.insert(std::make_pair(provider_def, &new_def));
    }
  }

  for (auto p_node : provider_nodes_) {
    // make a copy of replacement map as the node may exclude mapping for InputDefs with MemTypeOnCpuExplicitly
    auto dup_replacements = replacements;

    const KernelCreateInfo* kci = nullptr;
    auto status = kernel_registries.SearchKernelRegistry(*p_node, &kci);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    if (kci == nullptr) continue;
    if (kci->kernel_def == nullptr) continue;
    ORT_THROW_IF_ERROR(Node::ForEachWithIndex(
        p_node->InputDefs(),
        [kci, &p_node, &dup_replacements](const onnxruntime::NodeArg& arg, size_t index) {
          if (utils::IsInputOnCpu(*p_node, kci, index)) dup_replacements.erase(&arg);
          return Status::OK();
        }));

    // normally initializers are only inputs, but things may change with ops like assign
    ORT_THROW_IF_ERROR(Node::ForEachWithIndex(
        p_node->OutputDefs(),
        [kci, &p_node, &dup_replacements](const onnxruntime::NodeArg& arg, size_t index) {
          if (utils::IsOutputOnCpu(*p_node, kci, index)) {
            ORT_ENFORCE(dup_replacements.find(&arg) == dup_replacements.end());
          }
          return Status::OK();
        }));

    p_node->ReplaceDefs(dup_replacements);
  }

  // This denotes a modification to the graph
  return !replacements.empty();
}

}  // namespace onnxruntime
