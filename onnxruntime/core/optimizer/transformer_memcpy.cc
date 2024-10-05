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
// note that GraphTransformer::Apply() is supposed to be stateless, so this cannot derive from GraphTransformer
class TransformerMemcpyImpl {
 public:
  TransformerMemcpyImpl(onnxruntime::Graph& graph, const std::string& provider)
      : graph_(graph), provider_(provider) {
  }

  bool ModifyGraph(const KernelRegistryManager& schema_registries, const logging::Logger& logger,
                   int& copy_node_counter);

 private:
  void ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries,
                   InitializedTensorSet& initializers_consumed);
  void BuildDefsMapping(const NodeArg* arg, const KernelRegistryManager& kernel_registries);
  void AddCopyNode(const NodeArg* arg, bool is_input, const logging::Logger& logger);
  bool ProcessInitializers(const KernelRegistryManager& kernel_registries,
                           const InitializedTensorSet& initializers_consumed);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TransformerMemcpyImpl);

  // use value-based compare to make sure transformer output order is consistent
  struct NodeArgCompare {
    bool operator()(const NodeArg* lhs, const NodeArg* rhs) const {
      return lhs->Name() < rhs->Name();
    }
  };

  std::set<const NodeArg*, NodeArgCompare> cpu_input_defs_;  // input defs that are consumed on CPU
  std::set<const NodeArg*, NodeArgCompare> cpu_output_defs_;
  std::set<const NodeArg*, NodeArgCompare> gpu_input_defs_;  // input defs that are consumed on GPU
  std::set<const NodeArg*, NodeArgCompare> gpu_output_defs_;
  std::set<const Node*, NodeCompare> gpu_nodes_;  // GPU based nodes

  // Map of NodeArg to node that requires the value to be on GPU to consume as input
  std::map<const NodeArg*, std::set<Node*, NodeCompare>> gpu_input_nodes_;
  // Map of NodeArg to node that produces the value on GPU
  std::map<const NodeArg*, std::set<Node*, NodeCompare>> gpu_output_nodes_;

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

struct GpuEPs {
  bool nvidia{false};
  bool amd{false};
  bool webgpu{false};
};

// very simple GraphTransformer that uses TransformerMemcpyImpl for each graph
// and mainly provides the subgraph recursion functionality
common::Status MemcpyTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  // sanity check that we don't have unrelated GPU EPs, as the logic does not handle this scenario.
  // CUDA/TensorRT vs ROCm/MIGraphX vs WebGPU are not cross compatible.
  // To support this we'd need to be able to insert copy nodes between incompatible GPU devices.
  // As there's no known scenario where we would mix these GPUs the complexity of that is not currently justified.
  GpuEPs eps;
  bool incompatible_gpu_eps = false;
  for (auto& provider : provider_types_) {
    if (provider == kCudaExecutionProvider || provider == kTensorrtExecutionProvider) {
      eps.nvidia = true;
      incompatible_gpu_eps |= eps.amd || eps.webgpu;
    } else if (provider == kRocmExecutionProvider || provider == kMIGraphXExecutionProvider) {
      eps.amd = true;
      incompatible_gpu_eps |= eps.nvidia || eps.webgpu;
    } /*else if (provider == kWebGpuExecutionProvider) {
      eps.webgpu = true;
      incompatible_gpu_eps |= eps.nvidia || eps.amd;
    }*/
  }

  ORT_ENFORCE(!incompatible_gpu_eps, "Mixing CUDA/TensorRT, ROCm/MIGraphX, and WebGPU is not supported.");

  for (auto& provider : provider_types_) {
    if (!utils::ProviderIsCpuBased(provider)) {
      TransformerMemcpyImpl copy_impl(graph, provider);

      int copy_node_counter = 0;
      auto current_modified = copy_impl.ModifyGraph(registry_manager_, logger, copy_node_counter);
      if (copy_node_counter > 0) {
        LOGS(logger, WARNING) << copy_node_counter << " Memcpy nodes were added to the graph " << graph.Name()
                              << " for " << provider;
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

Overview: Transform the input graph as follows:

(1) For every initializer W that is referenced by both GPU and CPU nodes, we create a duplicate initializer W2
and change all GPU nodes to consume this copy.

(2) For every value X that is computed by a GPU node and referenced by a CPU node, we introduce a new value
X2. We replace all references to X in GPU nodes by X2, and introduce a copy from X2 to X. (All graph outputs are
considered to be CPU based.)

(3 For every value X that is computed by a CPU node and referenced by a GPU node, we introduce a new value X2.
We replace all references to X in GPU nodes by X2, and introduce a copy from X to X2. (All graph inputs are considered
to be CPU based.)

Note that every value is computed at a unique point (either GPU or CPU), but it may be referenced and used at multiple
points by both GPU and CPU.

It does not support multiple incompatible GPU devices (e.g. CUDA/TensorRT vs ROCm/MIGraphX vs WebGPU) being enabled.
It does not handle/optimize copies between two different compatible GPU devices (e.g. 2 different nVidia GPUs).
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

  // for initializers consumed on CPU and GPU, duplicate so there's a 1:1 relationship between initializer and device
  if (ProcessInitializers(kernel_registries, initializers_consumed)) {
    modified = true;
  }

  for (auto arg : graph_.GetInputs()) {
    BuildDefsMapping(arg, kernel_registries);
  }

  for (auto arg : cpu_input_defs_) {
    BuildDefsMapping(arg, kernel_registries);
  }

  for (auto arg : cpu_output_defs_) {
    BuildDefsMapping(arg, kernel_registries);
  }

  for (const auto* arg : graph_.GetInputs())
    // For inputs we need to create a copy node only when the input is connected to both GPU and CPU nodes.
    // Otherwise utils::CopyInputsAcrossDevices() will do the job.
    if (gpu_input_defs_.count(arg) && cpu_input_defs_.count(arg)) {
      AddCopyNode(arg, true, logger);
      ++copy_node_counter;
      modified = true;
    }

  for (auto arg : cpu_output_defs_) {
    if (gpu_input_defs_.count(arg)) {
      AddCopyNode(arg, true, logger);
      ++copy_node_counter;
      modified = true;
    }
  }

  for (auto arg : gpu_output_defs_) {
    if (cpu_input_defs_.count(arg)) {
      AddCopyNode(arg, false, logger);
      ++copy_node_counter;
      modified = true;
    }
  }

  // Process implicit inputs in subgraphs that are explicitly consumed on both GPU and CPU nodes.
  // This is replicating the copy logic for explicit graph inputs that is implemented in SessionState finalization.
  if (graph_.IsSubgraph()) {
    for (auto arg : graph_.ParentNode()->ImplicitInputDefs()) {
      if (gpu_input_defs_.count(arg) && cpu_input_defs_.count(arg)) {
        // There is an explicit consumer of the NodeArg on GPU and CPU nodes in the current graph.
        //
        // Implicit consumers (node in current graph has a nested subgraph with consumer node) will be handled by the
        // copy to device that happens during subgraph execution.

        // Looking into `gpu_input_defs_` using a NodeArg pointer from the outer scope is okay
        // because the `find` uses NodeArgCompare which only matches on name, and the name will match the implicit
        // input from the parent node.
        const auto* node_arg_in_current_graph_level = *gpu_input_defs_.find(arg);

        AddCopyNode(node_arg_in_current_graph_level, true, logger);
        ++copy_node_counter;
        modified = true;
      }
    }
  }

  return modified;
}

void TransformerMemcpyImpl::ProcessDefs(onnxruntime::Node& node, const KernelRegistryManager& kernel_registries,
                                        InitializedTensorSet& initializers_consumed) {
  const auto& node_provider_type = node.GetExecutionProviderType();
  const bool is_cpu_node = utils::ProviderIsCpuBased(node_provider_type);

  if (is_cpu_node == false) {
    gpu_nodes_.insert(&node);

    // KernelCreateInfo might be nullptr for custom kernel
    const KernelCreateInfo* kci = nullptr;
    ORT_IGNORE_RETURN_VALUE(kernel_registries.SearchKernelRegistry(node, &kci));

    bool is_implicit_input = false;
    auto process_inputs =
        [this, &node, &kci, &initializers_consumed, &is_implicit_input](const NodeArg& arg, size_t index) {
          // check if this NodeArg is an initializer
          const auto* initializer_tensor_proto = GetInitializer(graph_, arg.Name(), true);
          if (initializer_tensor_proto != nullptr) {
            initializers_consumed[arg.Name()] = initializer_tensor_proto;
          }

          // implicit inputs are consumed in subgraphs and have no location info in the kernel def.
          // The subgraph execution logic will do the necessary copy if the input is required on a different device.
          // PlannerImpl::ComputeUseCounts has matching logic so the allocation plan does the same thing
          if (!is_implicit_input) {
            if (utils::IsInputOnCpu(node, kci, index)) {
              cpu_input_defs_.insert(&arg);
            } else {
              gpu_input_defs_.insert(&arg);
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
        cpu_output_defs_.insert(arg);
      else
        gpu_output_defs_.insert(arg);
    }
  } else {
    for (const auto* arg : node.InputDefs()) {
      if (arg->Exists())
        cpu_input_defs_.insert(arg);
    }

    // Never add an implicit def to gpu_input_defs_ or cpu_input_defs_.
    // This is because we don't want to add copy nodes on account of implicit inputs to nodes.
    // We will rely on utils::CopyInputsAcrossDevices() to do the job during subgraph execution setup.
    //
    // for (const auto* arg : node.ImplicitInputDefs()) {
    //  if (arg->Exists())
    //    cpu_input_defs_.insert(arg);
    //}

    for (auto* arg : node.MutableOutputDefs()) {
      if (arg->Exists())
        cpu_output_defs_.insert(arg);
    }
  }
}

// collect the input/output NodeArg's that are GPU based
void TransformerMemcpyImpl::BuildDefsMapping(const NodeArg* arg, const KernelRegistryManager& kernel_registries) {
  for (auto& cur_node : graph_.Nodes()) {
    if (cur_node.OpType() == "MemcpyFromHost" || cur_node.OpType() == "MemcpyToHost") {
      continue;
    }

    auto node_provider_type = cur_node.GetExecutionProviderType();
    if (!utils::ProviderIsCpuBased(node_provider_type)) {
      auto input_it = std::find(cur_node.InputDefs().begin(), cur_node.InputDefs().end(), arg);
      auto output_it = std::find(cur_node.OutputDefs().begin(), cur_node.OutputDefs().end(), arg);

      int arg_input_index = input_it != cur_node.InputDefs().end()
                                ? static_cast<int>(input_it - cur_node.InputDefs().begin())
                                : -1;
      int arg_output_index = output_it != cur_node.OutputDefs().end()
                                 ? static_cast<int>(output_it - cur_node.OutputDefs().begin())
                                 : -1;

      if (arg_input_index == -1 && arg_output_index == -1) {
        continue;
      }

      const KernelCreateInfo* kci = nullptr;
      ORT_IGNORE_RETURN_VALUE(kernel_registries.SearchKernelRegistry(cur_node, &kci));

      // GPU based nodes can potentially consume/produce CPU based values. check the kernel definition before adding.
      // If no KernelCreateInfo is availabe assume all inputs/outputs are on GPU.
      if (arg_input_index != -1) {
        if (!kci || !utils::IsInputOnCpu(cur_node, kci, arg_input_index)) {
          gpu_input_nodes_[arg].insert(&cur_node);
        }
      }

      if (arg_output_index != -1) {
        if (!kci || !utils::IsOutputOnCpu(cur_node, kci, arg_output_index)) {
          gpu_output_nodes_[arg].insert(&cur_node);
        }
      }
    }
  }
}

void TransformerMemcpyImpl::AddCopyNode(const NodeArg* arg, bool is_input, const logging::Logger& logger) {
  // we need to convert `arg` to a non-const NodeArg* as the AddNode needs that for shape inferencing to work.
  // we _could_ do that by finding the graph input or producer node or consumer node and using the mutable graph_
  // member to access that. the code complexity is not worth it so we use const_cast to be pragmatic.
  NodeArg* mutable_arg = const_cast<NodeArg*>(arg);

  // create unique name for new def
  std::string new_def_name = graph_.GenerateNodeArgName(arg->Name() + "_" + provider_);

  NodeArg* new_arg = &graph_.GetOrCreateNodeArg(new_def_name, arg->TypeAsProto());
  NodeArg* src_arg = is_input ? mutable_arg : new_arg;
  NodeArg* dst_arg = is_input ? new_arg : mutable_arg;

  // create unique name for copy node
  std::string new_node_name = graph_.GenerateNodeName("Memcpy");

  const auto op_name = is_input ? "MemcpyFromHost" : "MemcpyToHost";
  LOGS(logger, INFO) << "Add " << op_name << (is_input ? " after " : " before ") << arg->Name()
                     << " for " << provider_;

  auto& new_node = graph_.AddNode(new_node_name, op_name, "Copy from/to host memory", {src_arg}, {dst_arg});

  new_node.SetExecutionProviderType(provider_);
  std::map<const NodeArg*, NodeArg*> map = {{arg, new_arg}};

  auto it = gpu_input_nodes_.find(arg);
  if (it != gpu_input_nodes_.end()) {
    for (auto* node : it->second)
      node->ReplaceDefs(map);
  }

  it = gpu_output_nodes_.find(arg);
  if (it != gpu_output_nodes_.end()) {
    for (auto* node : it->second)
      node->ReplaceDefs(map);
  }
}

template <typename NodeArgSetType>
static const NodeArg* FindNodeArg(const NodeArgSetType& def_set, const std::string& name) {
  NodeArg def(name, nullptr);
  auto it = def_set.find(&def);  // this works since we use name to compare NodeArg

  return it != def_set.end() ? *it : nullptr;
}

// We duplicate any initializer that is used by both provider nodes and non-provider nodes
// to ensure that provider nodes and non-provider nodes don't share initializers, as they
// need to stay in different memory locations.
bool TransformerMemcpyImpl::ProcessInitializers(const KernelRegistryManager& kernel_registries,
                                                const InitializedTensorSet& initializers_consumed) {
  std::map<const NodeArg*, NodeArg*> replacements;
  for (const auto& pair : initializers_consumed) {
    const auto& name = pair.first;
    const NodeArg* gpu_def = FindNodeArg(gpu_input_defs_, name);
    const NodeArg* cpu_def = FindNodeArg(cpu_input_defs_, name);

    // value is consumed on GPU and CPU and needs to be duplicated
    if (gpu_def != nullptr && cpu_def != nullptr) {
      std::string new_def_name = graph_.GenerateNodeArgName(name);
      auto& new_def = graph_.GetOrCreateNodeArg(new_def_name, gpu_def->TypeAsProto());

      // We make a copy of the initializer that is to be consumed by the GPU-based Node so that SessionState
      // finalization can copy it to the GPU.
      // TODO: The copy being made is possibly redundant if this occurs in a subgraph.
      // When multiple subgraphs consume the same initializer as an implicit input, multiple copies of the initializer
      // will be made to the GPU. This should not directly affect runtime performance as the copies occur during
      // initialization but overuse of the GPU memory is definitely inefficient.
      // In future, we need to "statefully" make the copy only once and use it in all subgraphs referencing
      // the initializer
      const TensorProto* tensor_proto = pair.second;
      TensorProto new_tensor_proto = *tensor_proto;
      *(new_tensor_proto.mutable_name()) = new_def_name;
      graph_.AddInitializedTensor(new_tensor_proto);

      replacements.insert(std::make_pair(gpu_def, &new_def));
    }
  }

  for (auto p_node : gpu_nodes_) {
    // make a copy of replacement map as the node may exclude mapping for InputDefs with MemTypeOnCpuExplicitly
    auto dup_replacements = replacements;

    const KernelCreateInfo* kci = nullptr;
    auto status = kernel_registries.SearchKernelRegistry(*p_node, &kci);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());

    if (kci == nullptr || kci->kernel_def == nullptr) {
      continue;
    }

    ORT_THROW_IF_ERROR(Node::ForEachWithIndex(p_node->InputDefs(),
                                              [kci, &p_node, &dup_replacements](const NodeArg& arg, size_t index) {
                                                if (utils::IsInputOnCpu(*p_node, kci, index)) {
                                                  dup_replacements.erase(&arg);
                                                }

                                                return Status::OK();
                                              }));

    // normally initializers are only inputs, but things may change with ops like assign
    ORT_THROW_IF_ERROR(Node::ForEachWithIndex(p_node->OutputDefs(),
                                              [kci, &p_node, &dup_replacements](const NodeArg& arg, size_t index) {
                                                if (utils::IsOutputOnCpu(*p_node, kci, index)) {
                                                  ORT_ENFORCE(dup_replacements.find(&arg) == dup_replacements.end());
                                                }
                                                return Status::OK();
                                              }));

    // convert const Node* to non-const via GetNode call and do the replacement
    graph_.GetNode(p_node->Index())->ReplaceDefs(dup_replacements);
  }

  // This denotes a modification to the graph
  return !replacements.empty();
}

}  // namespace onnxruntime
