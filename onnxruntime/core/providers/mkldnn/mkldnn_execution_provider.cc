// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include "core/framework/allocator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/mkldnn/subgraph/mkldnn_func_kernel.h"
#include "mkldnn_execution_provider.h"
#include "mkldnn_fwd.h"

namespace onnxruntime {

constexpr const char* MKLDNN = "MklDnn";
constexpr const char* MKLDNN_CPU = "MklDnnCpu";

MKLDNNExecutionProvider::MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMklDnnExecutionProvider} {
  DeviceAllocatorRegistrationInfo default_memory_info({OrtMemTypeDefault,
                                                       [](int) { return onnxruntime::make_unique<CPUAllocator>(onnxruntime::make_unique<OrtMemoryInfo>(MKLDNN, OrtAllocatorType::OrtDeviceAllocator)); }, std::numeric_limits<size_t>::max()});

  DeviceAllocatorRegistrationInfo cpu_memory_info({OrtMemTypeCPUOutput,
                                                   [](int) { return onnxruntime::make_unique<CPUAllocator>(onnxruntime::make_unique<OrtMemoryInfo>(MKLDNN_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput)); }, std::numeric_limits<size_t>::max()});

  if (info.create_arena) {
    InsertAllocator(CreateAllocator(default_memory_info));

    InsertAllocator(CreateAllocator(cpu_memory_info));
  } else {
    InsertAllocator(std::shared_ptr<IArenaAllocator>(
        onnxruntime::make_unique<DummyArena>(default_memory_info.factory(0))));

    InsertAllocator(std::shared_ptr<IArenaAllocator>(
        onnxruntime::make_unique<DummyArena>(cpu_memory_info.factory(0))));
  }
}  // namespace onnxruntime

MKLDNNExecutionProvider::~MKLDNNExecutionProvider() {
}

namespace mkl_dnn {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm);

void RegisterMKLDNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetMklDnnKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterMKLDNNKernels(*kernel_registry);
  return kernel_registry;
}
}  // namespace mkl_dnn

std::shared_ptr<KernelRegistry> MKLDNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::mkl_dnn::GetMklDnnKernelRegistry();
  return kernel_registry;
}

bool MKLDNNExecutionProvider::UseSubgraph(const onnxruntime::GraphViewer& graph_viewer) const {
  // switch between mkldnn-vanilla and mkldnn-subgraph implementation using
  // MKLDNN_SUBGRAPH environment variable
  bool use_subgraph = true;

  bool FP16_graph = false;
  bool mkldnn_nodes_in_the_graph = false;
  int max_node_index = graph_viewer.MaxNodeIndex();

  for (auto node_index = 0; node_index < max_node_index; node_index++) {
    auto node = graph_viewer.GetNode(node_index);
    if (node == NULL)
      continue;

    if (!node->InputDefs().empty() && node->InputDefs()[0]->Type() != nullptr) {
      FP16_graph = node->InputDefs()[0]->Type()->find("16") != std::string::npos;
      break;
    }
  }

  for (auto node_index = 0; node_index < max_node_index; node_index++) {
    auto node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }

    auto op_it = mkldnn_ops_.find(node->OpType());
    if (op_it != mkldnn_ops_.end()) {
      mkldnn_nodes_in_the_graph = true;
      break;
    }
  }

  if (FP16_graph || !mkldnn_nodes_in_the_graph) {
    // FP16 not supported yet.
    use_subgraph = false;
  } else {
    const char* env = getenv("ORT_MKLDNN_SUBGRAPH");
    if (env != nullptr) {
      if (atoi(env) == 0) {
        use_subgraph = false;
      }
    }
  }
  return use_subgraph;
}

void MKLDNNExecutionProvider::CreateOrUpdateMklDnnNode(const Node* node,
                                                       std::shared_ptr<mkl_dnn::Subgraph>& subgraph_ptr,
                                                       mkl_dnn::Subgraph::SubgraphVariables& sub_var,
                                                       bool fused,
                                                       std::map<std::string, size_t>& output_to_source_node_map,
                                                       NodeAttributes& subgraph_attributes) const {
  const auto& node_inputs = node->InputDefs();
  sub_var.outputs.push_back(node->OutputDefs()[0]->Name());

  if (!fused) {
    mkl_dnn::MklDnnNode mkldnn_node;
    mkldnn_node.name = node->OpType();
    mkldnn_node.num_inputs = static_cast<int>(node->InputDefs().size());
    mkldnn_node.input_start_index = static_cast<int>(sub_var.inputs.size()) - 1;
    mkldnn_node.node_index = static_cast<int>(subgraph_ptr->mkldnn_nodes.size()) + 1;
    const auto& node_outputs = node->OutputDefs();
    mkldnn_node.output_name = node_outputs[0]->Name();
    if (node->OpType() == "Conv") {
      mkldnn_node.weight_name = node->InputDefs()[1]->Name();
    }
    for (size_t i = 0; i < node_inputs.size(); i++) {
      auto iter = output_to_source_node_map.find(node_inputs[i]->Name());
      if (iter != output_to_source_node_map.end())
        mkldnn_node.parent_nodes.push_back(iter->second);
    }
    subgraph_ptr->mkldnn_nodes.push_back(mkldnn_node);
    output_to_source_node_map.insert(std::make_pair(node_outputs[0]->Name(), subgraph_ptr->mkldnn_nodes.size() - 1));
  } else {
    subgraph_ptr->mkldnn_nodes.back().num_inputs += static_cast<int>(node->InputDefs().size() - 1);
    const auto& node_outputs = node->OutputDefs();
    output_to_source_node_map.erase(subgraph_ptr->mkldnn_nodes.back().output_name);
    subgraph_ptr->mkldnn_nodes.back().output_name = node_outputs[0]->Name();
    output_to_source_node_map.insert(std::make_pair(node_outputs[0]->Name(), subgraph_ptr->mkldnn_nodes.size() - 1));
  }

  // Add inputs which are not in the outputs vector.
  for (size_t i = 0; i < node_inputs.size(); i++) {
    auto itr = std::find(sub_var.outputs.begin(), sub_var.outputs.end(), node_inputs[i]->Name());
    if (itr == sub_var.outputs.end()) {
      sub_var.inputs.push_back(node_inputs[i]->Name());
    } else {
      // Vector of node outputs, which is input to other node
      // if node output is not input to any other node, then it's the end node
      // which we will find later
      sub_var.outputs_as_input_other_node.push_back(node_inputs[i]->Name());
    }
  }

  NodeAttributes attributes = node->GetAttributes();
  if (attributes.size() > 0) {
    size_t index = subgraph_ptr->mkldnn_nodes.size();
    std::string op_name;
    if (fused) {
      for (auto iter = node->InputNodesBegin(); iter != node->InputNodesEnd(); ++iter) {
        op_name = (*iter).OpType();
      }
    } else {
      op_name = node->OpType();
    }

    for (auto att_it = attributes.begin(); att_it != attributes.end(); ++att_it) {
      std::string key = op_name + "-" + std::to_string(index) + "-" + att_it->first;
      std::pair<std::string, ONNX_NAMESPACE::AttributeProto> att(key, att_it->second);
      subgraph_attributes[key] = att_it->second;
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>> MKLDNNExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& kernel_registries) const {
  ORT_UNUSED_PARAMETER(kernel_registries);

  // temporary switch to toggle between mkldnn-vanilla and mkldnn-subgraph implementation using
  // ORT_MKLDNN_SUBGRAPH environment variable
  if (UseSubgraph(graph_viewer) == false) {
    return IExecutionProvider::GetCapability(graph_viewer, kernel_registries);
  }

  LOGS_DEFAULT(INFO) << "Using MKL-DNN Subgraph";
  // use sub-graph implementation
  std::vector<std::unique_ptr<ComputeCapability>> result;
  mkl_dnn::Subgraph::SubgraphVariables sub_var;
  std::shared_ptr<mkl_dnn::Subgraph> subgraph_ptr;

  // We need graph name make PrimitivePool keys unique.
  // There are several identical graphs in Model zoo and only differ in
  // few attribute values. GetGraphName return graph-name + first-node-output name
  std::string graph_name = GetGraphName(graph_viewer);
  subgraph_ptr = onnxruntime::make_unique<mkl_dnn::Subgraph>(
      mkl_dnn::Subgraph(graph_name));

  // output name to node index map. Using it to find sub-graph end nodes
  // if output of a node is not an input to any node in a sub-graph is end node
  std::map<std::string, size_t> output_to_source_node_map;
  NodeAttributes subgraph_attributes;
  int node_index = 0;

  while (node_index < graph_viewer.MaxNodeIndex()) {
    auto node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      node_index++;
      continue;
    }

    if (IsDimensionSupported(node) == false) {
      node_index++;
      if (subgraph_ptr->mkldnn_nodes.size() > 0) {
        CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
        subgraph_ptr = std::make_shared<mkl_dnn::Subgraph>(mkl_dnn::Subgraph(graph_name));
        subgraph_attributes.clear();
        output_to_source_node_map.clear();
      }
      continue;
    }

    auto op_it = mkldnn_ops_.find(node->OpType());
    if (op_it != mkldnn_ops_.end()) {
      sub_var.subgraph_node_indexes.push_back(node->Index());

      // can we fuse (at mkldnn level) nodes?
      bool fused = false;
      if (sub_var.subgraph_node_indexes.size() > 1 && node->OpType() == "BatchNormalization") {
        if (subgraph_ptr->mkldnn_nodes.back().name == "Conv") {
          subgraph_ptr->mkldnn_nodes.back().name += "-BatchNormalization";
          fused = true;
        }
      }
      if (sub_var.subgraph_node_indexes.size() > 1 && node->OpType() == "Relu") {
        if (subgraph_ptr->mkldnn_nodes.back().name == "Conv-BatchNormalization" || subgraph_ptr->mkldnn_nodes.back().name == "BatchNormalization" || subgraph_ptr->mkldnn_nodes.back().name == "Conv") {
          subgraph_ptr->mkldnn_nodes.back().name += "-Relu";
          fused = true;
        }
      }

      // Create MklDnn node:
      //   Update inputs, outputs and parent nodes
      //   Collect attributes and modify the key to make it unique
      CreateOrUpdateMklDnnNode(node, subgraph_ptr, sub_var, fused, output_to_source_node_map, subgraph_attributes);

      auto temp_index = node_index + 1;
      if (temp_index < graph_viewer.MaxNodeIndex()) {
        if (!sub_var.subgraph_node_indexes.empty()) {
          // if next node is mkldnn node and if it's input is not output of current node
          //   if next node input is output of any of the nodes in sub-graph continue
          // else
          //   break and create sub-graph
          auto next_node = graph_viewer.GetNode(temp_index);
          while (next_node == nullptr) {
            temp_index++;
            next_node = graph_viewer.GetNode(temp_index);
          }
          auto sub_it = mkldnn_ops_.find(next_node->OpType());
          if (sub_it != mkldnn_ops_.end()) {
            const auto& next_node_inputs = next_node->InputDefs();
            bool input_from_subgraph = true;
            size_t inputs_count = 1;
            if (next_node->OpType() == "Sum")
              inputs_count = next_node_inputs.size();
            for (size_t i = 0; i < inputs_count; i++) {
              auto in = next_node_inputs[i];
              auto itr = std::find(sub_var.outputs.begin(), sub_var.outputs.end(), in->Name());
              if (itr == sub_var.outputs.end()) {
                input_from_subgraph = false;
              }
            }
            if (input_from_subgraph == false) {
              CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
              subgraph_attributes.clear();
              subgraph_ptr = std::make_shared<mkl_dnn::Subgraph>(mkl_dnn::Subgraph(graph_name));
              output_to_source_node_map.clear();
            }
          }
        }
        if (!sub_var.subgraph_node_indexes.empty()) {
          if (node->GetOutputEdgesCount() > 1) {
            // If current node has branches
            //    iterate and see if all nodes are mkldnn ops OR
            //      it ends in node with same number of input edges (mkldnn node or cpu node)
            //      create sub-graph
            bool create_subgraph = false;
            bool break_loop = false;
            while (!break_loop) {
              if (temp_index > graph_viewer.MaxNodeIndex())
                break_loop = true;

              auto next_node = graph_viewer.GetNode(temp_index);
              while (next_node == nullptr) {
                temp_index++;
                next_node = graph_viewer.GetNode(temp_index);
              }
              if (next_node->GetInputEdgesCount() == node->GetOutputEdgesCount()) {
                // if all nodes in the branch loop are mkldnn nodes
                // then continue with adding nodes to sub-graph
                break_loop = true;
              }
              // inner nodes. if inner nodes are not  mkldnn nodes
              // create subgraph (inception v2)
              auto sub_it = mkldnn_ops_.find(next_node->OpType());
              if (sub_it == mkldnn_ops_.end()) {
                // break and create a sub-graph
                break_loop = true;
                create_subgraph = true;
              }
              temp_index++;
            }
            if (create_subgraph) {
              CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
              subgraph_ptr = std::make_shared<mkl_dnn::Subgraph>(mkl_dnn::Subgraph(graph_name));
              subgraph_attributes.clear();
              output_to_source_node_map.clear();
            }
          }
        }
      }
    } else {
      if (!sub_var.subgraph_node_indexes.empty()) {
        CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
        subgraph_ptr = std::make_shared<mkl_dnn::Subgraph>(mkl_dnn::Subgraph(graph_name));
        subgraph_attributes.clear();
        output_to_source_node_map.clear();
      }
    }
    node_index++;
  }  // graph_viewer node iterator ends
  if (!sub_var.subgraph_node_indexes.empty()) {
    CreateMetaDef(graph_viewer, subgraph_attributes, subgraph_ptr, sub_var, result);
    subgraph_ptr = std::make_shared<mkl_dnn::Subgraph>(mkl_dnn::Subgraph(graph_name));
    subgraph_attributes.clear();
    output_to_source_node_map.clear();
  }
  return result;
}

void MKLDNNExecutionProvider::CreateMetaDef(const onnxruntime::GraphViewer& graph_viewer,
                                            const NodeAttributes& subgraph_attributes,
                                            std::shared_ptr<mkl_dnn::Subgraph>& subgraph_ptr,
                                            mkl_dnn::Subgraph::SubgraphVariables& sub_var,
                                            std::vector<std::unique_ptr<ComputeCapability>>& result) const {
  std::string graph_fused_nodes;
  std::string node_list;
  std::string subgraph_id = std::to_string(subgraph_index_);
  subgraph_index_++;

  // This is a list of initializers that subgraph considers as constants.
  // Example weights, reshape shape etc.
  std::unordered_set<std::string> input_initializers;

  // Create ng_required_initializers attribute of NGraphCustomOp
  ONNX_NAMESPACE::AttributeProto initializers;
  initializers.set_name("initializers");
  initializers.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS);

  for (const auto& init : sub_var.inputs) {
    if (graph_viewer.GetAllInitializedTensors().count(init)) {
      auto tensor = initializers.add_tensors();
      *tensor = *(graph_viewer.GetAllInitializedTensors().at(init));
    }
  }

  auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->attributes["initializers"] = initializers;
  meta_def->name = "MkldnnCustomOp" + std::to_string(subgraph_index_);
  meta_def->domain = kMSDomain;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  meta_def->inputs = sub_var.inputs;
  meta_def->attributes.insert(subgraph_attributes.begin(), subgraph_attributes.end());

  // Find the end nodes
  for (auto& mklnode : subgraph_ptr->mkldnn_nodes) {
    auto itr = std::find(sub_var.outputs_as_input_other_node.begin(),
                         sub_var.outputs_as_input_other_node.end(), mklnode.output_name);
    if (itr == sub_var.outputs_as_input_other_node.end()) {
      meta_def->outputs.push_back(mklnode.output_name);
      mklnode.output_index = static_cast<int>(meta_def->outputs.size()) - 1;
    }
  }

  ONNX_NAMESPACE::AttributeProto ap;
  ap.set_s(subgraph_id);
  ap.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  meta_def->attributes["subgraph_id"] = ap;
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  sub_graph->nodes = sub_var.subgraph_node_indexes;
  sub_graph->SetMetaDef(meta_def);
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  mkl_subgraphs_.insert(std::make_pair(subgraph_id, subgraph_ptr));

  // Reset subgraph and meta_Def
  sub_var.Reset();
}

Status MKLDNNExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                        std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* fused_node : fused_nodes) {
    auto attributes = fused_node->GetAttributes();
    NodeComputeInfo compute_info;

    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      auto* p = new onnxruntime::mkl_dnn::MkldnnFuncKernel<float>(context, attributes, this);
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<onnxruntime::mkl_dnn::MkldnnFuncKernel<float>*>(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      onnxruntime::mkl_dnn::MkldnnFuncKernel<float>* custom_op = reinterpret_cast<mkl_dnn::MkldnnFuncKernel<float>*>(state);
      return custom_op->Compute(api, context);
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime
