// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nnapi_execution_provider.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "builders/model_builder.h"

namespace onnxruntime {

constexpr const char* NNAPI = "Nnapi";

NnapiExecutionProvider::NnapiExecutionProvider()
    : IExecutionProvider{onnxruntime::kNnapiExecutionProvider}, nnapi_(NnApiImplementation()) {
  DeviceAllocatorRegistrationInfo device_info{OrtMemTypeDefault,
                                              [](int) { return onnxruntime::make_unique<CPUAllocator>(
                                                            onnxruntime::make_unique<OrtMemoryInfo>(NNAPI,
                                                                                                    OrtAllocatorType::OrtDeviceAllocator)); },
                                              std::numeric_limits<size_t>::max()};
  InsertAllocator(CreateAllocator(device_info));

  DeviceAllocatorRegistrationInfo cpu_memory_info({OrtMemTypeCPUOutput,
                                                   [](int) { return onnxruntime::make_unique<CPUAllocator>(onnxruntime::make_unique<OrtMemoryInfo>(NNAPI, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput)); },
                                                   std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

NnapiExecutionProvider::~NnapiExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
NnapiExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // This method is based on that of TRT EP
  // Construct modelproto from graph
  onnxruntime::Model model(graph.Name(), true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                           graph.DomainToVersionMap(), std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
  onnxruntime::Graph& graph_build = model.MainGraph();
  std::set<NodeArg*> all_node_inputs;
  for (const auto& node : graph.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
      all_node_inputs.insert(&n_input);
    }
    for (auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }
  const auto graph_outputs = graph.GetOutputs();
  //Add initializer to graph
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(graph_build.Resolve().IsOK());
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  nnapi::ModelBuilder builder(model_proto, nnapi_);
  const auto supported_nodes_vector = builder.GetSupportedNodes();
  LOGS_DEFAULT(INFO) << "Support vector size is " << supported_nodes_vector.size();
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (1)
    return result;

  // Find inputs, initializers and outputs for each supported subgraph
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  int counter = 0;
  for (const auto& group : supported_nodes_vector) {
    if (group.empty())
      continue;

    std::unordered_set<size_t> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(node_index[index]);
    }

    std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
    // Find inputs and outputs of the subgraph
    std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add;
    std::unordered_set<const NodeArg*> erased;
    int input_order = 0;
    int output_order = 0;

    for (const auto& index : group) {
      sub_graph->nodes.push_back(node_index[index]);
      const auto& node = graph.GetNode(node_index[index]);

      for (const auto& input : node->InputDefs()) {
        const auto& it = fused_outputs.find(input);
        if (it != fused_outputs.end()) {
          fused_outputs.erase(it);
          erased.insert(input);
        }
        //only when input is neither in output list nor erased list, add the input to input list
        else if (erased.find(input) == erased.end()) {
          fused_inputs[input] = input_order++;
        }
      }

      // For output searching, there is a special case:
      // If node's OutputEdges are more than its outputs, meaning certain output is used more than once,
      // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
      // to the output list
      if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
        for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
          const auto& node_idx = it->GetNode().Index();
          const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];

          if (node_set.find(node_idx) != node_set.end()) {
            const auto& iter = fused_inputs.find(output);
            if (iter != fused_inputs.end()) {
              fused_inputs.erase(iter);
              erased.insert(output);
            } else if (erased.find(output) == erased.end()) {
              fused_outputs[output] = output_order++;
            }
          } else {
            fused_outputs_to_add[output] = output_order++;
          }
        }
      } else {
        for (const auto& output : node->OutputDefs()) {
          const auto& it = fused_inputs.find(output);
          if (it != fused_inputs.end()) {
            fused_inputs.erase(it);
            erased.insert(output);
          }
          // only when output is neither in input list nor erased list, add the output to output list
          else if (erased.find(output) == erased.end()) {
            fused_outputs[output] = output_order++;
          }
        }
      }
    }

    fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
    // Sort inputs and outputs by the order they were added
    std::multimap<int, const NodeArg*> inputs, outputs;

    for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
      inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
    }

    for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
      for (const auto& x : all_node_inputs) {
        if (x->Name() == it->first->Name()) {
          outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
          break;
        }
      }
      if (std::find(graph_outputs.begin(), graph_outputs.end(), it->first) != graph_outputs.end()) {
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }
    }

    // Assign inputs and outputs to subgraph's meta_def
    auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
    meta_def->name = "NNAPI_" + std::to_string(counter++);
    meta_def->domain = kMSDomain;

    for (const auto& input : inputs) {
      meta_def->inputs.push_back(input.second->Name());
    }

    for (const auto& output : outputs) {
      meta_def->outputs.push_back(output.second->Name());
    }

    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    meta_def->since_version = 1;
    sub_graph->SetMetaDef(meta_def);

    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  return result;
}

common::Status NnapiExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  (void)fused_nodes;
  (void)node_compute_funcs;
  return Status::OK();
}
}  // namespace onnxruntime
