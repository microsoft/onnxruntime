// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include "core/providers/shared_library/provider_api.h"
#include <unordered_set>
#include "dnnl_execution_provider.h"
#include "dnnl_fwd.h"
#include "dnnl_node_capability.h"

#include <iomanip>
#include <fstream>
#include "gsl/gsl"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

constexpr const char* DNNL = "Dnnl";
constexpr const char* DNNL_CPU = "DnnlCpu";

DNNLExecutionProvider::DNNLExecutionProvider(const DNNLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kDnnlExecutionProvider, true} {
  AllocatorCreationInfo default_memory_info(
      {[](int) {
        return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(DNNL, OrtAllocatorType::OrtDeviceAllocator));
      }},
      0, info.create_arena);

  AllocatorCreationInfo cpu_memory_info(
      {[](int) {
        return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(DNNL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0,
                                                             OrtMemTypeCPUOutput));
      }},
      0, info.create_arena);

  InsertAllocator(CreateAllocator(default_memory_info));
  InsertAllocator(CreateAllocator(cpu_memory_info));

  //debug env variable to dump subgraphs
  const std::string dump_subgraphs_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_DUMP_SUBGRAPHS");
  if (!dump_subgraphs_env.empty()) {
    dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
  }

  const std::string debug_log_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_DEBUG_LOG");
  if (!debug_log_env.empty()) {
    debug_log_ = (std::stoi(debug_log_env) == 0 ? false : true);
  }
}  // namespace onnxruntime

DNNLExecutionProvider::~DNNLExecutionProvider() {
}

std::vector<std::vector<NodeIndex>> DNNLExecutionProvider::GetSupportedNodes(const GraphViewer& graph_viewer) const {
  std::vector<std::vector<size_t>> supported_node_vecs;
  std::vector<size_t> supported_node_vec;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    auto node_idx = node_indices[i];
    const auto* node(graph_viewer.GetNode(node_idx));

    bool supported = opManager_.IsNodeSupported(node, graph_viewer);

    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "Operator type: [" << node->OpType()
                          << "] index: [" << node_idx
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    }

    if (supported) {
      supported_node_vec.push_back(node_idx);
    } else {
      if (!supported_node_vec.empty()) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (!supported_node_vec.empty()) {
    supported_node_vecs.push_back(supported_node_vec);
  }

  return supported_node_vecs;
}

void ToGraphProtoInternal(const GraphViewer& graph, ONNX_NAMESPACE::GraphProto& graph_proto) {
  for (const auto* input_arg : graph.GetInputs()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  // Add all graph's initializers to the subgraph
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    *(graph_proto.mutable_initializer()->Add()) = *(tensor.second);
  }

  for (const auto* output_arg : graph.GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  for (const auto* value_info : graph.GetValueInfo()) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph.GetNodesInTopologicalOrder()) {
    const gsl::not_null<ONNX_NAMESPACE::NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{graph.GetNode(node_idx)};
    p_node->ToProto(*node_proto);
  }
}

std::vector<std::unique_ptr<ComputeCapability>> DNNLExecutionProvider::GetCapability(
    const GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& kernel_registries) const {
  //follow from coreml ep's Getcapability

  ORT_UNUSED_PARAMETER(kernel_registries);

  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  const auto node_groups = GetSupportedNodes(graph_viewer);

  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  size_t num_of_supported_nodes = 0;
  for (const auto& group : node_groups) {
    if (group.empty())
      continue;

    num_of_supported_nodes += group.size();

    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "DNNLExecutionProvider::GetCapability, current supported node group size: "
                          << group.size();
    }

    std::unordered_set<NodeIndex> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(index);
    }

    auto sub_graph = onnxruntime::IndexedSubGraph::Create();
    std::unordered_set<const NodeArg*> node_outputs;
    std::unordered_set<const NodeArg*> subgraph_inputs;
    std::unordered_set<const NodeArg*> subgraph_outputs;
    std::vector<const NodeArg*> ordered_subgraph_inputs;
    std::vector<const NodeArg*> ordered_subgraph_outputs;

    for (const auto& index : group) {
      sub_graph->Nodes().push_back(index);
      const auto* node = graph_viewer.GetNode(index);

      for (const auto* input : node->InputDefs()) {
        // if the node input was not produced by this subgraph, add it to the subgraph inputs.
        if (node_outputs.count(input) == 0) {
          if (subgraph_inputs.count(input) == 0) {
            subgraph_inputs.insert(input);
            ordered_subgraph_inputs.push_back(input);
          }
        }
      }

      const auto& output_defs = node->OutputDefs();
      for (const auto* output_def : output_defs) {
        node_outputs.insert(output_def);
        // if output is overall graph output we need to produce it.
        if (graph_outputs.count(output_def) != 0) {
          ordered_subgraph_outputs.push_back(output_def);
        }
      }

      // if output connects to a node not in this subgraph we need to produce it
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        if (node_set.count(it->GetNode().Index()) == 0) {
          const auto* output_def = output_defs[it->GetSrcArgIndex()];
          if (subgraph_outputs.count(output_def) == 0 && graph_outputs.count(output_def) == 0) {
            subgraph_outputs.insert(output_def);
            ordered_subgraph_outputs.push_back(output_def);
          }
        }
      }
    }

    // Assign inputs and outputs to subgraph's meta_def
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    auto meta_def = ::onnxruntime::IndexedSubGraph_MetaDef::Create();
    meta_def->name() = "DNNL_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    meta_def->domain() = kMSDomain;
    meta_def->since_version() = 1;
    meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;

    for (const auto& input : ordered_subgraph_inputs) {
      meta_def->inputs().push_back(input->Name());
    }

    for (const auto& output : ordered_subgraph_outputs) {
      meta_def->outputs().push_back(output->Name());
    }

    sub_graph->SetMetaDef(std::move(meta_def));

    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }

  if (debug_log_) {
    float percent_dnnl = 100.0f * (static_cast<float>(num_of_supported_nodes) / static_cast<float>(graph_viewer.NumberOfNodes()));
    LOGS_DEFAULT(ERROR) << "DNNLExecutionProvider::GetCapability,"
                        << " number of partitions supported by DNNL: " << result.size()
                        << " number of nodes in the graph: " << graph_viewer.NumberOfNodes()
                        << " number of nodes supported by DNNL: " << num_of_supported_nodes
                        << std::fixed << std::setprecision(2) << " (" << percent_dnnl << "%)";
  }

  if (dump_subgraphs_) {
    auto model = graph_viewer.CreateModel(*GetLogger());
    auto model_proto = model->ToProto();
    ToGraphProtoInternal(graph_viewer, *model_proto->mutable_graph());
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    std::fstream dump("DNNL_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id) + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }

  return result;
}

Status DNNLExecutionProvider::Compile(const std::vector<Node*>& fused_nodes,
                                      std::vector<NodeComputeInfo>& node_compute_funcs) {
  //follow from coreml ep's Compile
  for (const auto* fused_node : fused_nodes) {
    const auto* func_body = fused_node->GetFunctionBody();
    if (!func_body) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    }
    const Graph& graph_body = func_body->Body();
    auto graph_body_viewer = graph_body.CreateGraphViewer();

    if (dump_subgraphs_) {
      auto model = graph_body_viewer->CreateModel(*GetLogger());
      auto model_proto = model->ToProto();
      *model_proto->mutable_graph() = *graph_body.ToGraphProto();
      model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
      std::fstream dump(fused_node->Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
      model_proto->SerializeToOstream(dump);
    }

    //subgraph
    auto dnnl_subgraph = std::make_unique<ort_dnnl::DnnlSubgraph>(ort_dnnl::DnnlSubgraph(*graph_body_viewer.get()));
    subgraphs_.emplace(fused_node->Name(), std::move(dnnl_subgraph));

    //subgraph primitive
    auto dnnl_subgraph_primitive = std::make_unique<ort_dnnl::DnnlSubgraphPrimitive>(*subgraphs_[fused_node->Name()].get());
    {
      const auto& input_defs = fused_node->InputDefs();
      std::vector<std::string> onnx_input_names(input_defs.size());
      for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
        onnx_input_names[i] = input_defs[i]->Name();
      }
      dnnl_subgraph_primitive->SetOrderedInputs(std::move(onnx_input_names));
    }
    {
      const auto& output_defs = fused_node->OutputDefs();
      std::vector<std::string> onnx_output_names(output_defs.size());
      for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
        onnx_output_names[i] = output_defs[i]->Name();
      }
      dnnl_subgraph_primitive->SetOrderedOutputs(std::move(onnx_output_names));
    }

    subgraph_primitives_.emplace(fused_node->Name(), std::move(dnnl_subgraph_primitive));

    NodeComputeInfo compute_info;

    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = subgraph_primitives_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      ort_dnnl::DnnlSubgraphPrimitive* subgraph_primitive = reinterpret_cast<ort_dnnl::DnnlSubgraphPrimitive*>(state);

      const size_t subgraph_num_inputs = subgraph_primitive->GetOrderedInputs().size();
      const size_t subgraph_num_outputs = subgraph_primitive->GetOrderedOutputs().size();
      const size_t context_num_outputs = ort.KernelContext_GetOutputCount(context);
      const size_t context_num_inputs = ort.KernelContext_GetInputCount(context);

      std::unordered_map<std::string, ort_dnnl::OnnxTensorData> inputs;
      inputs.reserve(subgraph_num_inputs);
      for (size_t i = 0; i < context_num_inputs; i++) {
        auto input_name = subgraph_primitive->GetOrderedInputs()[i];
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
        auto* tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        auto shape = ort.GetTensorShape(tensor_info);
        const void* inputBuffer = ort.GetTensorData<void>(input_tensor);
        inputs.emplace(
            input_name,
            ort_dnnl::OnnxTensorData{
                ort_dnnl::OnnxTensorInfo{ort.GetTensorElementType(tensor_info), shape},
                const_cast<void*>(inputBuffer),
            });

        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }

      //lock each subgraph_primitive as multiple threads have shared memories
      {
        std::unique_lock<OrtMutex> lock(subgraph_primitive->GetMutex());
        subgraph_primitive->Compile(inputs);
        std::unordered_map<std::string, ort_dnnl::OnnxTensorData> outputs;
        outputs.reserve(subgraph_num_outputs);
        for (size_t i = 0; i < context_num_outputs; i++) {
          auto output_name = subgraph_primitive->GetOrderedOutputs()[i];
          auto output_md = subgraph_primitive->GetOutputInfo(output_name);
          auto output_shape = output_md.dims();
          auto* output_tensor =
              ort.KernelContext_GetOutput(context, i, output_shape.data(), output_shape.size());
          auto* tensor_info = ort.GetTensorTypeAndShape(output_tensor);
          auto shape = ort.GetTensorShape(tensor_info);
          void* output_buffer = ort.GetTensorMutableData<void>(output_tensor);
          outputs.emplace(output_name,
                          ort_dnnl::OnnxTensorData{
                              ort_dnnl::OnnxTensorInfo{ort.GetTensorElementType(tensor_info), shape},
                              output_buffer,
                          });
          ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
        }

        return subgraph_primitive->Predict(inputs, outputs);
      }
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }
  return Status::OK();
}

}  // namespace onnxruntime
