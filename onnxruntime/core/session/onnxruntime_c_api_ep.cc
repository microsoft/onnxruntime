#include "core/session/onnxruntime_c_api_ep.h"
#include "ort_apis_ep.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/session_options.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/ort_apis.h"

#include <fstream>
#include <iostream>

using namespace onnxruntime;

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetName, const OrtGraphViewer* graph, _Out_ const char** out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *out = graph_viewer->Name().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_IsConstantInitializer, const OrtGraphViewer* graph, const char* name, bool check_outer_scope, _Out_ bool* out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *out = graph_viewer->IsConstantInitializer(name, check_outer_scope);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetNodesIndexInTopologicalOrder, const OrtGraphViewer* graph, int execution_order, _Out_ const size_t** nodes_index_in_topological_order, _Out_ size_t* num_nodes) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  const std::vector<size_t>& nodes = graph_viewer->GetNodesInTopologicalOrder(static_cast<ExecutionOrder>(execution_order));
  *nodes_index_in_topological_order = nodes.data();
  *num_nodes = nodes.size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_IsSubgraph, const OrtGraphViewer* graph, _Out_ bool* out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *out = graph_viewer->IsSubgraph();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetParenNode, const OrtGraphViewer* graph, _Outptr_ const OrtNode** parent_node) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *parent_node = reinterpret_cast<const OrtNode*>(graph_viewer->ParentNode());
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetModelPath, const OrtGraphViewer* graph, _Outptr_ const void** model_path) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *model_path = reinterpret_cast<const void*>(&graph_viewer->ModelPath());
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetRequiredInputs, const OrtGraphViewer* graph, _Outptr_ const char*** input_names, _Out_ size_t* input_len) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  const auto& inputs = graph_viewer->GetInputs();
  *input_len = inputs.size();
  *input_names = new const char*[*input_len];  // Should be released by the caller using OrtGraphApis::ReleaseCharArray
  for (size_t i = 0; i < *input_len; i++) (*input_names)[i] = inputs[i]->Name().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetAllInputs, const OrtGraphViewer* graph, _Outptr_ const char*** input_names, _Out_ size_t* input_len) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  const auto& inputs = graph_viewer->GetInputsIncludingInitializers();
  *input_len = inputs.size();
  *input_names = new const char*[*input_len];   // Should be released by the caller using OrtGraphApis::ReleaseCharArray
  for (size_t i = 0; i < *input_len; i++) (*input_names)[i] = inputs[i]->Name().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetAllInitializers, const OrtGraphViewer* graph, _Outptr_ const char*** initializer_names, _Out_ size_t* initializer_len) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  const auto& initializers = graph_viewer->GetAllInitializedTensors();
  *initializer_len = initializers.size();
  *initializer_names = new const char*[*initializer_len];   // Should be released by the caller using OrtGraphApis::ReleaseCharArray
  int i = 0;
  for (const auto& [key, value] : initializers) (*initializer_names)[i++] = key.c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::ReleaseCharArray, const char** char_array) {
  if (!char_array) {
    return nullptr;
  }
  delete[] char_array;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetOrtNode, const OrtGraphViewer* graph, size_t node_index, _Outptr_ const OrtNode** node) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *node = reinterpret_cast<const OrtNode*>(graph_viewer->GetNode(node_index));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetNodesConsumingInput, const OrtGraphViewer* graph, const char* input_name, _Outptr_ const OrtNode*** consumers, _Out_ size_t* num_consumers) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  std::vector<const ::onnxruntime::Node*> consumer_nodes = graph_viewer->GetConsumerNodes(input_name);
  *num_consumers = consumer_nodes.size();
  *consumers = new const OrtNode* [*num_consumers];
  for (size_t i = 0; i < *num_consumers; i++) (*consumers)[i] = reinterpret_cast<const OrtNode*>(consumer_nodes[i]);

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::ReleaseOrtNodeArray, const OrtNode** nodes) {
  if (!nodes) {
    return nullptr;
  }
  delete[] nodes;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetNodeProducingOutput, const OrtGraphViewer* graph, const char* output_name, _Outptr_ const OrtNode** node) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *node = reinterpret_cast<const OrtNode*>(graph_viewer->GetProducerNode(output_name));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_NumberOfNodes, const OrtGraphViewer* graph, _Out_ int* num_nodes) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *num_nodes = graph_viewer->NumberOfNodes();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_MaxNodeIndex, const OrtGraphViewer* graph, _Out_ int* max_node_index) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *max_node_index = graph_viewer->MaxNodeIndex();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetOutputSize, const OrtGraphViewer* graph, _Out_ size_t* output_len) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *output_len = graph_viewer->GetOutputs().size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetIthOutputName, const OrtGraphViewer* graph, size_t i, _Outptr_ const char** out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *out = graph_viewer->GetOutputs()[i]->Name().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetIthOutputElemType, const OrtGraphViewer* graph, size_t i, _Out_ int32_t* out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *out = graph_viewer->GetOutputs()[i]->TypeAsProto()->tensor_type().elem_type();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetInitializerTensor, const OrtGraphViewer* graph, const char* initializer_name, _Outptr_ OrtTensorRef** out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  const onnx::TensorProto* initializer = nullptr;
  if (!graph_viewer->GetInitializedTensor(initializer_name, initializer)) {
    return nullptr; // TODO(leca): not return nullptr for this case?
  }
  *out = new OrtTensorRef();  // TODO(leca): other datatypes in the following switch
  (*out)->shape_len = initializer->dims_size();
  (*out)->shape = new int64_t [initializer->dims_size()];
  for (size_t i = 0; i < (*out)->shape_len; i++) {
    ((*out)->shape)[i] = initializer->dims(static_cast<int>(i));
  }

  (*out)->data_type = static_cast<ONNXTensorElementDataType>(initializer->data_type());
  // see utils::ConvertRawDataInTensorProto()
  switch (initializer->data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      (*out)->data_len = initializer->float_data_size();
      (*out)->data = reinterpret_cast<const char*>(initializer->float_data().data());
      break;
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_ReleaseInitializerTensor, OrtTensorRef* tensor) {
  if (!tensor) {
    return nullptr;
  }
  if (tensor->shape) {
    delete[] tensor->shape;
  }
  delete tensor;
  return nullptr;
}

static ONNXTensorElementDataType GetDataTypeFromTypeProto(const onnx::TypeProto* type) {  // onnxruntime\core\optimizer\transpose_optimization\ort_optimizer_api_impl.cc
  if (!type || !utils::HasTensorType(*type) || !utils::HasElementType(*type)) return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  return static_cast<ONNXTensorElementDataType>(type->tensor_type().elem_type());
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetValueInfo, const OrtGraphViewer* graph, const char* name, _Outptr_ OrtValueInfoRef** out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  const NodeArg* node_arg = graph_viewer->GetNodeArg(name);

  *out = new OrtValueInfoRef();
  const onnx::TypeProto* type = node_arg->TypeAsProto();
  (*out)->data_type = GetDataTypeFromTypeProto(type);
  const auto& dims = utils::TryGetShape(*type)->dim();
  (*out)->shape_len = dims.size();
  (*out)->shape = new int64_t [(*out)->shape_len];
  for (size_t i = 0; i < (*out)->shape_len; i++) ((*out)->shape)[i] = utils::HasDimValue(dims[static_cast<int>(i)]) ? dims[static_cast<int>(i)].dim_value() : -1;

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_ReleaseValueInfo, OrtValueInfoRef* value_info) {
  if (!value_info) {
    return nullptr;
  }
  if (value_info->shape) {
    delete[] value_info->shape;
  }
  delete value_info;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_SerializeToArray, const OrtGraphViewer* graph, _Out_ void** data, _Out_ size_t* data_size) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  Model model(graph_viewer->Name(), true, ModelMetaData(), PathString(),
#if defined(ORT_MINIMAL_BUILD)
    IOnnxRuntimeOpSchemaRegistryList(),
#else
    IOnnxRuntimeOpSchemaRegistryList({graph_viewer->GetSchemaRegistry()}),
#endif
    graph_viewer->DomainToVersionMap(), std::vector<onnx::FunctionProto>(), graph_viewer->GetGraph().GetLogger());
  onnx::ModelProto model_proto = model.ToProto();
  GraphViewerToProto(*graph_viewer, *model_proto.mutable_graph(), true, true, ExecutionOrder::PRIORITY_BASED);
  *data_size = model_proto.ByteSizeLong();
  *data = malloc(*data_size);
  model_proto.SerializeToArray(*data, static_cast<int>(*data_size));
  return nullptr;
}

struct SubGraphContext2 {
  std::unordered_set<std::string> output_args;
  std::unordered_map<std::string, const NodeArg*> inputs_and_initializers;
  std::unordered_map<std::string, const NodeArg*> manually_added_graph_inputs;
};

static std::string GetUniqueGraphName(const Graph& graph) {
  HashValue model_hash = 0;
  uint32_t hash[4] = {0, 0, 0, 0};

  auto hash_str = [&hash](const std::string& str) {
    MurmurHash3::x86_128(str.data(), gsl::narrow_cast<int32_t>(str.size()), hash[0], &hash);
  };

  // Hash all nodes' name
  for (int i = 0; i < graph.MaxNodeIndex(); ++i) {
    auto node = graph.GetNode(i);
    if (node == nullptr) {
      continue;
    }
    hash_str(node->Name());
  }

  model_hash = hash[0] | (uint64_t(hash[1]) << 32);

  return graph.Name() + "_" + std::to_string(model_hash);
}

static bool IsLocalValue(const Graph& graph,
                                             const std::string& name,
                                             const std::unordered_map<std::string, std::unique_ptr<SubGraphContext2>>& subgraph_context_map) {
  std::string unique_graph_name = GetUniqueGraphName(graph);
  if (subgraph_context_map.find(unique_graph_name) == subgraph_context_map.end()) {
    return false;
  }
  SubGraphContext2* context = subgraph_context_map.at(unique_graph_name).get();
  return context->output_args.find(name) != context->output_args.cend() ||
         context->inputs_and_initializers.find(name) != context->inputs_and_initializers.cend();
}

static bool IsInputInitializerOrOutput(const Graph& graph,
                                                           const std::string& name,
                                                           bool check_ancestors,
                                                           const std::unordered_map<std::string, std::unique_ptr<SubGraphContext2>>& subgraph_context_map) {
  const Graph* parent_graph = nullptr;
  return IsLocalValue(graph, name, subgraph_context_map) ||
         (check_ancestors && (parent_graph = graph.ParentGraph()) != nullptr &&
          IsInputInitializerOrOutput(*parent_graph, name, check_ancestors, subgraph_context_map));
}

static bool IsOuterScopeValue(const Graph& graph,
                                                  const std::string& name,
                                                  const std::unordered_map<std::string, std::unique_ptr<SubGraphContext2>>& subgraph_context_map) {
  const Graph* parent_graph = nullptr;
  return (parent_graph = graph.ParentGraph()) != nullptr &&
         IsInputInitializerOrOutput(*parent_graph, name, true, subgraph_context_map);
}

static void BuildSubGraphContext(const Graph& graph, std::unordered_map<std::string, std::unique_ptr<SubGraphContext2>>& subgraph_context_map) {
  // Iterate all the nodes and recurse into inner most subgraph first
  for (int i = 0; i < graph.MaxNodeIndex(); ++i) {
    auto node = graph.GetNode(i);
    if (node == nullptr) {
      continue;
    }

    auto subgraph_map = node->GetAttributeNameToSubgraphMap();
    for (auto& entry : subgraph_map) {
      const Graph* subgraph = entry.second;
      BuildSubGraphContext(*subgraph, subgraph_context_map);
    }
  }

  std::string unique_graph_name = GetUniqueGraphName(graph);

  // Subgraph context has been built before, no need to do it again
  if (subgraph_context_map.find(unique_graph_name) != subgraph_context_map.end()) {
    return;
  }

  subgraph_context_map.emplace(unique_graph_name, std::make_unique<SubGraphContext2>());
  SubGraphContext2* context = subgraph_context_map.at(unique_graph_name).get();

  // Collect all nodes' outputs and nodes' name
  for (int i = 0; i < graph.MaxNodeIndex(); ++i) {
    auto node = graph.GetNode(i);
    if (node == nullptr) {
      continue;
    }

    for (const auto& output : node->OutputDefs()) {
      context->output_args.insert(output->Name());
    }
  }

  // Go thru all node's inputs
  for (int i = 0; i < graph.MaxNodeIndex(); ++i) {
    auto node = graph.GetNode(i);
    if (node == nullptr) {
      continue;
    }

    for (const auto& input : node->InputDefs()) {
      if (context->output_args.find(input->Name()) != context->output_args.end()) {
        continue;
      }
      // This input arg is not the output of another node so must come from either a graph input or an initializer.
      context->inputs_and_initializers[input->Name()] = input;
    }
  }
}

static void SetGraphOuterScopeValuesAndInputs(Graph& graph_build,
                                                                  const Graph& graph,
                                                                  std::unordered_map<std::string, std::unique_ptr<SubGraphContext2>>& subgraph_context_map) {
  // Iterate all the nodes and recurse into inner most subgraph first for both newly built graph and original graph
  for (int i = 0; i < graph_build.MaxNodeIndex(); ++i) {
    auto graph_build_node = graph_build.GetNode(i);
    if (graph_build_node == nullptr) {
      continue;
    }

    auto graph_build_map = graph_build_node->GetAttributeNameToMutableSubgraphMap();
    std::unordered_map<std::string, gsl::not_null<const Graph*>> subgraph_map;
    const Node* graph_node = nullptr;

    // Find corresponding original graph node's subgraphs
    for (int j = 0; j < graph.MaxNodeIndex(); ++j) {
      if (graph.GetNode(j) && graph.GetNode(j)->Name() == graph_build_node->Name()) {
        graph_node = graph.GetNode(j);
        subgraph_map = graph_node->GetAttributeNameToSubgraphMap();
        break;
      }
    }

    for (auto& entry : graph_build_map) {
      auto attr_name = entry.first;
      Graph* subgraph_build = entry.second;
      if (subgraph_map.find(attr_name) != subgraph_map.end()) {
        // recurse into subgraph
        const Graph* subgraph = subgraph_map.at(attr_name);
        SetGraphOuterScopeValuesAndInputs(*subgraph_build, *subgraph, subgraph_context_map);
      }
    }
  }

  // Start from the inner most subgraph first and check whether its outer scope values are existed in the
  // newly built graph. If not, we need to add those outer scope values as explicit inputs to the top-level
  // of newly built graph.
  if (graph_build.ParentNode()) {
    auto top_level_graph = &graph_build;
    while (top_level_graph->MutableParentGraph()) {
      top_level_graph = top_level_graph->MutableParentGraph();
    }
    std::string unique_graph_name = GetUniqueGraphName(*top_level_graph);
    if (subgraph_context_map.find(unique_graph_name) == subgraph_context_map.end()) {
      return;
    }

    SubGraphContext2* context = subgraph_context_map.at(unique_graph_name).get();

    // Iterate all the implicit inputs to set outer scope value for the newly built subgraph
    for (const auto& input : graph.ParentNode()->ImplicitInputDefs()) {
//      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] \t" << input->Name();

      // The node arg in parent node's implicit inputs could be used for parent node's other subgraph, for example
      // "If" op has two subgraphs. So we need to make sure that the node arg is used in current subgraph only.
      // (GetNodeArg searches for specific node arg in all node args in the graph)
      if (graph_build.GetNodeArg(input->Name())) {
        graph_build.AddOuterScopeNodeArg(input->Name());
//        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] \t" << input->Name() << " is used in this subgraph";

        if (context &&
            (context->manually_added_graph_inputs.find(input->Name()) != context->manually_added_graph_inputs.end())) {
//          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] \t" << input->Name() << " is already been added as an explicit input to graph";
          continue;
        }

        // Handle the case where this outer scope value is not existed in any outer scope levels of the
        // newly built graph (the newly built graph is the subgraph of the original graph). Need to add
        // the outer scope value as an explicit input to the top-level of newly built graph.
        if (!IsOuterScopeValue(graph_build, input->Name(), subgraph_context_map)) {
          const auto& name = input->Name();
          auto graph_inputs_including_initializers = top_level_graph->GetInputsIncludingInitializers();
          auto added_graph_input = std::find_if(graph_inputs_including_initializers.begin(),
                                                graph_inputs_including_initializers.end(),
                                                [&name](const NodeArg* entry) { return entry->Name() == name; });

          if (added_graph_input == graph_inputs_including_initializers.end()) {
            if (context) {
              auto type_proto = std::make_unique<ONNX_NAMESPACE::TypeProto>();
              type_proto->CopyFrom(*(input->TypeAsProto()));
              auto& n_input = top_level_graph->GetOrCreateNodeArg(name, type_proto.get());
              context->manually_added_graph_inputs[n_input.Name()] = &n_input;
//              LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] \t" << n_input.Name() << " is added as an explicit input into the newly built graph";
            }
          }
        }
      }
    }
  }
}

static void SetAllGraphInputs(Graph& graph, std::unordered_map<std::string, std::unique_ptr<SubGraphContext2>>& subgraph_context_map) {
  // If ORT TRT doesn't manully set graph input in TensorrtExecutionProvider::SetGraphOuterScopeValuesAndInputs(),
  // Graph::Resolve() will help set graph inputs in Graph::SetGraphInputsOutputs(), so no need to set graph inputs here.
  std::string unique_graph_name = GetUniqueGraphName(graph);
  if (subgraph_context_map.find(unique_graph_name) == subgraph_context_map.end() ||
      subgraph_context_map[unique_graph_name].get()->manually_added_graph_inputs.size() == 0) {
    return;
  }

  SubGraphContext2* context = subgraph_context_map[unique_graph_name].get();
  std::vector<const NodeArg*> graph_inputs_including_initializers;
  std::unordered_set<std::string> graph_inputs_including_initializers_set;

  for (const auto& entry : context->inputs_and_initializers) {
    graph_inputs_including_initializers.push_back(entry.second);
    graph_inputs_including_initializers_set.insert(entry.first);
  }

  for (const auto& entry : context->manually_added_graph_inputs) {
    if (graph_inputs_including_initializers_set.find(entry.first) == graph_inputs_including_initializers_set.end()) {
      graph_inputs_including_initializers.push_back(entry.second);
      graph_inputs_including_initializers_set.insert(entry.first);
    }
  }

  for (const auto& node_arg : graph.GetInputsIncludingInitializers()) {
    if (graph_inputs_including_initializers_set.find(node_arg->Name()) == graph_inputs_including_initializers_set.end()) {
      graph_inputs_including_initializers.push_back(node_arg);
      graph_inputs_including_initializers_set.insert(node_arg->Name());
    }
  }

  graph.SetInputs(graph_inputs_including_initializers);
}

/*
 * Given a graph, get the corresponding model and serialize it to disk.
 */
ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_DumpOnnxModel,
                    const OrtGraph* graph,
                    const char* onnx_model_path) {
  const ::onnxruntime::Graph* internal_graph = reinterpret_cast<const ::onnxruntime::Graph*>(graph);
  auto model = &(internal_graph->GetModel());

  // Two options to generate model proto:
  //   1. directly call model->ToProto()
  //   2. new model ---> model->ToProto ---> update graph proto in model proto with GraphViewerToProto()
  //
  // TODO: (Chi) Need more thinking on which to choose

  // option 1
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto = std::make_unique<ONNX_NAMESPACE::ModelProto>(model->ToProto());

  // option 2
  //auto model_proto = model->ToProto();
  //graph->ToProto(*model_proto->mutable_graph(), true, true);
  //model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::fstream dump(onnx_model_path, std::ios::out | std::ios::trunc | std::ios::binary);
  model_proto->SerializeToOstream(&dump);
  //LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Dumped " + ctx_model_path;
  return nullptr;
}

/* Construct an "EP Context" graph if the given ep_context_graph graph is empty, otherwise:
 *   1. if the given node name can't be found in the graph, add an new "EP Context" node to the existing graph
 *   2. if the node is already existed, update the node attributes only
 */
ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_CreateOrUpdateEpCtxGraph,
                    const OrtGraphViewer* graph,
                    const char* node_name,
                    const int64_t main_context,
                    const int64_t embed_mode,
                    const char* cache_path,
                    char* cache_data,
                    size_t size,
                    const char* const* extra_attr_keys,
                    const char* const* extra_attr_values,
                    size_t extra_attr_num,
                    _Outptr_ OrtGraph** ep_context_graph) {

  const std::string EPCONTEXT_OP = "EPContext";
  const std::string MAIN_CONTEXT = "main_context";
  const std::string EMBED_MODE = "embed_mode";
  const std::string EP_CACHE_CONTEXT = "ep_cache_context";
  const std::string ONNX_MODEL_FILENAME = "onnx_model_filename";
  const std::string EPCONTEXT_OP_DOMAIN = "com.microsoft";
  const std::string EPCONTEXT_WARNING =
    "It's suggested to set the ORT graph optimization level to 0 and  \
                                              make \"embed_mode\" to 0 (\"ep_cache_context\" is the cache path)\
                                              for the best model loading time";

  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  ::onnxruntime::Graph* graph_build;

  if (!graph_viewer && !(*ep_context_graph)) return nullptr;

  std::unordered_map<std::string, std::string> attr_keys_values;
  for (size_t i = 0; i < extra_attr_num; i++) {
    attr_keys_values[extra_attr_keys[i]] = extra_attr_values[i];
  }

  // Create a new graph or use the existing one
  if (*ep_context_graph == nullptr) {
    Model* model_build = new Model (graph_viewer->Name(), true, ModelMetaData(), PathString(),
#if !defined(ORT_MINIMAL_BUILD)
                                   IOnnxRuntimeOpSchemaRegistryList({graph_viewer->GetSchemaRegistry()}), graph_viewer->DomainToVersionMap(),
#else
                                   IOnnxRuntimeOpSchemaRegistryList(), graph_viewer->DomainToVersionMap(),
#endif  // ORT_MINIMAL_BUILD
                                   std::vector<ONNX_NAMESPACE::FunctionProto>(), graph_viewer->GetGraph().GetLogger());
    graph_build = &(model_build->MainGraph());
    *ep_context_graph = reinterpret_cast<OrtGraph*>(graph_build);
  } else {
    graph_build = reinterpret_cast<::onnxruntime::Graph*>(*ep_context_graph);
  }

  // Get graph inputs and outputs
  std::vector<onnxruntime::NodeArg*> inputs, outputs;
  if (graph_viewer) {
    for (auto input : graph_viewer->GetInputs()) {
      auto& n_input = graph_build->GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }

    for (auto output : graph_viewer->GetOutputs()) {
      auto& n_output = graph_build->GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
  }

  // locate specific node if any
  auto get_node_index = [&](Graph* graph, const char* node_name) -> size_t {
    std::string name = node_name;
    for (auto& node : graph->Nodes()) {
      if (name == node.Name()) {
        return node.Index();
      }
    }
    // return impossible value to indicate the node is not existed
    return std::numeric_limits<size_t>::max();
  };
  size_t node_idx = get_node_index(graph_build, node_name);
  bool node_existed = node_idx != std::numeric_limits<size_t>::max() ? true : false;

  // Create or get EP context node attributes
  auto new_node_attributes = NodeAttributes(); // using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>
  NodeAttributes* node_attributes;
  if (node_existed) {
    node_attributes = &graph_build->GetNode(node_idx)->GetMutableAttributes();
  } else {
    new_node_attributes.reserve(3 + extra_attr_num);
    node_attributes = &new_node_attributes;
  }
  std::unique_ptr<ONNX_NAMESPACE::AttributeProto> attr_0 = std::make_unique<ONNX_NAMESPACE::AttributeProto>(); // main_context
  std::unique_ptr<ONNX_NAMESPACE::AttributeProto> attr_1 = std::make_unique<ONNX_NAMESPACE::AttributeProto>(); // embed_mode
  std::unique_ptr<ONNX_NAMESPACE::AttributeProto> attr_2 = std::make_unique<ONNX_NAMESPACE::AttributeProto>(); // ep_cache_context

  std::string cache_data_str = "";
  std::string cache_path_str = cache_path;

  // main_context
  attr_0->set_name(MAIN_CONTEXT);
  attr_0->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_0->set_i(main_context);

  // embed_mode
  attr_1->set_name(EMBED_MODE);
  attr_1->set_type(onnx::AttributeProto_AttributeType_INT);
  attr_1->set_i(embed_mode);

  // ep_cache_context
  attr_2->set_name(EP_CACHE_CONTEXT);
  attr_2->set_type(onnx::AttributeProto_AttributeType_STRING);
  if (embed_mode) {
    if (size > 0) {
      cache_data_str.assign(cache_data, size);
    }
    attr_2->set_s(cache_data_str);
    //LOGS_DEFAULT(WARNING) << EPCONTEXT_WARNING;
  } else {
    attr_2->set_s(cache_path_str);
  }

  (*node_attributes)[MAIN_CONTEXT] = *attr_0;
  (*node_attributes)[EMBED_MODE] = *attr_1;
  (*node_attributes)[EP_CACHE_CONTEXT] = *attr_2;

  // other attributes
  std::unordered_map<std::string, std::string>::iterator it;
  for (it = attr_keys_values.begin(); it != attr_keys_values.end(); ++it) {
    std::string key = it->first;
    std::string value = it->second;
    if (key == ONNX_MODEL_FILENAME) value = std::filesystem::path(value).filename().string();

    std::unique_ptr<ONNX_NAMESPACE::AttributeProto> attr = std::make_unique<ONNX_NAMESPACE::AttributeProto>();
    attr->set_name(key);
    attr->set_type(onnx::AttributeProto_AttributeType_STRING);
    attr->set_s(value);
    (*node_attributes)[key] = *attr;
  }

  if (!node_existed && graph_viewer) {
    std::string name = node_name;
    graph_build->AddNode(name, EPCONTEXT_OP, "", inputs, outputs, node_attributes, EPCONTEXT_OP_DOMAIN);
  }

  common::Status status = graph_build->Resolve();
  if (status != Status::OK()) return onnxruntime::ToOrtStatus(status);

  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_GetSubGraph, const OrtGraphViewer* graph, const int node_num, const size_t* node_indices, _Outptr_ const OrtGraphViewer** subgraph) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph_viewer->GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }
  // NOTE!!: cannot use unique_ptr here, otherwise when this function exits, sub_graph_viewer->graph_->graph_proto_, which is from model_build->model_proto_, will be nullptr.
  // Pay special attention when Graph object is releasing. We need to release model_build seperately then.
  Model* model_build = new Model (graph_viewer->Name(), true, ModelMetaData(), PathString(),
#if !defined(ORT_MINIMAL_BUILD)
                                   IOnnxRuntimeOpSchemaRegistryList({graph_viewer->GetSchemaRegistry()}), graph_viewer->DomainToVersionMap(),
#else
                                   IOnnxRuntimeOpSchemaRegistryList(), graph_viewer->DomainToVersionMap(),
#endif  // ORT_MINIMAL_BUILD
                                   std::vector<ONNX_NAMESPACE::FunctionProto>(), graph_viewer->GetGraph().GetLogger());

  auto& graph_build = model_build->MainGraph();
  bool has_control_flow_op = false;

  std::vector<std::string> subgraph_output_names;
  const std::vector<NodeIndex>& node_index = graph_viewer->GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
  for(int i = 0; i < node_num; i++) {
    const auto& node = graph_viewer->GetNode(node_index[node_indices[i]]);
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node->InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
      const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
      if (graph_viewer->GetInitializedTensor(input->Name(), initializer)) {
        const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
        if (!graph_build.GetInitializedTensor(input->Name(), subgraph_initializer)) {
          graph_build.AddInitializedTensor(*(initializer));
        }
      }
    }
    for (auto input : node->ImplicitInputDefs()) {
      const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
      if (graph_viewer->GetInitializedTensor(input->Name(), initializer)) {
        const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
        if (!graph_build.GetInitializedTensor(input->Name(), subgraph_initializer)) {
          graph_build.AddInitializedTensor(*(initializer));
        }
      }
    }
    for (auto output : node->OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
      const auto name = output->Name();
      if (graph_output_names.find(name) != graph_output_names.end()) {
        subgraph_output_names.push_back(name);
      }
    }

    std::unordered_set<std::string> control_flow_op_set = {"If", "Loop", "Scan"};
    if (control_flow_op_set.find(node->OpType()) != control_flow_op_set.end()) {
      has_control_flow_op = true;
    }

    // If the node has subgraph, it's possible that the ORT graph of that subgraph and the GraphProto in the node attributes are not in sync because of graph optimization.
    // Therefore, we need to force GraphProto attributes to be updated in order to get the valid GraphProto.
    if (node->GetAttributes().size() > 0) {
      auto node_proto = std::make_unique<ONNX_NAMESPACE::NodeProto>();
      // we need to update any GraphProto attributes for subgraphs so that any changes made by things
      // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
      node->ToProto(*node_proto, /* update_subgraphs */ true);
      const int num_attributes = node_proto->attribute_size();
      NodeAttributes node_attributes;
      node_attributes.reserve(num_attributes);

      for (int ii = 0; ii < num_attributes; ++ii) {
        auto& attr = node_proto->attribute(ii);
        node_attributes.emplace(attr.name(), attr);
      }

      // The GraphProto attributes are the updated ones.
      graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node_attributes, node->Domain());
    } else {
      // The GraphProto attributes are the original ones.
      graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
    }
  }

  // TODO:yang
  // Only if the newly built graph has control flow op as well as it has parent node,
  // it needs to handle outer scope values before calling graph.Resolve().
  std::unordered_map<std::string, std::unique_ptr<SubGraphContext2>> subgraph_context_map;
  if (has_control_flow_op && graph_viewer->ParentNode()) {
  //   LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Handle outer scope values for the subgraph " << graph_build.Name();
     BuildSubGraphContext(graph_build, subgraph_context_map);
     SetGraphOuterScopeValuesAndInputs(graph_build, graph_viewer->GetGraph(), subgraph_context_map);
     SetAllGraphInputs(graph_build, subgraph_context_map);
  }

  common::Status status = graph_build.Resolve();
  if (status != Status::OK()) return onnxruntime::ToOrtStatus(status);

  // Add parent graph output to the subgraph
  int i = 0;
  std::vector<const NodeArg*> subgraph_outputs;
  subgraph_outputs.resize(subgraph_output_names.size());
  for (auto& name : subgraph_output_names) {
    auto output_arg = graph_viewer->GetNodeArg(name);
    auto& subgraph_output_arg = graph_build.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
    subgraph_outputs[i] = &subgraph_output_arg;
    ++i;
  }
  auto& graph_build_outputs = graph_build.GetOutputs();
  subgraph_outputs.insert(subgraph_outputs.begin(), graph_build_outputs.begin(), graph_build_outputs.end());
  graph_build.SetOutputs(graph_build_outputs);
  status = graph_build.Resolve();
  if (status != Status::OK()) return onnxruntime::ToOrtStatus(status);

  // TODO(leca): Maybe we should just return graph_build in the form of OrtGraph, so that we can reuse OrtGraph_ReleaseGraph
  auto sub_graph_viewer = std::make_unique<GraphViewer>(graph_build);
  *subgraph = reinterpret_cast<const OrtGraphViewer*>(sub_graph_viewer.release());
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_ReleaseGraph, const OrtGraph* ort_graph) {
  if (ort_graph) {
    const ::onnxruntime::Graph* graph = reinterpret_cast<const ::onnxruntime::Graph*>(ort_graph);
    delete &(graph->GetModel());
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_ReleaseGraphViewer, const OrtGraphViewer* graph) {
  if (graph) {
    const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
    delete &(graph_viewer->GetGraph()).GetModel();
    delete graph_viewer;
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_IsSameGraph, const OrtGraphViewer* graph1, const OrtGraphViewer* graph2, bool* is_same) {
  const ::onnxruntime::GraphViewer* graph_viewer1 = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph1);
  const ::onnxruntime::GraphViewer* graph_viewer2 = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph2);
  *is_same = (&(graph_viewer1->GetGraph()) == &(graph_viewer2->GetGraph()));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetName, const OrtNode* node, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->Name().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetDescription, const OrtNode* node, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->Description().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetDomain, const OrtNode* node, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->Domain().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_SinceVersion, const OrtNode* node, _Out_ int* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->SinceVersion();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetExecutionProviderType, const OrtNode* node, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetExecutionProviderType().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetOpType, const OrtNode* node, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->OpType().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetImplicitInputSize, const OrtNode* node, _Out_ size_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->ImplicitInputDefs().size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetIthImplicitInputName, const OrtNode* node, size_t i, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  assert(i < n->ImplicitInputDefs().size());
  *out = n->ImplicitInputDefs()[i]->Name().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetNumInputs, const OrtNode* node, _Out_ size_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->InputDefs().size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetIthInputName, const OrtNode* node, size_t i, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  assert(i < n->InputDefs().size());
  *out = n->InputDefs()[i]->Name().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetNumOutputs, const OrtNode* node, _Out_ size_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->OutputDefs().size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetIthOutputName, const OrtNode* node, size_t i, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  assert(i < n->OutputDefs().size());
  if (n->OutputDefs()[i]->Exists()) {
    *out = n->OutputDefs()[i]->Name().c_str();
    return nullptr;
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetIndex, const OrtNode* node, _Out_ size_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->Index();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeNames, const OrtNode* node, _Out_ const char*** names, _Out_ size_t* num) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *num = n->GetAttributes().size();
  *names = new const char* [*num];
  int i = 0;
  for (const auto& [k, v] : n->GetAttributes()) {
    (*names)[i++] = k.c_str();
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeSize, const OrtNode* node, _Out_ size_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeType, const OrtNode* node, const char* attribute, _Out_ int* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = static_cast<int>(n->GetAttributes().at(attribute).type());
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeKeyCount, const OrtNode* node, const char* key, _Out_ size_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().count(key);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeIntSize, const OrtNode* node, const char* key, _Out_ int* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).ints_size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeFloatSize, const OrtNode* node, const char* key, _Out_ int* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).floats_size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeStringSize, const OrtNode* node, const char* key, _Out_ int* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).strings_size();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeIthInt, const OrtNode* node, const char* key, int i, _Out_ int64_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).ints(i);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeIthFloat, const OrtNode* node, const char* key, int i, _Out_ float* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).floats(i);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeIthStr, const OrtNode* node, const char* key, int i, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).strings(i).c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeIthStrWithSize, const OrtNode* node, const char* key, int i, _Outptr_ const char** out, _Outptr_ size_t* size) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *size = n->GetAttributes().at(key).strings(i).size();
  *out = n->GetAttributes().at(key).strings(i).c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeStr, const OrtNode* node, const char* key, _Outptr_ const char** out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).s().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeStrWithSize, const OrtNode* node, const char* key,  _Outptr_ const char** out, _Outptr_ size_t* size) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *size = n->GetAttributes().at(key).s().size();
  *out = n->GetAttributes().at(key).s().c_str();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeInt, const OrtNode* node, const char* key, _Out_ int64_t* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).i();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetAttributeFloat, const OrtNode* node, const char* key, _Out_ float* out) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  *out = n->GetAttributes().at(key).f();
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtNode_GetSubgraphs, const OrtNode* node, _Outptr_ const OrtGraphViewer*** subgraphs, _Out_ size_t* num) {
  const ::onnxruntime::Node* n = reinterpret_cast<const ::onnxruntime::Node*>(node);
  std::vector<gsl::not_null<const Graph*>> subg = n->GetSubgraphs();
  *num = subg.size();
  *subgraphs = new const OrtGraphViewer* [*num];
  for (size_t i = 0; i < *num; i++) {
    const ::onnxruntime::GraphViewer* graph_viewer = new const ::onnxruntime::GraphViewer(*subg[i]);
    (*subgraphs)[i] = reinterpret_cast<const OrtGraphViewer*>(graph_viewer);
  }
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtGraphApis::OrtFreeMem, void* p) {
  if (p) {
    free(p);
  }
  return nullptr;
}

static constexpr OrtGraphApi ort_graph_api = {
    &OrtGraphApis::OrtGraph_GetName,
    &OrtGraphApis::OrtGraph_IsConstantInitializer,
    &OrtGraphApis::OrtGraph_GetNodesIndexInTopologicalOrder,
    &OrtGraphApis::OrtGraph_IsSubgraph,
    &OrtGraphApis::OrtGraph_GetParenNode,
    &OrtGraphApis::OrtGraph_GetModelPath,
    &OrtGraphApis::OrtGraph_GetRequiredInputs,
    &OrtGraphApis::OrtGraph_GetAllInputs,
    &OrtGraphApis::OrtGraph_GetAllInitializers,
    &OrtGraphApis::ReleaseCharArray,
    &OrtGraphApis::OrtGraph_GetOrtNode,
    &OrtGraphApis::OrtGraph_GetNodesConsumingInput,
    &OrtGraphApis::ReleaseOrtNodeArray,
    &OrtGraphApis::OrtGraph_GetNodeProducingOutput,
    &OrtGraphApis::OrtGraph_NumberOfNodes,
    &OrtGraphApis::OrtGraph_MaxNodeIndex,
    &OrtGraphApis::OrtGraph_GetOutputSize,
    &OrtGraphApis::OrtGraph_GetIthOutputName,
    &OrtGraphApis::OrtGraph_GetIthOutputElemType,
    &OrtGraphApis::OrtGraph_GetInitializerTensor,
    &OrtGraphApis::OrtGraph_ReleaseInitializerTensor,
    &OrtGraphApis::OrtGraph_GetValueInfo,
    &OrtGraphApis::OrtGraph_ReleaseValueInfo,
    &OrtGraphApis::OrtGraph_SerializeToArray,
    &OrtGraphApis::OrtGraph_DumpOnnxModel,
    &OrtGraphApis::OrtGraph_CreateOrUpdateEpCtxGraph,
    &OrtGraphApis::OrtGraph_GetSubGraph,
    &OrtGraphApis::OrtGraph_ReleaseGraph,
    &OrtGraphApis::OrtGraph_ReleaseGraphViewer,
    &OrtGraphApis::OrtGraph_IsSameGraph,
    &OrtGraphApis::OrtNode_GetName,
    &OrtGraphApis::OrtNode_GetDescription,
    &OrtGraphApis::OrtNode_GetDomain,
    &OrtGraphApis::OrtNode_SinceVersion,
    &OrtGraphApis::OrtNode_GetExecutionProviderType,
    &OrtGraphApis::OrtNode_GetOpType,
    &OrtGraphApis::OrtNode_GetImplicitInputSize,
    &OrtGraphApis::OrtNode_GetIthImplicitInputName,
    &OrtGraphApis::OrtNode_GetNumInputs,
    &OrtGraphApis::OrtNode_GetIthInputName,
    &OrtGraphApis::OrtNode_GetNumOutputs,
    &OrtGraphApis::OrtNode_GetIthOutputName,
    &OrtGraphApis::OrtNode_GetIndex,
    &OrtGraphApis::OrtNode_GetAttributeNames,
    &OrtGraphApis::OrtNode_GetAttributeSize,
    &OrtGraphApis::OrtNode_GetAttributeType,
    &OrtGraphApis::OrtNode_GetAttributeKeyCount,
    &OrtGraphApis::OrtNode_GetAttributeIntSize,
    &OrtGraphApis::OrtNode_GetAttributeFloatSize,
    &OrtGraphApis::OrtNode_GetAttributeStringSize,
    &OrtGraphApis::OrtNode_GetAttributeIthInt,
    &OrtGraphApis::OrtNode_GetAttributeIthFloat,
    &OrtGraphApis::OrtNode_GetAttributeIthStr,
    &OrtGraphApis::OrtNode_GetAttributeIthStrWithSize,
    &OrtGraphApis::OrtNode_GetAttributeStr,
    &OrtGraphApis::OrtNode_GetAttributeStrWithSize,
    &OrtGraphApis::OrtNode_GetAttributeInt,
    &OrtGraphApis::OrtNode_GetAttributeFloat,
    &OrtGraphApis::OrtNode_GetSubgraphs,
    &OrtGraphApis::OrtFreeMem,
};

ORT_API(const OrtGraphApi*, OrtGraphApis::GetGraphApi, uint32_t) {
  // No constraints on the API version yet.
  return &ort_graph_api;
}
