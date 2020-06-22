// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include "core/graph/onnx_protobuf.h"

#include "tensorrt_execution_provider.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/framework/memcpy.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_fence.h"
#include "core/platform/env.h"
#include "core/common/safeint.h"
#include "core/common/status.h"
#include "onnx/shape_inference/implementation.h"
#include "cuda_runtime_api.h"
#include "gsl/gsl"
#include "core/graph/model.h"
#include "core/providers/cuda/gpu_data_transfer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  Status st;
};
}  // namespace
namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .ExecQueueId(kCudaStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .ExecQueueId(kCudaStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static Status RegisterTensorrtKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }
  return Status::OK();
}

KernelRegistryAndStatus GetTensorrtKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterTensorrtKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry> TensorrtExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = onnxruntime::GetTensorrtKernelRegistry();
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& GetTensorrtLogger() {
  static TensorrtLogger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
  return trt_logger;
}

TensorrtExecutionProvider::TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTensorrtExecutionProvider}, device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));

  DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault,
       [](int id) {
         return onnxruntime::make_unique<CUDAAllocator>(id, TRT);
       },
       std::numeric_limits<size_t>::max()});
  allocator_ = CreateAllocator(default_memory_info, device_id_);

  InsertAllocator(allocator_);

  DeviceAllocatorRegistrationInfo pinned_allocator_info(
      {OrtMemTypeCPUOutput,
       [](int) {
         return onnxruntime::make_unique<CUDAPinnedAllocator>(0, TRT_PINNED);
       },
       std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(pinned_allocator_info, device_id_));

  // Get environment variables
  const Env& env_instance = Env::Default();
  const std::string max_partition_iterations_env = env_instance.GetEnvironmentVar(tensorrt_env_vars::kMaxPartitionIterations);
  if (!max_partition_iterations_env.empty()) {
    max_partition_iterations_ = std::stoi(max_partition_iterations_env);
  }

  const std::string min_subgraph_size_env = env_instance.GetEnvironmentVar(tensorrt_env_vars::kMinSubgraphSize);
  if (!min_subgraph_size_env.empty()) {
    min_subgraph_size_ = std::stoi(min_subgraph_size_env);
  }

  const std::string max_workspace_size_env = env_instance.GetEnvironmentVar(tensorrt_env_vars::kMaxWorkspaceSize);
  if (!max_workspace_size_env.empty()) {
    max_workspace_size_ = std::stoull(max_workspace_size_env);
  }

  const std::string fp16_enable_env = env_instance.GetEnvironmentVar(tensorrt_env_vars::kFP16Enable);
  if (!fp16_enable_env.empty()) {
    fp16_enable_ = (std::stoi(fp16_enable_env) == 0 ? false : true);
  }

  const std::string dump_subgraphs_env = env_instance.GetEnvironmentVar(tensorrt_env_vars::kDumpSubgraphs);
  if (!dump_subgraphs_env.empty()) {
    dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
  }
}

TensorrtExecutionProvider::~TensorrtExecutionProvider() {}

AllocatorPtr TensorrtExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

std::unique_ptr<onnxruntime::IDataTransfer> TensorrtExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<onnxruntime::GPUDataTransfer>();
}

// Convert GraphViewer graph to GraphProto
void ToGraphProtoInternal(const onnxruntime::GraphViewer& graph, ONNX_NAMESPACE::GraphProto& graph_proto) {
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
    const gsl::not_null<NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{graph.GetNode(node_idx)};
    p_node->ToProto(*node_proto);
  }
}

// Check if cycle exists in the graph after partitioning
bool FindCycleHelper(int i, const std::list<int>* adjacency_map,
                     bool visited[], bool* st, std::vector<int>& cycles) {
  if (!visited[i]) {
    visited[i] = true;
    st[i] = true;
    for (auto iter = adjacency_map[i].begin(); iter != adjacency_map[i].end(); ++iter) {
      if (!visited[*iter] && FindCycleHelper(*iter, adjacency_map, visited, st, cycles)) {
        cycles.push_back(*iter);
        return true;
      } else if (st[*iter]) {
        cycles.push_back(*iter);
        return true;
      }
    }
  }
  st[i] = false;
  return false;
}

// Remove nodes with empty shape (for example [1, 0]) because TensorRT 7 doens't support empty shape
SubGraphCollection_t RemoveEmptyShapeNodes(const onnxruntime::GraphViewer& graph) {
  // Here only NonZero, NonMaxSuppression and TopK related empty shape nodes are removed, particularly for RCNN models.
  // TODO: Remove the code if TensorRT fixed the issue in the future release, or find a better generic way here to work around
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  const std::vector<std::string> exclude_dim_names{"NonZero", "NonMaxSuppression", "TopK"};
  SubGraphCollection_t parser_nodes_vector = {{{}, false}};
  std::vector<size_t> nodes_vector(node_index.size());
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  for (const auto& index : nodes_vector) {
    // Check if node has empty input shape
    const auto& node = graph.GetNode(node_index[index]);
    bool exclude_node = false;
    for (const auto& input : node->InputDefs()) {
      const auto& input_shape = input->Shape();
      if (input_shape) {
        for (const auto& dim : input_shape->dim()) {
          std::string dim_name = dim.dim_param();
          if (!dim_name.empty()) {
            for (const auto& exclude : exclude_dim_names) {
              if (dim_name.find(exclude) != std::string::npos) {
                exclude_node = true;
                break;
              }
            }
            if (exclude_node) {
              break;
            }
          }
        }
      }

      if (exclude_node) {
        break;
      }
    }

    // Remove the node with empty input shape
    if (!exclude_node) {
      parser_nodes_vector.back().first.push_back(index);
    } else if (!parser_nodes_vector.back().first.empty()) {
      parser_nodes_vector.push_back({{}, false});
    }
  }

  return parser_nodes_vector;
}

std::unique_ptr<IndexedSubGraph> TensorrtExecutionProvider::GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index, const onnxruntime::GraphViewer& graph) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.first.size());
  for (const auto& index : graph_nodes_index.first) {
    node_set.insert(node_index[index]);
  }

  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }

  // Find inputs and outputs of the subgraph
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add, graph_outputs_to_add;
  std::unordered_set<const NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  for (const auto& index : graph_nodes_index.first) {
    sub_graph->nodes.push_back(node_index[index]);
    const auto& node = graph.GetNode(node_index[index]);
    for (const auto& input : node->InputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    // For output searching, there are two special cases,
    // One is, if node's OutputEdges are more than its outputs, meaning certain output is used more than once,
    // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
    // to the output list
    // The other one is, if subgraph's node output is parent graph's output. the node output should
    // be also added to the subgraph's output list
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
            if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
              graph_outputs_to_add[output] = output_order;
            }
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
        // Only when output is neither in input list nor erased list, add the output to output list
        else if (erased.find(output) == erased.end()) {
          if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
            graph_outputs_to_add[output] = output_order;
          }
          fused_outputs[output] = output_order++;
        }
      }
    }
  }

  fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
  fused_outputs.insert(graph_outputs_to_add.begin(), graph_outputs_to_add.end());

  // Sort inputs and outputs by the order they were added
  std::multimap<int, const NodeArg*> inputs, outputs;
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // Assign inputs and outputs to subgraph's meta_def
  auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  const std::string graph_type = graph.IsSubgraph() ? "subgraph" : "graph";
  meta_def->name = "TRTKernel_" + graph_type + "_" + graph.Name() + "_" + std::to_string(kernels_index++);
  meta_def->domain = kMSDomain;

  for (const auto& input : inputs) {
    if (input.second->Exists()) {
      meta_def->inputs.push_back(input.second->Name());
    }
  }

  for (const auto& output : outputs) {
    if (output.second->Exists()) {
      meta_def->outputs.push_back(output.second->Name());
    }
  }

  meta_def->since_version = 1;
  sub_graph->SetMetaDef(meta_def);

  return sub_graph;
}

SubGraphCollection_t TensorrtExecutionProvider::GetSupportedList(SubGraphCollection_t nodes_vector_input, int iterations, const int max_iterations,
                                                                 const onnxruntime::GraphViewer& graph, bool* early_termination) const {
  // Return if iterations are exceeding predefined number
  SubGraphCollection_t nodes_list_output;
  if (iterations > max_iterations) {
    *early_termination = true;
    return nodes_list_output;
  }

  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }

  iterations++;
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  for (const auto& group : nodes_vector_input) {
    // Construct subgraph
    if (!group.first.empty()) {
      if (group.second) {
        nodes_list_output.push_back(group);
      } else {
        onnxruntime::Model model_build(graph.Name(), true, ModelMetaData(), PathString(),
                                       IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap(),
                                       std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
        onnxruntime::Graph& graph_build = model_build.MainGraph();

        // Add node and node args
        // If node output is also parent graph output, the  output will be added to the
        // subgraph's output list
        std::vector<std::string> subgraph_output_names;
        for (const auto& index : group.first) {
          const auto& node = graph.GetNode(node_index[index]);
          std::vector<onnxruntime::NodeArg*> inputs, outputs;
          for (auto input : node->InputDefs()) {
            auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
            inputs.push_back(&n_input);
          }
          for (auto output : node->OutputDefs()) {
            auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
            outputs.push_back(&n_output);
            const auto name = output->Name();
            if (graph_output_names.find(name) != graph_output_names.end()) {
              subgraph_output_names.push_back(name);
            }
          }
          graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
        }

        // Add initializers to the subgraph
        const auto& init_tensors = graph.GetAllInitializedTensors();
        for (const auto& tensor : init_tensors) {
          graph_build.AddInitializedTensor(*(tensor.second));
        }

        ORT_ENFORCE(graph_build.Resolve().IsOK());

        // Add parent graph output to the subgraph
        int i = 0;
        std::vector<const NodeArg*> subgraph_outputs;
        subgraph_outputs.resize(subgraph_output_names.size());
        for (auto& name : subgraph_output_names) {
          auto output_arg = graph.GetNodeArg(name);
          auto& subgraph_output_arg = graph_build.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
          subgraph_outputs[i] = &subgraph_output_arg;
          ++i;
        }
        auto& graph_build_outputs = graph_build.GetOutputs();
        subgraph_outputs.insert(subgraph_outputs.begin(), graph_build_outputs.begin(), graph_build_outputs.end());
        graph_build.SetOutputs(graph_build_outputs);
        ORT_ENFORCE(graph_build.Resolve().IsOK());

        // Check if input tensors have shapes
        if (iterations > 1) {
          for (const auto* input_arg : graph_build.GetInputs()) {
            if (input_arg->Shape() == nullptr) {
              ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                                 "TensorRT input: " + input_arg->Name() + " has no shape specified. " +
                                                     "Please run shape inference on the onnx model first. Details can be found in " +
                                                     "https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md#shape-inference-for-tensorrt-subgraphs"));
            }
          }
        }

        // Serialize modelproto to string
        const onnxruntime::GraphViewer graph_viewer(graph_build);

        onnxruntime::Model model(graph_viewer.Name(), true, ModelMetaData(), PathString(),
                                 IOnnxRuntimeOpSchemaRegistryList(), graph_viewer.DomainToVersionMap(),
                                 std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
        ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
        ToGraphProtoInternal(graph_viewer, *(model_proto.mutable_graph()));
        model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

        std::string string_buf;
        model_proto.SerializeToString(&string_buf);

        if (dump_subgraphs_) {
          // Dump TensorRT subgraph for debugging if enabled via ORT_TENSORRT_DUMP_SUBGRAPHS env variable.
          std::fstream dump("TensorrtExecutionProvider_TRT_Subgraph.onnx", std::ios::out | std::ios::trunc | std::ios::binary);
          model_proto.SerializeToOstream(&dump);
        }
	
        // Get supported node list recursively
        SubGraphCollection_t parser_nodes_list;
        TensorrtLogger& trt_logger = GetTensorrtLogger();
        auto trt_builder = tensorrt_ptr::unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto trt_network = tensorrt_ptr::unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));

        auto trt_parser = tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
        trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_list);

        SubGraphCollection_t next_nodes_list;
        const std::vector<NodeIndex>& subgraph_node_index = graph_viewer.GetNodesInTopologicalOrder();
        next_nodes_list = GetSupportedList(parser_nodes_list, iterations, max_iterations, graph_viewer, early_termination);
        for (int i = 0, end = next_nodes_list.size(); i < end; ++i) {
          for (int j = 0, end = next_nodes_list[i].first.size(); j < end; ++j) {
            next_nodes_list[i].first[j] = group.first[subgraph_node_index[next_nodes_list[i].first[j]]];
          }
          nodes_list_output.push_back(next_nodes_list[i]);
        }
      }
    }
  }
  return nodes_list_output;
}

// Detect and remove cycles from supported node list
void TensorrtExecutionProvider::RemoveTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const onnxruntime::GraphViewer& graph) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  bool trt_cycle = true;
  while (trt_cycle) {
    trt_cycle = false;
    std::unordered_map<std::string, int> node_to_index_map;
    std::unordered_map<int, std::string> index_to_node_map;
    std::unordered_map<std::string, std::unordered_set<std::string>> input_to_nodes_map, node_to_outputs_map;
    std::unordered_set<int> non_trt_node_index(node_index.begin(), node_index.end());
    int counter = 0, id = 0;
    for (const auto& group : supported_nodes_vector) {
      if (!group.first.empty()) {
        // Construct subgraph from node list
        std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, counter, graph);

        // Create node to inputs/outputs/index maps
        const auto& meta_def = sub_graph->GetMetaDef();
        const std::string node_name = meta_def->name;
        if (node_to_index_map.find(node_name) == node_to_index_map.end()) {
          index_to_node_map[id] = node_name;
          node_to_index_map[node_name] = id++;
        }

        if (meta_def != nullptr) {
          for (const auto& input : meta_def->inputs) {
            input_to_nodes_map[input].insert(node_name);
          }
          for (const auto& output : meta_def->outputs) {
            node_to_outputs_map[node_name].insert(output);
          }
        }

        // Remove TensorRT nodes from node index list
        for (const auto& index : group.first) {
          non_trt_node_index.erase(node_index[index]);
        }
      }
    }

    // Add non TensorRT nodes to the maps
    for (const auto& index : non_trt_node_index) {
      const auto& node = graph.GetNode(index);
      std::string node_name = node->Name();
      if (node_to_index_map.find(node_name) == node_to_index_map.end()) {
        index_to_node_map[id] = node_name;
        node_to_index_map[node_name] = id++;
      }

      for (const auto& input : node->InputDefs()) {
        input_to_nodes_map[input->Name()].insert(node_name);
      }

      for (const auto& output : node->OutputDefs()) {
        node_to_outputs_map[node_name].insert(output->Name());
      }
    }

    // Create adjacency list
    int graph_size = node_to_index_map.size();
    std::list<int>* adjacency_map = new std::list<int>[graph_size];
    for (const auto& node : node_to_outputs_map) {
      for (auto iter = node.second.begin(); iter != node.second.end(); ++iter) {
        const auto& loc = input_to_nodes_map.find(*iter);
        if (loc != input_to_nodes_map.end()) {
          int parent_node_index = node_to_index_map.find(node.first)->second;
          for (auto child_node : loc->second) {
            int child_node_index = node_to_index_map.find(child_node)->second;
            adjacency_map[parent_node_index].push_back(child_node_index);
          }
        }
      }
    }

    // Check cycle in the graph
    bool* visited = new bool[graph_size];
    bool* st = new bool[graph_size];
    for (int i = 0; i < graph_size; ++i) {
      visited[i] = false;
      st[i] = false;
    }

    std::vector<int> cycles;
    bool has_cycle = false;
    for (int i = 0; i < graph_size; ++i) {
      if (FindCycleHelper(i, adjacency_map, visited, st, cycles)) {
        has_cycle = true;
        break;
      }
    }

    // Remove TensorRT subgraph from the supported node list if it's part of the cycle
    if (has_cycle) {
      for (int i = 0; i < static_cast<int>(cycles.size()); ++i) {
        auto loc = index_to_node_map.find(cycles[i]);
        if (loc != index_to_node_map.end() && loc->second.find("TRTKernel") != std::string::npos) {
          int trt_node_index = std::stoi(loc->second.substr(10));
          supported_nodes_vector.erase(supported_nodes_vector.begin() + trt_node_index);
          trt_cycle = true;
          break;
        }
      }
    }

    delete[] adjacency_map;
    delete[] visited;
    delete[] st;
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
TensorrtExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                         const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Remove nodes with empty shape
  SubGraphCollection_t parser_nodes_vector = RemoveEmptyShapeNodes(graph);

  // Get supported node list by TensorRT parser
  SubGraphCollection_t supported_nodes_vector;
  bool early_termination = false;
  supported_nodes_vector = GetSupportedList(parser_nodes_vector, 0, max_partition_iterations_, graph, &early_termination);
  if (early_termination) {
    supported_nodes_vector.clear();
  }

  // Remove subgraphs if its size is less than the predefined minimal size
  for (auto it = supported_nodes_vector.begin(); it != supported_nodes_vector.end(); ++it) {
    const int subgraph_size = it->first.size();
    if (subgraph_size < min_subgraph_size_) {
      supported_nodes_vector.erase(it--);
    }
  }

  // Detect and remove cycles from supported node list
  RemoveTensorRTGraphCycles(supported_nodes_vector, graph);

  // Construct subgraph capability from node list
  std::vector<std::unique_ptr<ComputeCapability>> result;
  int counter = 0;
  for (const auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, counter, graph);
      result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

common::Status TensorrtExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                                  std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* fused_node : fused_nodes) {
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;
    std::unordered_map<int, std::unordered_map<int, std::pair<int64_t, int64_t>>> input_shape_ranges;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int> output_types;

    // Build map from input name to its index in input definitions
    std::unordered_map<std::string, int> input_map;
    const auto& input_defs = fused_node->InputDefs();
    input_map.reserve(input_defs.size());
    for (int i = 0, end = input_defs.size(); i < end; ++i) {
      input_map[input_defs[i]->Name()] = i;
    }

    // Build map from output name to its index in output definitions
    std::unordered_map<std::string, int> output_map;
    const auto& output_defs = fused_node->OutputDefs();
    output_map.reserve(output_defs.size());
    for (int i = 0, end = output_defs.size(); i < end; ++i) {
      output_map[output_defs[i]->Name()] = i;
    }

    // Reconstruct graph proto from fused node's function body
    const auto* func_body = fused_node->GetFunctionBody();
    if (!func_body) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    }
    const Graph& graph_body = func_body->Body();
    onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(), PathString(),
                             IOnnxRuntimeOpSchemaRegistryList(), graph_body.DomainToVersionMap(),
                             std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
    *(model_proto.mutable_graph()) = graph_body.ToGraphProto();
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    std::string string_buf;
    model_proto.SerializeToString(&string_buf);

    if (dump_subgraphs_) {
      // Dump the TensorRT subgraph if enabled via ORT_TENSORRT_DUMP_SUBGRAPHS env variable.
      std::fstream dump(fused_node->Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
      model_proto.SerializeToOstream(&dump);
    }

    // Create TensorRT engine
    TensorrtLogger& trt_logger = GetTensorrtLogger();
    auto trt_builder = tensorrt_ptr::unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto trt_network = tensorrt_ptr::unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));
    auto trt_config = tensorrt_ptr::unique_pointer<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
    auto trt_parser = tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
    trt_parser->parse(string_buf.data(), string_buf.size());
    trt_config->setMaxWorkspaceSize(max_workspace_size_);

    // Set optimization profile for dynamic shapes
    auto trt_profile = trt_builder->createOptimizationProfile();
    for (unsigned int i = 0, end = trt_network->getNbInputs(); i < end; ++i) {
      auto input = trt_network->getInput(i);
      nvinfer1::Dims dims = input->getDimensions();
      nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);

      int nb_dims = dims.nbDims;
      if (input->isShapeTensor()) {  // Shape tensor
        std::vector<int32_t> shapes_min(nb_dims), shapes_opt(nb_dims), shapes_max(nb_dims);
        for (int j = 0, end = nb_dims; j < end; ++j) {
          shapes_min[j] = 1;
          shapes_opt[j] = 1;
          shapes_max[j] = 1000;
        }
        trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], nb_dims);
        trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], nb_dims);
        trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], nb_dims);
      } else {  // Execution tensor
        bool is_dynamic_shape = false;
        for (int j = 0, end = nb_dims; j < end; ++j) {
          // For dynamic shape subgraph, a dummy engine is created at compile phase.
          // Real engine will be created at compute phase based on input data
          if (dims.d[j] == -1) {  // Dynamic shape
            dims_min.d[j] = 1;
            dims_opt.d[j] = 1;
            dims_max.d[j] = 1;
            is_dynamic_shape = true;
          }
        }

        if (is_dynamic_shape) {
          trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_min);
          trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
          trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_max);
        }
      }
    }

    trt_config->addOptimizationProfile(trt_profile);
    if (fp16_enable_ && trt_builder->platformHasFastFp16()) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    auto trt_engine = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(trt_builder->buildEngineWithConfig(*trt_network, *trt_config));
    if (trt_engine == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not build Engine for fused node: " + fused_node->Name());
    }

    // Build TensorRT context
    auto trt_context = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (trt_context == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not build Execution Context for fused node: " + fused_node->Name());
    }

    // Get input shape and binding index
    int num_inputs = trt_network->getNbInputs();
    input_indexes.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      auto input = trt_network->getInput(i);
      const std::string& name = input->getName();
      size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());
      nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(bindingIndex));
      auto iter = input_map.find(name);
      if (iter != input_map.end()) {
        input_indexes[bindingIndex] = iter->second;
      }
      if (input->isShapeTensor()) {  // Shape tensor
        for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
          input_shape_ranges[bindingIndex][j] = std::make_pair(INT_MAX, INT_MIN);
        }
      } else {
        for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
          if (dimensions.d[j] == -1) {
            input_shape_ranges[bindingIndex][j] = std::make_pair(INT_MAX, INT_MIN);
          }
        }
      }
    }

    // Get output shape and binding index
    int num_outputs = trt_network->getNbOutputs();
    output_indexes.resize(num_outputs);
    output_shapes.resize(num_outputs);
    output_types.resize(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      const std::string& name = trt_network->getOutput(i)->getName();
      size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());
      nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(bindingIndex));
      bindingIndex -= num_inputs;
      auto iter = output_map.find(name);
      if (iter != output_map.end()) {
        output_indexes[bindingIndex] = iter->second;
      }
      for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
        output_shapes[bindingIndex].push_back(dimensions.d[j]);
      }

      const auto& graph_output = model_proto.graph().output();
      const auto& tensor_type = graph_output[i].type().tensor_type();
      output_types[bindingIndex] = tensor_type.elem_type();
    }

    ORT_ENFORCE(trt_engine->getNbBindings() == (num_inputs + num_outputs));

    // Save engine, context and input/output info to map
    parsers_.emplace(fused_node->Name(), std::move(trt_parser));
    engines_.emplace(fused_node->Name(), std::move(trt_engine));
    contexts_.emplace(fused_node->Name(), std::move(trt_context));
    builders_.emplace(fused_node->Name(), std::move(trt_builder));
    networks_.emplace(fused_node->Name(), std::move(trt_network));
    input_info_[fused_node->Name()].push_back(input_indexes);
    output_info_[fused_node->Name()].push_back(output_indexes);
    output_info_[fused_node->Name()].push_back(output_types);
    input_shape_ranges_[fused_node->Name()] = input_shape_ranges;
    output_shapes_[fused_node->Name()] = output_shapes;

    // Create function state
    // TODO: remove default capture
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<TensorrtFuncState> p = onnxruntime::make_unique<TensorrtFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, parsers_[context->node_name].get(),
            &engines_[context->node_name], &contexts_[context->node_name], builders_[context->node_name].get(),
            networks_[context->node_name].get(), input_info_[context->node_name], output_info_[context->node_name],
            input_shape_ranges_[context->node_name], output_shapes_[context->node_name], &tensorrt_mu_, &fp16_enable_,
            &max_workspace_size_};
      *state = p.release();
      return 0;
    };

    // Release function state
    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<TensorrtFuncState*>(state);
    };

    // Create compute function
    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      TensorrtFuncState* trt_state = reinterpret_cast<TensorrtFuncState*>(state);
      std::lock_guard<OrtMutex> lock(*(trt_state->tensorrt_mu_ptr));
      const std::vector<int>& input_indexes = (trt_state->input_info)[0];
      const std::vector<int>& output_indexes = (trt_state->output_info)[0];
      const std::vector<int>& output_types = (trt_state->output_info)[1];

      int num_binding_inputs = input_indexes.size();
      int num_binding_outputs = output_indexes.size();
      int total_bindings = num_binding_inputs + num_binding_outputs;
      std::vector<void*> buffers(total_bindings);

      // Update shape ranges
      bool dimension_update = false;
      auto trt_context = trt_state->context->get();
      auto trt_builder = trt_state->builder;
      nvinfer1::IOptimizationProfile* trt_profile = nullptr;
      for (int i = 0, end = num_binding_inputs; i < end; ++i) {
        // Check and update shape ranges for dynamic shape inputs
        auto& shape_ranges = trt_state->input_shape_ranges;
        if (shape_ranges.find(i) != shape_ranges.end()) {
          // TODO: check if getInput indexing is same with binding index
          auto input = trt_state->network->getInput(i);
          nvinfer1::Dims dims = input->getDimensions();
          nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);

          const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_indexes[i]);
          auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
          const auto& tensor_shape = ort.GetTensorShape(tensor_info);
          auto& engine = trt_context->getEngine();
          nvinfer1::Dims dimensions = engine.getBindingDimensions(static_cast<int>(i));
          int nb_dims = dimensions.nbDims;
          for (int j = 0, end = nb_dims; j < end; ++j) {
            auto& shape_range = shape_ranges[i];
            if (shape_range.find(j) != shape_range.end()) {
              dims_min.d[j] = shape_range[j].first;
              dims_opt.d[j] = shape_range[j].second;
              dims_max.d[j] = shape_range[j].second;

              // Update minimum dimension
              if (tensor_shape[j] < shape_range[j].first) {
                shape_range[j].first = tensor_shape[j];
                dims_min.d[j] = tensor_shape[j];
                dims_opt.d[j] = tensor_shape[j];
                dimension_update = true;
              }
              // Update maximum dimension
              if (tensor_shape[j] > shape_range[j].second) {
                shape_range[j].second = tensor_shape[j];
                dims_max.d[j] = tensor_shape[j];
                dims_opt.d[j] = tensor_shape[j];
                dimension_update = true;
              }
            }
          }

          if (dimension_update) {
            if (trt_profile == nullptr) {
              trt_profile = trt_builder->createOptimizationProfile();
            }
            if (engine.isShapeBinding(i)) {
              std::vector<int32_t> shapes_min(nb_dims), shapes_opt(nb_dims), shapes_max(nb_dims);
              for (int j = 0, end = nb_dims; j < end; ++j) {
                shapes_min[j] = dims_min.d[j];
                shapes_opt[j] = dims_opt.d[j];
                shapes_max[j] = dims_max.d[j];
              }
              trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], nb_dims);
              trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], nb_dims);
              trt_profile->setShapeValues(input->getName(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], nb_dims);
            } else {
              trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_min);
              trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
              trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_max);
            }
          }
        }
      }

      // Regenerate engine and context
      // Only one profile is generated, so no need to explicitly set optimization profile
      if (dimension_update) {
        auto trt_config = tensorrt_ptr::unique_pointer<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
        trt_config->setMaxWorkspaceSize(*(trt_state->max_workspace_size_ptr));
        trt_config->addOptimizationProfile(trt_profile);
        if (*(trt_state->fp16_enable_ptr) && trt_builder->platformHasFastFp16()) {
          trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        trt_state->context->reset();
        trt_state->engine->reset();
        *(trt_state->engine) =  tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(
                                  trt_builder->buildEngineWithConfig(*trt_state->network, *trt_config));

        if (trt_state->engine->get() == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP Failed to Build Engine.");
        }
        *(trt_state->context) = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(
                                  trt_state->engine->get()->createExecutionContext());
        if (trt_state->context->get() == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP Failed to Create Context.");
        }
        trt_context = trt_state->context->get();
      }

      // Set input shapes and assign input buffers
      for (int i = 0, end = num_binding_inputs; i < end; ++i) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_indexes[i]);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);

        // Set dynamic shapes
        nvinfer1::Dims dimensions = trt_context->getBindingDimensions(static_cast<int>(i));
        int nb_dims = dimensions.nbDims;
        if (dimension_update) {
          for (int j = 0, end = nb_dims; j < end; ++j)
            dimensions.d[j] = tensor_shape[j];
          trt_context->setBindingDimensions(i, dimensions);
        }

        auto tensor_type = ort.GetTensorElementType(tensor_info);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
        if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i] = const_cast<float*>(ort.GetTensorData<float>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
          buffers[i] = const_cast<MLFloat16*>(ort.GetTensorData<MLFloat16>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
          buffers[i] = const_cast<bool*>(ort.GetTensorData<bool>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i] = const_cast<int8_t*>(ort.GetTensorData<int8_t>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          buffers[i] = const_cast<int32_t*>(ort.GetTensorData<int32_t>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
          SafeInt<int> input_dim_size = 1;
          for (int j = 0, end = nb_dims; j < end; ++j) {
            input_dim_size *= tensor_shape[j];
          }
          CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[i], input_dim_size * sizeof(int32_t)));
          cuda::Impl_Cast<int64_t, int32_t>(ort.GetTensorData<int64_t>(input_tensor), reinterpret_cast<int32_t*>(buffers[i]), input_dim_size);
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.");
        }
      }

      // Set output shapes and assign output buffers
      std::vector<int> output_dim_sizes(num_binding_outputs, 1);
      std::vector<OrtValue*> output_tensor(num_binding_outputs, nullptr);
      for (int i = 0, end = num_binding_outputs; i < end; ++i) {
        // Set dynamic shapes
        nvinfer1::Dims dimensions = trt_context->getBindingDimensions(static_cast<int>(i + num_binding_inputs));
        int nb_dims = dimensions.nbDims;
        for (int j = 0, end = nb_dims; j < end; ++j) {
          trt_state->output_shapes[i][j] = dimensions.d[j];
        }

        int output_index = output_indexes[i];
        output_tensor[i] = ort.KernelContext_GetOutput(context, output_index, trt_state->output_shapes[i].data(), trt_state->output_shapes[i].size());

        if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<float>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<MLFloat16>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<bool>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int8_t>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int32_t>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Allocate INT32 CUDA memory for INT64 output type because TensorRT doesn't fully support INT64
          SafeInt<int> output_dim_size(output_dim_sizes[i]);
          for (int j = 0, end = nb_dims; j < end; ++j) {
            output_dim_size *= dimensions.d[j];
          }
          CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[i + num_binding_inputs], output_dim_size * sizeof(int32_t)));
          output_dim_sizes[i] = output_dim_size;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP output onnx tensor data type: " + std::to_string(output_types[i]) + " not supported.");
        }
      }

      // Run TRT inference
      if (!trt_context->enqueueV2(&buffers[0], nullptr, nullptr)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TensorRT EP Execution Context Enqueue Failed.");
      }

      // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
      for (int i = 0, end = num_binding_outputs; i < end; ++i) {
        if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          cuda::Impl_Cast<int32_t, int64_t>(reinterpret_cast<int32_t*>(buffers[i + num_binding_inputs]), ort.GetTensorMutableData<int64_t>(output_tensor[i]), output_dim_sizes[i]);
          cudaDeviceSynchronize();
          cudaFree(buffers[i + num_binding_inputs]);
        }
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime
