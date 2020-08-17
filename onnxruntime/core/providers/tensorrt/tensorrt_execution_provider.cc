// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <list>
#include <unordered_set>
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/safeint.h"

#include "tensorrt_execution_provider.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "cuda_runtime_api.h"
#include "gsl/gsl"
#include <experimental/filesystem>

#define CUDA_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(CUDA_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error executing ", #expr))

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
namespace fs = std::experimental::filesystem;
namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::Provider_KernelRegistry> kernel_registry{onnxruntime::Provider_KernelRegistry::Create()};
  Status st;
};

std::string GetEnginePath(const ::std::string& root, const std::string& name) {
  if (root.empty()) {
    return name + ".engine";
  } else {
    fs::path path = root;
    path.append(name + ".engine");
    return path.string();
  }
}
}  // namespace

namespace google {
namespace protobuf {
void ShutdownProtobufLibrary();
}
}  // namespace google

struct ShutdownProtobuf {
  ~ShutdownProtobuf() {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
} g_protobuf;

namespace onnxruntime {

namespace cuda {
template <>
void Impl_Cast(
    const int64_t* input_data, int32_t* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(input_data, output_data, count);
}

template <>
void Impl_Cast(
    const int32_t* input_data, int64_t* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(input_data, output_data, count);
}

}  // namespace cuda

template <>
bool CudaCall<cudaError, false>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg) {
  return g_host->CudaCall_false(retCode, exprString, libName, successCode, msg);
}

template <>
bool CudaCall<cudaError, true>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg) {
  return g_host->CudaCall_true(retCode, exprString, libName, successCode, msg);
}

class Memcpy final : public Provider_OpKernel {
 public:
  Memcpy(const Provider_OpKernelInfo&) {}

  Status Compute(Provider_OpKernelContext* ctx, const Provider_OpKernel_Base& base) const override {
    const auto* X = ctx->Input<Provider_Tensor>(0);
    Provider_Tensor* Y = ctx->Output(0, X->Shape());
    Status retval = base.GetInfo().GetDataTransferManager().CopyTensor(*X, *Y, base.GetInfo().GetKernelDef_ExecQueueId());
    return retval;
  }
};

template <typename T>
Provider_KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    (*Provider_KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .ExecQueueId(kCudaStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    (*Provider_KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .ExecQueueId(kCudaStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static Status RegisterTensorrtKernels(Provider_KernelRegistry& kernel_registry) {
  static const Provider_BuildKernelCreateInfoFn function_table[] = {
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

std::shared_ptr<Provider_KernelRegistry> TensorrtExecutionProvider::Provider_GetKernelRegistry() const {
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
    : Provider_IExecutionProvider{onnxruntime::kTensorrtExecutionProvider}, device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));

  Provider_DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault, [](int id) { return Provider_CreateCUDAAllocator(id, TRT); }, std::numeric_limits<size_t>::max()});
  allocator_ = CreateAllocator(default_memory_info, device_id_);
  Provider_InsertAllocator(allocator_);

  Provider_DeviceAllocatorRegistrationInfo pinned_allocator_info(
      {OrtMemTypeCPUOutput, [](int) { return Provider_CreateCUDAPinnedAllocator(0, TRT_PINNED); }, std::numeric_limits<size_t>::max()});
  Provider_InsertAllocator(CreateAllocator(pinned_allocator_info, device_id_));

  // Get environment variables
  const std::string max_partition_iterations_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMaxPartitionIterations);
  if (!max_partition_iterations_env.empty()) {
    max_partition_iterations_ = std::stoi(max_partition_iterations_env);
  }

  const std::string min_subgraph_size_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMinSubgraphSize);
  if (!min_subgraph_size_env.empty()) {
    min_subgraph_size_ = std::stoi(min_subgraph_size_env);
  }

  const std::string max_workspace_size_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMaxWorkspaceSize);
  if (!max_workspace_size_env.empty()) {
    max_workspace_size_ = std::stoull(max_workspace_size_env);
  }

  const std::string fp16_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kFP16Enable);
  if (!fp16_enable_env.empty()) {
    fp16_enable_ = (std::stoi(fp16_enable_env) == 0 ? false : true);
  }

  const std::string dump_subgraphs_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDumpSubgraphs);
  if (!dump_subgraphs_env.empty()) {
    dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
  }

  const std::string engine_cache_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEngineCacheEnable);
  if (!engine_cache_enable_env.empty()) {
    engine_cache_enable_ = (std::stoi(engine_cache_enable_env) == 0 ? false : true);
  }

  if (engine_cache_enable_) {
    engine_cache_path_ = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEngineCachePath);
    if (!engine_cache_path_.empty() && !fs::is_directory(engine_cache_path_)) {
      if (!fs::create_directory(engine_cache_path_)) {
        throw std::runtime_error("Failed to create directory " + engine_cache_path_);
      }
    }
    runtime_ = nvinfer1::createInferRuntime(GetTensorrtLogger());
  }
}

TensorrtExecutionProvider::~TensorrtExecutionProvider() {}

Provider_AllocatorPtr TensorrtExecutionProvider::Provider_GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return Provider_IExecutionProvider::Provider_GetAllocator(id, mem_type);
  }
}

std::unique_ptr<onnxruntime::Provider_IDataTransfer> TensorrtExecutionProvider::Provider_GetDataTransfer() const {
  return onnxruntime::Provider_CreateGPUDataTransfer();
}

// Convert GraphViewer graph to GraphProto
void ToGraphProtoInternal(const onnxruntime::Provider_GraphViewer& graph, Provider_GraphProto& graph_proto) {
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
    const gsl::not_null<Provider_NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Provider_Node*> p_node{graph.GetNode(node_idx)};
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

std::unique_ptr<Provider_IndexedSubGraph> TensorrtExecutionProvider::GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index, const onnxruntime::Provider_GraphViewer& graph) const {
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
  std::unique_ptr<Provider_IndexedSubGraph> sub_graph = onnxruntime::Provider_IndexedSubGraph::Create();
  std::unordered_map<const Provider_NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add, graph_outputs_to_add;
  std::unordered_set<const Provider_NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  for (const auto& index : graph_nodes_index.first) {
    sub_graph->Nodes().push_back(node_index[index]);
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

    for (const auto& input : node->ImplicitInputDefs()) {
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
  std::multimap<int, const Provider_NodeArg*> inputs, outputs;
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    inputs.insert(std::pair<int, const Provider_NodeArg*>(it->second, it->first));
  }

  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const Provider_NodeArg*>(it->second, it->first));
  }

  // Assign inputs and outputs to subgraph's meta_def
  auto meta_def = Provider_IndexedSubGraph_MetaDef::Create();
  const std::string graph_type = graph.IsSubgraph() ? "subgraph" : "graph";
  meta_def->name() = "TRTKernel_" + graph_type + "_" + graph.Name() + "_" + std::to_string(kernels_index++);
  meta_def->domain() = kMSDomain;

  for (const auto& input : inputs) {
    if (input.second->Exists()) {
      meta_def->inputs().push_back(input.second->Name());
    }
  }

  for (const auto& output : outputs) {
    if (output.second->Exists()) {
      meta_def->outputs().push_back(output.second->Name());
    }
  }

  meta_def->since_version() = 1;
  sub_graph->SetMetaDef(std::move(meta_def));

  return sub_graph;
}

SubGraphCollection_t TensorrtExecutionProvider::GetSupportedList(SubGraphCollection_t nodes_vector_input, int iterations, const int max_iterations,
                                                                 const onnxruntime::Provider_GraphViewer& graph, bool* early_termination) const {
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
        auto model_build = graph.CreateModel(*GetLogger());
        auto& graph_build = model_build->MainGraph();

        // Add node and node args
        // If node output is also parent graph output, the  output will be added to the
        // subgraph's output list
        std::vector<std::string> subgraph_output_names;
        for (const auto& index : group.first) {
          const auto& node = graph.GetNode(node_index[index]);
          std::vector<onnxruntime::Provider_NodeArg*> inputs, outputs;
          for (auto input : node->InputDefs()) {
            auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
            inputs.push_back(&n_input);
            const Provider_TensorProto* initializer = nullptr;
            if (graph.GetInitializedTensor(input->Name(), initializer)) {
              const Provider_TensorProto* subgraph_initializer = nullptr;
              if (!graph_build.GetInitializedTensor(input->Name(), subgraph_initializer)) {
                graph_build.AddInitializedTensor(*(initializer));
              }
            }
          }

          for (auto input : node->ImplicitInputDefs()) {
            const Provider_TensorProto* initializer = nullptr;
            if (graph.GetInitializedTensor(input->Name(), initializer)) {
              const Provider_TensorProto* subgraph_initializer = nullptr;
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
          graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
        }

        ORT_ENFORCE(graph_build.Resolve().IsOK());

        // Add parent graph output to the subgraph
        int i = 0;
        std::vector<const Provider_NodeArg*> subgraph_outputs;
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
        auto graph_viewer = graph_build.CreateGraphViewer();
        auto model = graph_viewer->CreateModel(*GetLogger());
        auto model_proto = model->ToProto();
        ToGraphProtoInternal(*graph_viewer, *model_proto->mutable_graph());
        model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

        std::string string_buf;
        model_proto->SerializeToString(string_buf);

        if (dump_subgraphs_) {
          // Dump TensorRT subgraph for debugging if enabled via ORT_TENSORRT_DUMP_SUBGRAPHS env variable.
          std::fstream dump("TensorrtExecutionProvider_TRT_Subgraph.onnx", std::ios::out | std::ios::trunc | std::ios::binary);
          model_proto->SerializeToOstream(dump);
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
        const std::vector<NodeIndex>& subgraph_node_index = graph_viewer->GetNodesInTopologicalOrder();
        next_nodes_list = GetSupportedList(parser_nodes_list, iterations, max_iterations, *graph_viewer, early_termination);
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
void TensorrtExecutionProvider::RemoveTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const onnxruntime::Provider_GraphViewer& graph) const {
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
        std::unique_ptr<Provider_IndexedSubGraph> sub_graph = GetSubGraph(group, counter, graph);

        // Create node to inputs/outputs/index maps
        const auto& meta_def = sub_graph->GetMetaDef();
        const std::string node_name = meta_def->name();
        if (node_to_index_map.find(node_name) == node_to_index_map.end()) {
          index_to_node_map[id] = node_name;
          node_to_index_map[node_name] = id++;
        }

        if (meta_def != nullptr) {
          for (const auto& input : meta_def->inputs()) {
            input_to_nodes_map[input].insert(node_name);
          }
          for (const auto& output : meta_def->outputs()) {
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
          std::size_t found = loc->second.rfind("_");
          if (found != std::string::npos) {
            int trt_node_index = std::stoi(loc->second.substr(found + 1));
            supported_nodes_vector.erase(supported_nodes_vector.begin() + trt_node_index);
            trt_cycle = true;
            break;
          }
        }
      }
    }

    delete[] adjacency_map;
    delete[] visited;
    delete[] st;
  }
}

std::vector<std::unique_ptr<Provider_ComputeCapability>>
TensorrtExecutionProvider::Provider_GetCapability(const onnxruntime::Provider_GraphViewer& graph,
                                                  const std::vector<const Provider_KernelRegistry*>& /*kernel_registries*/) const {
  // Get supported node list from TensorRT parser
  std::vector<size_t> nodes_vector(graph.NumberOfNodes());
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  SubGraphCollection_t supported_nodes_vector, parser_nodes_vector = {{nodes_vector, false}};
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
  std::vector<std::unique_ptr<Provider_ComputeCapability>> result;
  int counter = 0, number_of_trt_nodes = 0;
  for (const auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      std::unique_ptr<Provider_IndexedSubGraph> sub_graph = GetSubGraph(group, counter, graph);
      result.push_back(Provider_ComputeCapability::Create(std::move(sub_graph)));
      number_of_trt_nodes += group.first.size();
    }
  }

  const int number_of_subgraphs = supported_nodes_vector.size();
  if (number_of_subgraphs == 0) {
    LOGS_DEFAULT(WARNING) << "No graph is running on TensorRT exeuction provider.";
  } else {
    LOGS_DEFAULT(INFO) << "Number of subgraphs running on TensorRT exeuction provider: " << number_of_subgraphs;
  }

  return result;
}

common::Status TensorrtExecutionProvider::Provider_Compile(const std::vector<onnxruntime::Provider_Node*>& fused_nodes,
                                                           std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* fused_node : fused_nodes) {
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
    const Provider_Graph& graph_body = func_body->Body();
    auto model = graph_body.CreateGraphViewer()->CreateModel(*GetLogger());
    auto model_proto = model->ToProto();

    *model_proto->mutable_graph() = *graph_body.ToGraphProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    std::string string_buf;
    model_proto->SerializeToString(string_buf);

    if (dump_subgraphs_) {
      // Dump the TensorRT subgraph if enabled via ORT_TENSORRT_DUMP_SUBGRAPHS env variable.
      std::fstream dump(fused_node->Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
      model_proto->SerializeToOstream(dump);
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

    int num_inputs = trt_network->getNbInputs();
    int num_outputs = trt_network->getNbOutputs();
    std::unordered_map<std::string, int> input_indexes(num_inputs);
    std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>> input_shape_ranges;
    std::unordered_map<std::string, int> output_indexes(num_outputs);
    std::unordered_map<std::string, int> output_types(num_outputs);

    // Initialize shape range for dynamic shape tensors
    bool has_dynamic_shape = false;
    for (unsigned int i = 0, end = num_inputs; i < end; ++i) {
      auto input = trt_network->getInput(i);
      const std::string& input_name = input->getName();
      nvinfer1::Dims dims = input->getDimensions();
      int nb_dims = dims.nbDims;

      if (input->isShapeTensor()) {
        // Shape tensor
        input_shape_ranges[input_name][0] = std::make_pair(INT_MAX, INT_MIN);
        has_dynamic_shape = true;
      } else {
        // Execution tensor
        for (int j = 0, end = nb_dims; j < end; ++j) {
          if (dims.d[j] == -1) {
            input_shape_ranges[input_name][j] = std::make_pair(INT_MAX, INT_MIN);
            has_dynamic_shape = true;
          }
        }
      }
    }

    std::string trt_node_name_with_precision = fused_node->Name();
    if (fp16_enable_ && trt_builder->platformHasFastFp16()) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
      trt_node_name_with_precision += "_fp16";
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] FP16 mode is enabled.";
    }

    // Build TRT engine here if the graph doesn't have dynamic shape input. Otherwise engine will
    // be built at runtime
    tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine> trt_engine;
    tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext> trt_context;
    if (!has_dynamic_shape) {
      std::ifstream planFile(GetEnginePath(engine_cache_path_, trt_node_name_with_precision), std::ios::binary | std::ios::in);
      if (planFile && engine_cache_enable_) {
        planFile.seekg(0, std::ios::end);
        int engine_size = planFile.tellg();
        planFile.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        planFile.read((char*)engine_buf.get(), engine_size);
        planFile.close();
        trt_engine = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size, nullptr));
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Deserialized " + GetEnginePath(engine_cache_path_, trt_node_name_with_precision);
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not deserialize engine from " + trt_node_name_with_precision + ".engine");
        }
      } else {
        trt_engine = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(trt_builder->buildEngineWithConfig(*trt_network, *trt_config));
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not build engine for fused node: " + fused_node->Name());
        }

        if (engine_cache_enable_) {
          nvinfer1::IHostMemory* serializedModel = trt_engine->serialize();
          std::ofstream file(GetEnginePath(engine_cache_path_, trt_node_name_with_precision), std::ios::binary | std::ios::out);
          file.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
          serializedModel->destroy();
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + GetEnginePath(engine_cache_path_, trt_node_name_with_precision);
        }
      }
      trt_context = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
      if (trt_context == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "TensorRT EP could not build execution context for fused node: " + fused_node->Name());
      }
    }

    // Create input to index map
    for (int i = 0; i < num_inputs; ++i) {
      auto input = trt_network->getInput(i);
      const std::string& input_name = input->getName();
      const auto& iter = input_map.find(input_name);
      if (iter != input_map.end()) {
        input_indexes[input_name] = iter->second;
      }
    }

    //  Create output to index and type maps
    const auto& graph_output = model_proto->graph().output();
    for (int i = 0; i < num_outputs; ++i) {
      const std::string& output_name = trt_network->getOutput(i)->getName();
      const auto& iter = output_map.find(output_name);
      if (iter != output_map.end()) {
        output_indexes[output_name] = iter->second;
      }
      const auto& tensor_type = graph_output[i].type().tensor_type();
      output_types[output_name] = tensor_type.elem_type();
    }

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

    // Create function state
    // TODO: remove default capture
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<TensorrtFuncState> p = onnxruntime::make_unique<TensorrtFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, parsers_[context->node_name].get(),
            &engines_[context->node_name], &contexts_[context->node_name], builders_[context->node_name].get(),
            networks_[context->node_name].get(), input_info_[context->node_name], output_info_[context->node_name],
            input_shape_ranges_[context->node_name], &tensorrt_mu_, &fp16_enable_,
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
      const std::unordered_map<std::string, int>& input_indexes = (trt_state->input_info)[0];
      const std::unordered_map<std::string, int>& output_indexes = (trt_state->output_info)[0];
      const std::unordered_map<std::string, int>& output_types = (trt_state->output_info)[1];
      auto& shape_ranges = trt_state->input_shape_ranges;
      auto trt_builder = trt_state->builder;
      auto trt_engine = trt_state->engine->get();
      auto trt_context = trt_state->context->get();
      int num_inputs = input_indexes.size();
      int num_outputs = output_indexes.size();

      // Update shape ranges
      bool engine_update = false;
      std::unordered_map<std::string, bool> dimension_update;
      std::unordered_map<std::string, std::vector<int32_t>> tensor_shape_values;
      nvinfer1::IOptimizationProfile* trt_profile = nullptr;
      for (int i = 0, end = num_inputs; i < end; ++i) {
        auto input = trt_state->network->getInput(i);
        const std::string& input_name = input->getName();
        nvinfer1::Dims dims = input->getDimensions();
        int nb_dims = dims.nbDims;

        // Check and update shape ranges for dynamic shape inputs
        dimension_update[input_name] = false;
        if (shape_ranges.find(input_name) != shape_ranges.end()) {
          int input_index = 0;
          const auto& iter = input_indexes.find(input_name);
          if (iter != input_indexes.end()) {
            input_index = iter->second;
          }

          const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
          auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
          const auto& tensor_shapes = ort.GetTensorShape(tensor_info);
          auto& shape_range = shape_ranges[input_name];

          // Create shape profile
          if (input->isShapeTensor()) {
            // Get shape values for shape tensor input
            const auto& tensor_type = ort.GetTensorElementType(tensor_info);
            int shape_size = nb_dims == 0 ? 1 : tensor_shapes[0];
            tensor_shape_values[input_name].resize(shape_size);
            switch (tensor_type) {
              case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                int32_t* input = new int32_t[shape_size];
                CUDA_RETURN_IF_ERROR(cudaMemcpy(input, ort.GetTensorData<int32_t>(input_tensor), shape_size * sizeof(int32_t), cudaMemcpyDeviceToHost));
                for (int j = 0; j < shape_size; ++j) {
                  tensor_shape_values[input_name][j] = input[j];
                }
                delete[] input;
                break;
              }
              case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                int64_t* input = new int64_t[shape_size];
                CUDA_RETURN_IF_ERROR(cudaMemcpy(input, ort.GetTensorData<int64_t>(input_tensor), shape_size * sizeof(int64_t), cudaMemcpyDeviceToHost));
                for (int j = 0; j < shape_size; ++j) {
                  tensor_shape_values[input_name][j] = static_cast<int32_t>(input[j]);
                }
                delete[] input;
                break;
              }
              default: {
                return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                       "TensorRT shape tensor data type: " + std::to_string(tensor_type) + " not supported.");
              }
            }

            // Update shape ranges
            std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
            int shape_range_size = shape_range.size();
            if (shape_size == shape_range_size) {
              // If shape size matches, check/update shape range
              for (int j = 0; j < shape_size; ++j) {
                shapes_min[j] = shape_range[j].first;
                shapes_opt[j] = shape_range[j].second;
                shapes_max[j] = shape_range[j].second;

                const auto& tensor_shape_value = tensor_shape_values[input_name][j];
                // Update shape range lower bound
                if (tensor_shape_value < shape_range[j].first) {
                  shape_range[j].first = tensor_shape_value;
                  shapes_min[j] = tensor_shape_value;
                  shapes_opt[j] = tensor_shape_value;
                  dimension_update[input_name] = true;
                }
                // Update shape range upper bound
                if (tensor_shape_value > shape_range[j].second) {
                  shape_range[j].second = tensor_shape_value;
                  shapes_max[j] = tensor_shape_value;
                  shapes_opt[j] = tensor_shape_value;
                  dimension_update[input_name] = true;
                }
              }
            } else {
              // If shape size doesn't match, initialize shape_range with the new shape value
              shape_range.clear();
              for (int j = 0; j < shape_size; ++j) {
                const auto& tensor_shape_value = tensor_shape_values[input_name][j];
                shape_range[j] = std::make_pair(tensor_shape_value, tensor_shape_value);
                shapes_min[j] = tensor_shape_value;
                shapes_opt[j] = tensor_shape_value;
                shapes_max[j] = tensor_shape_value;
              }
              dimension_update[input_name] = true;
            }

            if (dimension_update[input_name]) {
              if (trt_profile == nullptr) {
                trt_profile = trt_builder->createOptimizationProfile();
              }
              trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
              trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
              trt_profile->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);
            }

          } else  //execution tensor
          {
            nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);
            for (int j = 0, end = nb_dims; j < end; ++j) {
              const auto& tensor_shape = tensor_shapes[j];
              if (shape_range.find(j) != shape_range.end()) {
                dims_min.d[j] = shape_range[j].first;
                dims_opt.d[j] = shape_range[j].second;
                dims_max.d[j] = shape_range[j].second;

                // Update minimum dimension
                if (tensor_shape < shape_range[j].first) {
                  shape_range[j].first = tensor_shape;
                  dims_min.d[j] = tensor_shape;
                  dims_opt.d[j] = tensor_shape;
                  dimension_update[input_name] = true;
                }
                // Update maximum dimension
                if (tensor_shape > shape_range[j].second) {
                  shape_range[j].second = tensor_shape;
                  dims_max.d[j] = tensor_shape;
                  dims_opt.d[j] = tensor_shape;
                  dimension_update[input_name] = true;
                }
              }
            }

            if (dimension_update[input_name]) {
              if (trt_profile == nullptr) {
                trt_profile = trt_builder->createOptimizationProfile();
              }
              trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
              trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
              trt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
            }
          }
          ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
        }

        if (!engine_update && dimension_update[input_name]) {
          engine_update = true;
        }
      }

      // Regenerate engine
      // Only one profile is generated, so no need to explicitly set optimization profile
      if (engine_update) {
        auto trt_config = tensorrt_ptr::unique_pointer<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
        trt_config->setMaxWorkspaceSize(*(trt_state->max_workspace_size_ptr));
        trt_config->addOptimizationProfile(trt_profile);
        if (*(trt_state->fp16_enable_ptr) && trt_builder->platformHasFastFp16()) {
          trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        trt_state->context->reset();
        trt_state->engine->reset();
        *(trt_state->engine) = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(
            trt_builder->buildEngineWithConfig(*trt_state->network, *trt_config));

        if (trt_state->engine->get() == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to build engine.");
        }
        trt_engine = trt_state->engine->get();

        *(trt_state->context) = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(
            trt_state->engine->get()->createExecutionContext());
        if (trt_state->context->get() == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to create context.");
        }
        trt_context = trt_state->context->get();
      }

      // Get input and output binding names
      int total_bindings = trt_engine->getNbBindings();
      std::vector<void*> buffers(total_bindings);
      std::vector<std::string> input_binding_names, output_binding_names;
      for (int i = 0, end = total_bindings; i < end; ++i) {
        if (trt_engine->bindingIsInput(i)) {
          input_binding_names.push_back(trt_engine->getBindingName(i));
        } else {
          output_binding_names.push_back(trt_engine->getBindingName(i));
        }
      }

      // Set input shapes and assign input buffers
      std::vector<int> binding_buffers_to_freeup;
      for (int i = 0, end = input_binding_names.size(); i < end; ++i) {
        const std::string& input_name = input_binding_names[i];
        int binding_index = trt_engine->getBindingIndex(input_name.c_str());
        if (binding_index == -1) {
          continue;
        }

        int input_index = 0;
        const auto& iter = input_indexes.find(input_name);
        if (iter != input_indexes.end()) {
          input_index = iter->second;
        }
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shapes = ort.GetTensorShape(tensor_info);

        // Set dynamic shapes
        nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(binding_index));
        int nb_dims = dimensions.nbDims;
        if (dimension_update.find(input_name) != dimension_update.end()) {
          if (trt_engine->isShapeBinding(binding_index)) {
            trt_context->setInputShapeBinding(binding_index, &tensor_shape_values[input_name][0]);
          } else {
            for (int j = 0, end = nb_dims; j < end; ++j) {
              dimensions.d[j] = tensor_shapes[j];
            }
            trt_context->setBindingDimensions(binding_index, dimensions);
          }
        }

        const auto& input_type = ort.GetTensorElementType(tensor_info);
        switch (input_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            auto input_tensor_ptr = ort.GetTensorData<float>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(float)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = const_cast<float*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            auto input_tensor_ptr = ort.GetTensorData<uint16_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(uint16_t)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = const_cast<uint16_t*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            auto input_tensor_ptr = ort.GetTensorData<bool>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(bool)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = const_cast<bool*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            auto input_tensor_ptr = ort.GetTensorData<int8_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(int8_t)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = const_cast<int8_t*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            auto input_tensor_ptr = ort.GetTensorData<int32_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(int32_t)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = const_cast<int32_t*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
            auto input_tensor_ptr = ort.GetTensorData<int64_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(int32_t)));
            } else {
              SafeInt<int> input_dim_size = 1;
              for (int j = 0, end = nb_dims; j < end; ++j) {
                if (tensor_shapes[j] == 0) {
                  input_dim_size = 1;
                  break;
                } else {
                  input_dim_size *= tensor_shapes[j];
                }
              }
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], input_dim_size * sizeof(int32_t)));
              cuda::Impl_Cast<int64_t, int32_t>(input_tensor_ptr, reinterpret_cast<int32_t*>(buffers[binding_index]), input_dim_size);
            }
            binding_buffers_to_freeup.push_back(binding_index);
            break;
          }
          default: {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP input onnx tensor data type: " + std::to_string(input_type) + " not supported.");
          }
        }
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }

      // Set output shapes and assign output buffers
      std::vector<int> output_dim_sizes(num_outputs, 1);
      std::vector<OrtValue*> output_tensor(num_outputs, nullptr);
      for (int i = 0, end = output_binding_names.size(); i < end; ++i) {
        // Set dynamic shapes
        const std::string& output_name = output_binding_names[i];
        int binding_index = trt_engine->getBindingIndex(output_name.c_str());
        if (binding_index == -1) {
          continue;
        }

        int output_index = 0;
        const auto& index_iter = output_indexes.find(output_name);
        if (index_iter != output_indexes.end()) {
          output_index = index_iter->second;
        }
        nvinfer1::Dims dimensions = trt_context->getBindingDimensions(static_cast<int>(binding_index));
        int nb_dims = dimensions.nbDims;
        std::vector<int64_t> output_shapes(nb_dims);
        for (int j = 0, end = nb_dims; j < end; ++j) {
          output_shapes[j] = dimensions.d[j];
        }
        output_tensor[i] = ort.KernelContext_GetOutput(context, output_index, output_shapes.data(), output_shapes.size());

        int output_type = 0;
        const auto& type_iter = output_types.find(output_name);
        if (type_iter != output_types.end()) {
          output_type = type_iter->second;
        }

        switch (output_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            auto output_tensor_ptr = ort.GetTensorMutableData<float>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(float)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            auto output_tensor_ptr = ort.GetTensorMutableData<uint16_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(uint16_t)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            auto output_tensor_ptr = ort.GetTensorMutableData<bool>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(bool)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            auto output_tensor_ptr = ort.GetTensorMutableData<int8_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(int8_t)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            auto output_tensor_ptr = ort.GetTensorMutableData<int32_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(int32_t)));
              binding_buffers_to_freeup.push_back(binding_index);
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            // Allocate INT32 CUDA memory for INT64 output type because TensorRT doesn't fully support INT64
            auto output_tensor_ptr = ort.GetTensorMutableData<int64_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], sizeof(int32_t)));
              output_dim_sizes[i] = 1;
            } else {
              SafeInt<int> output_dim_size(output_dim_sizes[i]);
              for (int j = 0, end = nb_dims; j < end; ++j) {
                if (dimensions.d[j] == 0) {
                  output_dim_size = 1;
                  break;
                } else {
                  output_dim_size *= dimensions.d[j];
                }
              }
              CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[binding_index], output_dim_size * sizeof(int32_t)));
              output_dim_sizes[i] = output_dim_size;
            }
            binding_buffers_to_freeup.push_back(binding_index);
            break;
          }
          default: {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP output tensor data type: " + std::to_string(output_type) + " not supported.");
          }
        }
      }

      // Run TRT inference
      if (!trt_context->enqueueV2(&buffers[0], nullptr, nullptr)) {
        for (const auto& binding_index : binding_buffers_to_freeup) {
          cudaFree(buffers[binding_index]);
        }
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TensorRT EP execution context enqueue failed.");
      }

      // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
      for (int i = 0, end = output_binding_names.size(); i < end; ++i) {
        const std::string& output_name = output_binding_names[i];
        size_t binding_index = trt_engine->getBindingIndex(output_name.c_str());
        int output_type = 0;
        const auto& iter = output_types.find(output_name);
        if (iter != output_types.end()) {
          output_type = iter->second;
        }
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          auto output_tensor_ptr = ort.GetTensorMutableData<int64_t>(output_tensor[i]);
          if (output_tensor_ptr != nullptr) {
            cuda::Impl_Cast<int32_t, int64_t>(reinterpret_cast<int32_t*>(buffers[binding_index]), output_tensor_ptr, output_dim_sizes[i]);
          }
        }
      }

      cudaDeviceSynchronize();
      for (const auto& binding_index : binding_buffers_to_freeup) {
        cudaFree(buffers[binding_index]);
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime