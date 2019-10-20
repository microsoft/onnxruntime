// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "core/common/status.h"
#include "onnx/shape_inference/implementation.h"
#include "cuda_runtime_api.h"
#include "gsl/gsl"
#include "core/graph/model.h"
#include "core/providers/cuda/gpu_data_transfer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

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

static void RegisterTensorrtKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetTensorrtKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterTensorrtKernels(*kernel_registry);

  return kernel_registry;
}

std::shared_ptr<KernelRegistry> TensorrtExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::GetTensorrtKernelRegistry();
  return kernel_registry;
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
      {OrtMemTypeDefault, [](int id) { return onnxruntime::make_unique<CUDAAllocator>(id, TRT); }, std::numeric_limits<size_t>::max()});
  allocator_ = CreateAllocator(default_memory_info, device_id_);
  InsertAllocator(allocator_);

  DeviceAllocatorRegistrationInfo pinned_memory_info(
      {OrtMemTypeCPUOutput, [](int) { return onnxruntime::make_unique<CUDAPinnedAllocator>(0, TRT_PINNED); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(pinned_memory_info, device_id_));
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

std::unique_ptr<IndexedSubGraph> TensorrtExecutionProvider::GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index, const onnxruntime::GraphViewer& graph) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.first.size());
  for (const auto& index : graph_nodes_index.first) {
    node_set.insert(node_index[index]);
  }
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();

  // Find inputs and outputs of the subgraph
  std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add;
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
        //only when input is neither in output list nor erased list, add the input to input list
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
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // Assign inputs and outputs to subgraph's meta_def
  auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "TRTKernel_" + std::to_string(kernels_index++);
  meta_def->domain = kMSDomain;

  for (const auto& input : inputs) {
    meta_def->inputs.push_back(input.second->Name());
  }

  for (const auto& output : outputs) {
    meta_def->outputs.push_back(output.second->Name());
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

  iterations++;
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  int counter = 0;
  for (const auto& group : nodes_vector_input) {
    //construct subgraph
    if (!group.first.empty()) {
      std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, counter, graph);

      if (group.second) {
        nodes_list_output.push_back(group);
      } else {
        onnxruntime::Model model_build(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
        onnxruntime::Graph& graph_build = model_build.MainGraph();

        //Add node and node args
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
          }
          graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
        }

        ORT_ENFORCE(graph_build.Resolve().IsOK());

        for (const auto& input : sub_graph->GetMetaDef()->inputs) {
          const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
          if (graph.GetInitializedTensor(input, initializer)) {
            graph_build.AddInitializedTensor(*initializer);
          }
        }

        // Serialize modelproto to string
        ONNX_NAMESPACE::ModelProto model_proto = model_build.ToProto();
        std::string string_buf;
        model_proto.SerializeToString(&string_buf);

        // Get supported node list recursively
        SubGraphCollection_t parser_nodes_list;
        TensorrtLogger& trt_logger = GetTensorrtLogger();
        auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));

        auto trt_parser = unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
        trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_list);

        SubGraphCollection_t next_nodes_list;
        const onnxruntime::GraphViewer graph_viewer(graph_build);
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

std::vector<std::unique_ptr<ComputeCapability>>
TensorrtExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                         const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Construct modelproto from graph
  onnxruntime::Model model(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
  onnxruntime::Graph& graph_build = model.MainGraph();
  for (const auto& node : graph.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }
    for (auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }

  auto status = graph_build.Resolve();

  //Add initializer to graph
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(status.IsOK(), status);
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize modelproto to string
  std::string string_buf;
  model_proto.SerializeToString(&string_buf);

  // Get supported node list
  SubGraphCollection_t parser_nodes_vector;
  TensorrtLogger& trt_logger = GetTensorrtLogger();
  auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));
  auto trt_parser = unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
  trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_vector);

  SubGraphCollection_t supported_nodes_vector;
  const char* batch_env = getenv("ORT_TENSORRT_MAX_PARSER_ITERATIONS");
  const int max_iterations = batch_env ? atoi(batch_env) : max_parser_iterations_;
  bool early_termination = false;
  supported_nodes_vector = GetSupportedList(parser_nodes_vector, 0, max_iterations, graph, &early_termination);
  if (early_termination) {
    supported_nodes_vector.clear();
  }

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
    std::vector<int> input_dim_sizes;
    std::vector<int> output_indexes;
    std::vector<int> output_dim_sizes;
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
    onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph_body.DomainToVersionMap());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
    *(model_proto.mutable_graph()) = graph_body.ToGraphProto();
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    std::string string_buf;
    model_proto.SerializeToString(&string_buf);

    // Create TensorRT engine
    TensorrtLogger& trt_logger = GetTensorrtLogger();
    auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));

    auto trt_parser = unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
    trt_parser->parse(string_buf.data(), string_buf.size());

    const char* batch_env = getenv("ORT_TENSORRT_MAX_BATCH_SIZE");
    if (batch_env) {
      const int max_batch_size = atoi(batch_env);
      SetMaxBatchSize(max_batch_size);
    }

    const char* workspace_env = getenv("ORT_TENSORRT_MAX_WORKSPACE_SIZE");
    if (workspace_env) {
      const size_t max_workspace_size = atoi(workspace_env);
      SetMaxWorkspaceSize(max_workspace_size);
    }

    trt_builder->setMaxBatchSize(max_batch_size_);
    auto trt_config = unique_pointer<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
    trt_config->setMaxWorkspaceSize(max_workspace_size_);

    //Set optimization profile for dynamic shapes
    //Only support dynamic batch size on the first dimension for now
    //TODO: add full dynamic shape support
    auto trt_profile = trt_builder->createOptimizationProfile();
    bool dynamic_shape = false;
    for (unsigned int i = 0, n = trt_network->getNbInputs(); i < n; i++) {
      auto input = trt_network->getInput(i);
      nvinfer1::Dims dims = input->getDimensions();
      nvinfer1::Dims dims_min = dims;
      nvinfer1::Dims dims_opt = dims;
      nvinfer1::Dims dims_max = dims;
      if (dims.d[0] == -1) {
        dims_min.d[0] = 1;
        dims_opt.d[0] = max_batch_size_;
        dims_max.d[0] = max_batch_size_;
        trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims_min);
        trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
        trt_profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims_max);
        dynamic_shape = true;
      }
    }
    if (dynamic_shape) {
      trt_config->addOptimizationProfile(trt_profile);
    }

    auto trt_engine = unique_pointer<nvinfer1::ICudaEngine>(trt_builder->buildEngineWithConfig(*trt_network, *trt_config));
    if (trt_engine == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not build Engine for fused node: " + fused_node->Name());
    }

    // Build TensorRT context
    auto trt_context = unique_pointer<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (trt_context == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                             "TensorRT EP could not build Execution Context for fused node: " + fused_node->Name());
    }

    // Get input shape and binding index
    int num_inputs = trt_network->getNbInputs();
    input_indexes.resize(num_inputs);
    input_dim_sizes.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const std::string& name = trt_network->getInput(i)->getName();
      size_t bindingIndex = trt_engine->getBindingIndex(name.c_str());
      nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(bindingIndex));
      auto iter = input_map.find(name);
      if (iter != input_map.end()) {
        input_indexes[bindingIndex] = iter->second;
      }
      size_t dim_size = 1;
      for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
        dim_size *= dimensions.d[j];
      }
      input_dim_sizes[bindingIndex] = dim_size;
    }

    // Get output shape and binding index
    int num_outputs = trt_network->getNbOutputs();
    output_indexes.resize(num_outputs);
    output_dim_sizes.resize(num_outputs);
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
      size_t dim_size = 1;
      for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
        output_shapes[bindingIndex].push_back(dimensions.d[j]);
        dim_size *= dimensions.d[j];
      }
      output_dim_sizes[bindingIndex] = dim_size;

      const auto& graph_output = model_proto.graph().output();
      const auto& tensor_type = graph_output[i].type().tensor_type();
      output_types[bindingIndex] = tensor_type.elem_type();
    }

    ORT_ENFORCE(trt_engine->getNbBindings() == (num_inputs + num_outputs));

    // Save engine, context and input/output info to map
    parsers_.emplace(fused_node->Name(), std::move(trt_parser));
    engines_.emplace(fused_node->Name(), std::move(trt_engine));
    contexts_.emplace(fused_node->Name(), std::move(trt_context));
    input_info_[fused_node->Name()].push_back(input_indexes);
    input_info_[fused_node->Name()].push_back(input_dim_sizes);
    output_info_[fused_node->Name()].push_back(output_indexes);
    output_info_[fused_node->Name()].push_back(output_dim_sizes);
    output_info_[fused_node->Name()].push_back(output_types);
    output_shapes_[fused_node->Name()] = output_shapes;

    // Create function state
    // TODO: remove default capture
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<TensorrtFuncState> p = onnxruntime::make_unique<TensorrtFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, parsers_[context->node_name].get(), engines_[context->node_name].get(), contexts_[context->node_name].get(),
            input_info_[context->node_name], output_info_[context->node_name], output_shapes_[context->node_name], &tensorrt_mu_};
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
      const std::vector<int>& input_indexes = (trt_state->input_info)[0];
      const std::vector<int>& output_indexes = (trt_state->output_info)[0];
      const std::vector<int>& output_types = (trt_state->output_info)[2];

      int num_binding_inputs = input_indexes.size();
      int num_binding_outputs = output_indexes.size();
      int total_bindings = num_binding_inputs + num_binding_outputs;
      std::vector<void*> buffers(total_bindings);

      bool dynamic_shape = false;
      if (!trt_state->context->allInputDimensionsSpecified()) {
        dynamic_shape = true;
      }

      // Get batch size and allocate cuda memory for inputs
      for (int i = 0, end = num_binding_inputs; i < end; ++i) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_indexes[i]);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);

        //Set dynamic shapes
        nvinfer1::Dims dimensions = trt_state->context->getEngine().getBindingDimensions(static_cast<int>(i));
        if (dynamic_shape) {
          for (int j = 0, end = tensor_shape.size(); j < end; ++j)
            dimensions.d[j] = tensor_shape[j];
          trt_state->context->setBindingDimensions(i, dimensions);
        }

        auto tensor_type = ort.GetTensorElementType(tensor_info);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

        if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i] = const_cast<float*>(ort.GetTensorData<float>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i] = const_cast<int8_t*>(ort.GetTensorData<int8_t>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          buffers[i] = const_cast<int32_t*>(ort.GetTensorData<int32_t>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
          int input_dim_size = 1;
          for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
            input_dim_size *= tensor_shape[j];
          }
          CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[i], input_dim_size * sizeof(int32_t)));
          cuda::Impl_Cast<int64_t, int32_t>(ort.GetTensorData<int64_t>(input_tensor), reinterpret_cast<int32_t*>(buffers[i]), input_dim_size);
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP input onnx tensor data type: " + std::to_string(tensor_type) + " not supported.");
        }
      }

      // Allocate CUDA memory for outputs
      std::vector<int> output_dim_size(num_binding_outputs, 1);
      std::vector<OrtValue*> output_tensor(num_binding_outputs, nullptr);
      for (int i = 0, end = num_binding_outputs; i < end; ++i) {
        // Set dynamic shapes
        nvinfer1::Dims dimensions = trt_state->context->getBindingDimensions(static_cast<int>(i + num_binding_inputs));
        for (int j = 0, end = trt_state->output_shapes[i].size(); j < end; ++j) {
          trt_state->output_shapes[i][j] = dimensions.d[j];
        }

        int output_index = output_indexes[i];
        output_tensor[i] = ort.KernelContext_GetOutput(context, output_index, trt_state->output_shapes[i].data(), trt_state->output_shapes[i].size());

        if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<float>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int8_t>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int32_t>(output_tensor[i]);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Allocate INT32 CUDA memory for INT64 output type because TensorRT doesn't fully support INT64
          for (int j = 0, end = dimensions.nbDims; j < end; ++j) {
            output_dim_size[i] *= dimensions.d[j];
          }
          CUDA_RETURN_IF_ERROR(cudaMalloc(&buffers[i + num_binding_inputs], output_dim_size[i] * sizeof(int32_t)));

        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP output onnx tensor data type: " + std::to_string(output_types[i]) + " not supported.");
        }
      }

      // Run TRT inference
      std::lock_guard<OrtMutex> lock(*(trt_state->tensorrt_mu_ptr));
      trt_state->context->enqueueV2(&buffers[0], nullptr, nullptr);

      // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
      for (int i = 0, end = num_binding_outputs; i < end; ++i) {
        if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          cuda::Impl_Cast<int32_t, int64_t>(reinterpret_cast<int32_t*>(buffers[i + num_binding_inputs]), ort.GetTensorMutableData<int64_t>(output_tensor[i]), output_dim_size[i]);
        }
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime
