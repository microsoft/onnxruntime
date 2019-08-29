// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorrt_execution_provider.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_fence.h"
#include "core/platform/env.h"
#include "core/common/status.h"
#include "onnx/shape_inference/implementation.h"
#include "cuda_runtime_api.h"
#include "gsl/pointers"
#include "core/graph/model.h"
#include "cuda_runtime_api.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& GetTensorrtLogger() {
  static TensorrtLogger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
  return trt_logger;
}

#define CHECK_CUDA(call)                                        \
  do {                                                          \
    cudaError_t status = call;                                  \
    if (status != cudaSuccess) {                                \
      return common::Status(common::ONNXRUNTIME, common::FAIL); \
    }                                                           \
  } while (0)

TensorrtExecutionProvider::TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTensorrtExecutionProvider}, device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));

  DeviceAllocatorRegistrationInfo default_allocator_info(
      {OrtMemTypeDefault, [](int id) { return std::make_unique<CUDAAllocator>(id, TRT); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(default_allocator_info, device_id_));

  DeviceAllocatorRegistrationInfo pinned_allocator_info(
      {OrtMemTypeCPUOutput, [](int) { return std::make_unique<CUDAPinnedAllocator>(0, TRT_PINNED); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(pinned_allocator_info, device_id_));
}

TensorrtExecutionProvider::~TensorrtExecutionProvider() {}

std::unique_ptr<IndexedSubGraph> TensorrtExecutionProvider::GetSubGraph(SubGraph_t graph_nodes_index, int& kernels_index, const onnxruntime::GraphViewer& graph) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.first.size());
  for (const auto& index : graph_nodes_index.first) {
    node_set.insert(node_index[index]);
  }
  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

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
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
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
        string string_buf;
        model_proto.SerializeToString(&string_buf);

        // Get supported node list recursively
        SubGraphCollection_t parser_nodes_list;
        TensorrtLogger& trt_logger = GetTensorrtLogger();
        auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
        auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetwork());
        auto trt_parser = unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
        trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_list);

        SubGraphCollection_t next_nodes_list;
        const onnxruntime::GraphViewer graph_viewer(graph_build);
        next_nodes_list = GetSupportedList(parser_nodes_list, iterations, max_iterations, graph_viewer, early_termination);
        for (int i = 0, end = next_nodes_list.size(); i < end; ++i) {
          for (int j = 0, end = next_nodes_list[i].first.size(); j < end; ++j) {
            next_nodes_list[i].first[j] = group.first[next_nodes_list[i].first[j]];
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
  string string_buf;
  model_proto.SerializeToString(&string_buf);

  // Get supported node list
  SubGraphCollection_t parser_nodes_vector;
  TensorrtLogger& trt_logger = GetTensorrtLogger();
  auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
  auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetwork());
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
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
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
    string string_buf;
    model_proto.SerializeToString(&string_buf);

    // Create TensorRT engine
    TensorrtLogger& trt_logger = GetTensorrtLogger();
    auto trt_builder = unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
    auto trt_network = unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetwork());
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
    trt_builder->setMaxWorkspaceSize(max_workspace_size_);
    auto trt_engine = unique_pointer<nvinfer1::ICudaEngine>(trt_builder->buildCudaEngine(*trt_network.get()));
    ORT_ENFORCE(trt_engine != nullptr);

    // Build TensorRT context
    auto trt_context = unique_pointer<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    ORT_ENFORCE(trt_context != nullptr);

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

      const auto& tensor_shape = tensor_type.shape();
      if (tensor_shape.dim_size() == 1 && output_shapes[bindingIndex].back() == 1) {
        output_shapes[bindingIndex].pop_back();
      }
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
      std::unique_ptr<TensorrtFuncState> p = std::make_unique<TensorrtFuncState>();
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
      std::vector<std::vector<int64_t>> output_shapes = trt_state->output_shapes;

      int num_binding_inputs = input_indexes.size();
      int num_binding_outputs = output_indexes.size();
      int total_bindings = num_binding_inputs + num_binding_outputs;
      std::vector<void*> buffers(total_bindings);
      int batch_size = 1;

      // Get batch size and allocate cuda memory for inputs
      for (int i = 0, end = num_binding_inputs; i < end; ++i) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_indexes[i]);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);
        auto tensor_type = ort.GetTensorElementType(tensor_info);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

        const int input_batch_size = tensor_shape[0];
        if (i > 0 && batch_size != input_batch_size) {
          ORT_THROW("Input batch size is inconsistent");
        }
        batch_size = input_batch_size;

        if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i] = const_cast<float*>(ort.GetTensorData<float>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i] = const_cast<int8_t*>(ort.GetTensorData<int8_t>(input_tensor));
        } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          buffers[i] = const_cast<int32_t*>(ort.GetTensorData<int32_t>(input_tensor));
        } else {
          return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
        }
      }

      // Allocate cuda memory for outputs
      for (int i = 0, end = num_binding_outputs; i < end; ++i) {
        int output_index = output_indexes[i];
        output_shapes[i].insert(output_shapes[i].begin(), batch_size);

        OrtValue* output_tensor = ort.KernelContext_GetOutput(context, output_index, output_shapes[i].data(), output_shapes[i].size());
        if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<float>(output_tensor);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int8_t>(output_tensor);
        } else if (output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || output_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          buffers[i + num_binding_inputs] = ort.GetTensorMutableData<int32_t>(output_tensor);
        } else {
          return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
        }
      }

      // Run TRT inference
      std::lock_guard<OrtMutex> lock(*(trt_state->tensorrt_mu_ptr));
      bool ret = trt_state->context->enqueue(batch_size, &buffers[0], nullptr, nullptr);
      if (!ret) {
        if (trt_state->context->getEngine().getMaxBatchSize() < batch_size) {
	  return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
			        "TRT enqueue failed: Set ORT_TENSORRT_MAX_BATCH_SIZE environment variable to at least " + to_string(batch_size));
        }
        return common::Status(common::ONNXRUNTIME, common::FAIL, "Failed to enqueue to TRT execution context.");
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime
