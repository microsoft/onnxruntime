// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nnapi_execution_provider.h"

#include "builders/model_builder.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"

namespace onnxruntime {

constexpr const char* NNAPI = "Nnapi";

NnapiExecutionProvider::NnapiExecutionProvider()
    : IExecutionProvider{onnxruntime::kNnapiExecutionProvider} {
  DeviceAllocatorRegistrationInfo device_info(
      {OrtMemTypeDefault,
       [](int) {
         return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(NNAPI, OrtAllocatorType::OrtDeviceAllocator));
       },
       std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(device_info));

  DeviceAllocatorRegistrationInfo cpu_memory_info(
      {OrtMemTypeCPUOutput,
       [](int) {
         return onnxruntime::make_unique<CPUAllocator>(
             OrtMemoryInfo(NNAPI, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
       },
       std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

NnapiExecutionProvider::~NnapiExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
NnapiExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_view,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Need access to model_path_
  for (const auto& tensor : graph_view.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() &&
        tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(WARNING) << "NNAPI: Initializers with external data"
                               " location are not currently supported";
      return result;
    }
  }

  // TODO, switch to use graph instead of model
  // This method is based on that of TRT EP
  // Construct modelproto from graph
  onnxruntime::Model model(graph_view.Name(), true, ModelMetaData(),
                           PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(),
                           graph_view.DomainToVersionMap(),
                           std::vector<ONNX_NAMESPACE::FunctionProto>(),
                           *GetLogger());
  std::unordered_set<std::string> all_node_inputs;
  onnxruntime::Graph& graph_build = model.MainGraph();
  for (const auto& node : graph_view.Nodes()) {
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node.InputDefs()) {
      auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
      all_node_inputs.insert(input->Name());
    }
    for (auto output : node.OutputDefs()) {
      auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    graph_build.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }
  //Add initializer to graph
  const auto& init_tensors = graph_view.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    graph_build.AddInitializedTensor(*(tensor.second));
  }

  ORT_ENFORCE(graph_build.Resolve().IsOK());
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  nnapi::ModelBuilder builder(model_proto);
  const auto supported_nodes_vector = builder.GetSupportedNodes();

  // Find inputs, initializers and outputs for each supported subgraph
  const std::vector<NodeIndex>& node_index = graph_view.GetNodesInTopologicalOrder();
  const auto graph_outputs = graph_view.GetOutputs();
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
      const auto node = graph_view.GetNode(node_index[index]);

      for (const auto& input : node->InputDefs()) {
        const auto it = fused_outputs.find(input);
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
      // If certain output is used more than once,
      // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
      // to the output list

      std::unordered_set<const NodeArg*> processed_outputs;
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        const auto node_idx = it->GetNode().Index();
        const auto output = node->OutputDefs()[it->GetSrcArgIndex()];

        if (node_set.find(node_idx) != node_set.end()) {
          const auto iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            fused_outputs[output] = output_order++;
          }
        } else {
          fused_outputs_to_add[output] = output_order++;
        }

        processed_outputs.insert(output);
      }

      for (const auto& output : node->OutputDefs()) {
        if (processed_outputs.find(output) != processed_outputs.end())
          continue;

        const auto iter = fused_inputs.find(output);
        if (iter != fused_inputs.end()) {
          fused_inputs.erase(iter);
          erased.insert(output);
        }
        // only when output is neither in input list nor erased list, add the output to output list
        else if (erased.find(output) == erased.end() && output->Exists()) {
          fused_outputs[output] = output_order++;
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
      if (all_node_inputs.find(it->first->Name()) != all_node_inputs.end()) {
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
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

    // meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    meta_def->since_version = 1;
    sub_graph->SetMetaDef(meta_def);

    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  return result;
}

std::string GetShape(const std::vector<uint32_t>& dimensions) {
  std::string ret = "";
  for (auto dim : dimensions)
    ret += std::to_string(dim) + " ";
  return "[" + ret + "]";
}

common::Status NnapiExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  using namespace android::nn::wrapper;
  for (const auto* fused_node : fused_nodes) {
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

    {
      nnapi::ModelBuilder builder(model_proto);
      std::unique_ptr<nnapi::Model> nnapi_model = builder.Compile();

      // Build map from input name to its index in input definitions
      {
        std::unordered_map<std::string, size_t> input_map;
        const auto& input_defs = fused_node->InputDefs();
        input_map.reserve(input_defs.size());
        for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
          input_map[input_defs[i]->Name()] = i;
        }
        nnapi_model->SetInputMap(std::move(input_map));
      }

      // Build map from output name to its index in output definitions
      {
        std::unordered_map<std::string, size_t> output_map;
        const auto& output_defs = fused_node->OutputDefs();
        output_map.reserve(output_defs.size());
        for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
          output_map[output_defs[i]->Name()] = i;
        }
        nnapi_model->SetOutputMap(std::move(output_map));
      }

      nnapi_models_.emplace(fused_node->Name(), std::move(nnapi_model));
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = nnapi_models_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      // the `state` is a dnn::model managed by unique_ptr
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};

      // TODO[VSO:798241], need to have exclusive access to the model within the scope of this compute_func
      nnapi::Model* model = reinterpret_cast<nnapi::Model*>(state);
      const size_t num_inputs = ort.KernelContext_GetInputCount(context);
      const size_t num_outputs = ort.KernelContext_GetOutputCount(context);
      ORT_ENFORCE(model->GetInputs().size() <= num_inputs, "Inconsistent input sizes");
      ORT_ENFORCE(model->GetOutputs().size() == num_outputs, "Inconsistent output sizes");

      std::vector<nnapi::Model::InputBuffer> inputs;
      inputs.reserve(model->GetInputs().size());
      for (size_t i = 0; i < model->GetInputs().size(); i++) {
        const auto& input_name = model->GetInputs()[i];
        const auto& model_input_type = model->GetInputType(input_name);

        auto input_idx = model->GetMappedInputIdx(input_name);
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_idx);
        const auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);
        std::vector<uint32_t> dimensions;
        for (const auto& dim : tensor_shape)
          dimensions.push_back(static_cast<uint32_t>(dim));

        // it is possible that the input has the detailed size while
        // the model has an operand with unknown size, use the size
        // of the actual input
        OperandType type(model_input_type.type, dimensions,
                         model_input_type.operandType.scale,
                         model_input_type.operandType.zeroPoint);

        if (type.dimensions != model_input_type.dimensions && model_input_type.GetOperandBlobByteSize() != 0) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "The actual input dimanesions should match the model input "
                        "dimensions, or model input dimension has 0 (dynamic)");
        }

        const void* inputBuffer = ort.GetTensorData<void>(input_tensor);
        inputs.push_back({inputBuffer, std::move(type)});

        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }

      model->SetInputBuffers(inputs);
      std::vector<nnapi::Model::OutputBuffer> outputs;
      outputs.reserve(num_outputs);
      for (size_t i = 0; i < num_outputs; i++) {
        const auto output_name = model->GetOutputs()[i];
        const auto model_output_type = model->GetOutputType(output_name);
        const auto output_shape = model_output_type.dimensions;

        std::vector<int64_t> int64_output_shape(output_shape.begin(),
                                                output_shape.end());
        auto output_idx = model->GetMappedOutputIdx(output_name);
        auto* output_tensor = ort.KernelContext_GetOutput(context, output_idx,
                                                          int64_output_shape.data(),
                                                          int64_output_shape.size());

        void* output_buffer = nullptr;
        switch (model_output_type.type) {
          case Type::TENSOR_FLOAT32:
            output_buffer = ort.GetTensorMutableData<float>(output_tensor);
            break;
          case Type::TENSOR_INT32:
            output_buffer = ort.GetTensorMutableData<int32_t>(output_tensor);
            break;
          default:
            ORT_THROW("Unsupported output type: " + TypeToStr(model_output_type.type));
            break;
        }

        if (model_output_type.GetOperandBlobByteSize() == 0) {
          return Status(common::ONNXRUNTIME, common::FAIL, "We do not support dynamic output shape for now");
        }

        outputs.push_back({output_buffer, std::move(model_output_type)});
      }

      model->SetOutputBuffers(outputs);

      model->Predict();

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}  // namespace onnxruntime
}  // namespace onnxruntime
