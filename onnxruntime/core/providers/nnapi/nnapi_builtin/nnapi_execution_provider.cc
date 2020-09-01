// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nnapi_execution_provider.h"

#include "builders/model_builder.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

constexpr const char* NNAPI = "Nnapi";

NnapiExecutionProvider::NnapiExecutionProvider()
    : IExecutionProvider{onnxruntime::kNnapiExecutionProvider} {
  AllocatorCreationInfo device_info(
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(NNAPI, OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));

  AllocatorCreationInfo cpu_memory_info(
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
            OrtMemoryInfo(NNAPI, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      });

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

NnapiExecutionProvider::~NnapiExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
NnapiExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_view,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // TODO: Task 812756: NNAPI EP, add support for subgraph (If and Loop operators)
  if (graph_view.IsSubgraph()) {
    return result;
  }

  std::unordered_set<std::string> all_node_inputs;
  for (const auto& node : graph_view.Nodes()) {
    for (auto* input : node.InputDefs()) {
      all_node_inputs.insert(input->Name());
    }
  }

  nnapi::ModelBuilder builder(graph_view);
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
      const auto* node = graph_view.GetNode(node_index[index]);

      for (const auto* input : node->InputDefs()) {
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
        const auto* output = node->OutputDefs()[it->GetSrcArgIndex()];

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

      for (const auto* output : node->OutputDefs()) {
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
      } else if (std::find(graph_outputs.begin(), graph_outputs.end(), it->first) != graph_outputs.end()) {
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
    sub_graph->SetMetaDef(std::move(meta_def));

    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }

  return result;
}

static Status GetOutputBuffer(Ort::CustomOpApi& ort,
                              OrtKernelContext* context,
                              const nnapi::Model& model,
                              const std::string& output_name,
                              const std::vector<uint32_t>& output_shape,
                              const android::nn::wrapper::Type output_type,
                              void** output_buffer) ORT_MUST_USE_RESULT;
static Status GetOutputBuffer(Ort::CustomOpApi& ort,
                              OrtKernelContext* context,
                              const nnapi::Model& model,
                              const std::string& output_name,
                              const std::vector<uint32_t>& output_shape,
                              const android::nn::wrapper::Type output_type,
                              void** output_buffer) {
  using namespace android::nn::wrapper;
  std::vector<int64_t> int64_output_shape(output_shape.begin(),
                                          output_shape.end());
  auto output_idx = model.GetMappedOutputIdx(output_name);
  auto* output_tensor = ort.KernelContext_GetOutput(context, output_idx,
                                                    int64_output_shape.data(),
                                                    int64_output_shape.size());

  switch (output_type) {
    case Type::TENSOR_FLOAT32:
      *output_buffer = ort.GetTensorMutableData<float>(output_tensor);
      break;
    case Type::TENSOR_INT32:
      *output_buffer = ort.GetTensorMutableData<int32_t>(output_tensor);
      break;
    case Type::TENSOR_QUANT8_ASYMM:
      *output_buffer = ort.GetTensorMutableData<uint8_t>(output_tensor);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported output type: ", TypeToStr(output_type));
      break;
  }

  return Status::OK();
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
    {
      onnxruntime::GraphViewer graph_viewer(graph_body);
      nnapi::ModelBuilder builder(graph_viewer);
      builder.SetUseNCHW(false);
      builder.SetUseFp16(false);
      std::unique_ptr<nnapi::Model> nnapi_model;
      ORT_RETURN_IF_ERROR(builder.Compile(nnapi_model));

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
      // the `state` is a nnapi::model managed by unique_ptr
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      nnapi::Model* model = reinterpret_cast<nnapi::Model*>(state);
      const size_t num_inputs = ort.KernelContext_GetInputCount(context);
      const size_t num_outputs = ort.KernelContext_GetOutputCount(context);
      const auto& model_inputs = model->GetInputs();
      const auto& model_outputs = model->GetOutputs();

      ORT_RETURN_IF_NOT(model_inputs.size() <= num_inputs, "Inconsistent input sizes");
      ORT_RETURN_IF_NOT(model_outputs.size() == num_outputs, "Inconsistent output sizes");

      std::vector<nnapi::Execution::InputBuffer> inputs;
      inputs.reserve(model_inputs.size());
      for (size_t i = 0; i < model_inputs.size(); i++) {
        const auto& input_name = model_inputs[i];
        const auto& model_input_type = model->GetInputType(input_name);

        auto input_idx = model->GetMappedInputIdx(input_name);
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_idx);
        auto* tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        std::vector<uint32_t> dimensions;
        for (const auto& dim : ort.GetTensorShape(tensor_info))
          dimensions.push_back(static_cast<uint32_t>(dim));

        // NNAPI has strict input type requirements which separates tensor inputs and scalar inputs
        // For ONNX the we do not have clear line between scalar inputs and tensor inputs
        // Also NNAPI treats a tensor input with empty shape as dynamic shape input
        // Disable support of the scalar input (tensor input with an empty shape) for now
        // TODO, add support for ONNX scalar input (tensor input with an empty shape)
        if (dimensions.empty())
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "NNAPI does not support scalar input");

        // it is possible that the input has the detailed size while
        // the model has an operand with unknown size, use the size
        // of the actual input
        OperandType input_type = model_input_type;
        input_type.SetDimensions(dimensions);

        if (input_type.GetOperandBlobByteSize() == 0)
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The actual input cannot have 0 dim (dynamic)");

        if (input_type.dimensions != model_input_type.dimensions && model_input_type.GetOperandBlobByteSize() != 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "The actual input dimanesions should match the model input "
                                 "dimensions, or model input dimension has 0 (dynamic)");
        }

        const void* inputBuffer = ort.GetTensorData<void>(input_tensor);
        inputs.push_back({input_name, inputBuffer, std::move(input_type)});

        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }

      // From this point we will need to take the exclusive lock on the model until the Predict is
      // performed, to block other threads to perform Predict on the same model
      // TODO, investigate concurrent runs for different executions from the same model
      {
        std::unique_ptr<nnapi::Execution> execution;
        std::unique_lock<OrtMutex> lock(model->GetMutex());
        ORT_RETURN_IF_ERROR(model->PrepareForExecution(execution));

        ORT_RETURN_IF_ERROR(execution->SetInputBuffers(inputs));
        std::vector<nnapi::Execution::OutputBuffer> outputs;
        outputs.reserve(num_outputs);
        std::vector<int32_t> dynamic_shape_output_indices;
        std::vector<OperandType> dynamic_shape_output_types;
        std::vector<std::unique_ptr<uint8_t[]>> dynamic_shape_output_buffers;
        for (size_t i = 0; i < num_outputs; i++) {
          const auto output_name = model_outputs[i];
          const auto model_output_type = model->GetOutputType(output_name, *execution);
          const auto output_shape = model_output_type.dimensions;

          bool is_dynamic_shape_output = false;
          if (model_output_type.GetOperandBlobByteSize() == 0) {
            if (!model->SupportsDynamicOutputShape()) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                     "We do not support dynamic output shape or empty output for now");
            }

            is_dynamic_shape_output = true;
          }

          void* output_buffer = nullptr;
          size_t output_buffer_byte_size;
          if (!is_dynamic_shape_output) {
            ORT_RETURN_IF_ERROR(GetOutputBuffer(ort, context,
                                                *model,
                                                output_name, output_shape, model_output_type.type,
                                                &output_buffer));
            output_buffer_byte_size = model_output_type.GetOperandBlobByteSize();
          } else {
            // This output is dynamic (size unknown), will need allocate a buffer for the result
            // and copy the content to ORT output tensors afte the execution (will know output shape after the execution)
            output_buffer_byte_size = model->GetDynamicOutputBufferSize() * model_output_type.GetElementByteSize();
            std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[output_buffer_byte_size]);
            output_buffer = buffer_holder.get();
            dynamic_shape_output_types.push_back(model_output_type);
            dynamic_shape_output_indices.push_back(static_cast<int32_t>(i));
            dynamic_shape_output_buffers.push_back(std::move(buffer_holder));
          }

          outputs.push_back({output_buffer, std::move(model_output_type), output_buffer_byte_size});
        }

        ORT_RETURN_IF_ERROR(execution->SetOutputBuffers(outputs));
        std::vector<std::vector<uint32_t>> dynamic_output_shapes;
        ORT_RETURN_IF_ERROR(
            execution->Predict(dynamic_shape_output_indices, dynamic_output_shapes));

        // We have dynamic output buffers, need to copy the content from temp buffers to ORT output tensors
        for (size_t i = 0; i < dynamic_shape_output_indices.size(); i++) {
          const int32_t model_output_idx = dynamic_shape_output_indices[i];
          const auto output_name = model_outputs[model_output_idx];

          const auto& output_shape = dynamic_output_shapes[i];
          auto model_output_type = dynamic_shape_output_types[i];
          model_output_type.SetDimensions(output_shape);
          ORT_RETURN_IF_NOT(model_output_type.GetOperandBlobByteSize() != 0, "We do not support 0 size output for now");

          void* model_output_buffer = dynamic_shape_output_buffers[i].get();
          void* onnx_output_buffer = nullptr;
          ORT_RETURN_IF_ERROR(GetOutputBuffer(ort, context,
                                              *model,
                                              output_name, output_shape, model_output_type.type,
                                              &onnx_output_buffer));

          size_t output_buffer_byte_size = model_output_type.GetOperandBlobByteSize();
          memcpy(onnx_output_buffer, model_output_buffer, output_buffer_byte_size);
        }
      }
      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}  // namespace onnxruntime
}  // namespace onnxruntime
