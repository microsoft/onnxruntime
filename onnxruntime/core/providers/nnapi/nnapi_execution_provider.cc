// Copyright 2019 JD.com Inc. JD AI

#include "nnapi_execution_provider.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "dnnlibrary/ModelBuilder.h"
#include "dnnlibrary/OnnxReader.h"
#include "tools/onnx2daq/OnnxConverter.h"

namespace onnxruntime {

constexpr const char* NNAPI = "Nnapi";

NnapiExecutionProvider::NnapiExecutionProvider()
    : IExecutionProvider{onnxruntime::kNnapiExecutionProvider} {
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

std::vector<std::vector<int>> NnapiExecutionProvider::GetSupportedNodes(const ONNX_NAMESPACE::ModelProto& model_proto) const {
  dnn::OnnxConverter converter;
  return converter.GetSupportedNodes(model_proto);
}

std::vector<std::unique_ptr<ComputeCapability>>
NnapiExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // This method is based on that of TRT EP
  // Construct modelproto from graph
  onnxruntime::Model model(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
  onnxruntime::Graph& graph_build = model.MainGraph();
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
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

  const auto supported_nodes_vector = GetSupportedNodes(model_proto);

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();

  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;

  int counter = 0;

  for (const auto& group : supported_nodes_vector) {
    if (!group.empty()) {
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

      meta_def->since_version = 1;
      sub_graph->SetMetaDef(meta_def);

      result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

common::Status NnapiExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* fused_node : fused_nodes) {
    // Reconstruct graph proto from fused node's function body
    const auto* func_body = fused_node->GetFunctionBody();
    if (!func_body) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    }
    const Graph& graph_body = func_body->Body();
    onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(),
                             IOnnxRuntimeOpSchemaRegistryList(), graph_body.DomainToVersionMap());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
    *(model_proto.mutable_graph()) = graph_body.ToGraphProto();
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

    dnn::OnnxReader onnx_reader;
    dnn::ModelBuilder model_builder;
    onnx_reader.ReadOnnx(model_proto, model_builder);
    model_builder.AllowFp16(true);
    auto dnn_model = model_builder.Compile(model_builder.PREFERENCE_SUSTAINED_SPEED);
    dnn_models_.emplace(fused_node->Name(), std::move(dnn_model));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = dnn_models_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      // the `state` is a dnn::model managed by unique_ptr
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      dnn::Model* model = reinterpret_cast<dnn::Model*>(state);
      const size_t num_inputs = ort.KernelContext_GetInputCount(context);
      const size_t num_outputs = ort.KernelContext_GetOutputCount(context);
      ORT_ENFORCE(model->GetInputs().size() <= num_inputs, "Inconsistent input sizes");
      ORT_ENFORCE(model->GetOutputs().size() == num_outputs, "Inconsistent output sizes");
      // Maintain the created nhwc buffers so that they can be deleted after inferencing
      std::vector<float*> nhwc_inputs;
      std::vector<std::tuple<size_t, float*, std::vector<int64_t>>> nhwc_outputs;
      for (size_t i = 0; i < num_outputs; i++) {
        const auto output_name = model->GetOutputs()[i];
        const auto output_shape = model->GetShape(output_name);
        std::vector<int64_t> int64_output_shape(output_shape.begin(), output_shape.end());
        if (int64_output_shape.size() == 4) {
          // NHWC to NCHW
          std::swap(int64_output_shape[1], int64_output_shape[3]);
          std::swap(int64_output_shape[2], int64_output_shape[3]);
          float* nhwc_output = new float[model->GetSize(output_name)];
          model->SetOutputBuffer(i, nhwc_output);
          nhwc_outputs.push_back(std::make_tuple(i, nhwc_output, int64_output_shape));
        } else {
          auto* output_tensor = ort.KernelContext_GetOutput(context, i, int64_output_shape.data(), int64_output_shape.size());
          model->SetOutputBuffer(i, ort.GetTensorMutableData<float>(output_tensor));
        }
      }
      std::vector<float*> inputs;
      for (size_t i = 0; i < model->GetInputs().size(); i++) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
        float* input = const_cast<float*>(ort.GetTensorData<float>(input_tensor));

        const auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shape = ort.GetTensorShape(tensor_info);

        if (tensor_shape.size() == 4) {
          // Transpose nchw -> nhwc manually
          const int N = tensor_shape[0], C = tensor_shape[1], H = tensor_shape[2], W = tensor_shape[3];
          float* nhwc_input = new float[N * C * H * W];
          for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
              for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                  nhwc_input[n * H * W * C + h * W * C + w * C + c] = input[n * C * H * W + c * H * W + h * W + w];
                }
              }
            }
          }
          inputs.push_back(nhwc_input);
          nhwc_inputs.push_back(nhwc_input);
        } else {
          inputs.push_back(input);
        }
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }
      model->Predict(inputs);
      // Transpose nhwc -> nchw manually
      for (size_t i = 0; i < nhwc_outputs.size(); i++) {
        const auto output = nhwc_outputs[i];
        size_t index;
        float* nhwc_data;
        std::vector<int64_t> nchw_shape;
        std::tie(index, nhwc_data, nchw_shape) = output;
        auto* output_tensor = ort.KernelContext_GetOutput(context, index, nchw_shape.data(), nchw_shape.size());
        const int N = nchw_shape[0], C = nchw_shape[1], H = nchw_shape[2], W = nchw_shape[3];
        float* nchw_output = ort.GetTensorMutableData<float>(output_tensor);
        for (int n = 0; n < N; n++) {
          for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
              for (int w = 0; w < W; w++) {
                nchw_output[n * C * H * W + c * H * W + h * W + w] = nhwc_data[n * H * W * C + h * W * C + w * C + c];
              }
            }
          }
        }
      }
      for (auto nhwc_input : nhwc_inputs) {
        delete[] nhwc_input;
      }
      for (auto nhwc_output : nhwc_outputs) {
        delete[] std::get<1>(nhwc_output);
      }
      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime
