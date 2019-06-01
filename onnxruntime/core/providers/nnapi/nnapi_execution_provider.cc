#include "nnapi_execution_provider.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "dnnlibrary/ModelBuilder.h"
#include "dnnlibrary/OnnxReader.h"

namespace onnxruntime {

constexpr const char* NNAPI = "Nnapi";

NnapiExecutionProvider::NnapiExecutionProvider()
    : IExecutionProvider{onnxruntime::kNnapiExecutionProvider} {
  DeviceAllocatorRegistrationInfo device_info{OrtMemTypeDefault,
                                              [](int) { return std::make_unique<CPUAllocator>(
                                                  std::make_unique<OrtAllocatorInfo>(NNAPI, 
                                                    OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault)); },
                                              std::numeric_limits<size_t>::max()};
  InsertAllocator(
      std::shared_ptr<IArenaAllocator>(
          std::make_unique<DummyArena>(device_info.factory(0))));
}

NnapiExecutionProvider::~NnapiExecutionProvider() {}

Status NnapiExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  if (!(strcmp(src.Location().name, NNAPI) == 0 && strcmp(dst.Location().name, CPU) == 0) &&
      !(strcmp(src.Location().name, CPU) == 0 && strcmp(dst.Location().name, NNAPI) == 0) &&
      !(strcmp(src.Location().name, NNAPI) == 0 && strcmp(dst.Location().name, NNAPI) == 0)) {
    ORT_NOT_IMPLEMENTED(src.Location().name, " copy to ", dst.Location().name, " is not implemented");
  }

  // Todo: Copy for now. May optimize later to avoid copy.
  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  memcpy(dst_data, src_data, bytes);

  return Status::OK();
}

std::shared_ptr<KernelRegistry> NnapiExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
NnapiExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Construct modelproto from graph
  onnxruntime::Model model(graph.Name(), true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), graph.DomainToVersionMap());
  onnxruntime::Graph& graph_build = model.MainGraph();
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  for (const auto& node : graph.Nodes()) {
    graph_build.AddNode(node);
  }
  ORT_ENFORCE(graph_build.Resolve().IsOK());
  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  try {
    dnn::OnnxReader onnx_reader;
    dnn::ModelBuilder model_builder;
    onnx_reader.ReadOnnx(model_proto, model_builder);
  } catch (const std::invalid_argument&) {
    return {};
  }

  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;

  int counter = 0;

  std::vector<std::vector<int>> supported_nodes_vector;
  std::vector<int> temp(model_proto.graph().node_size());
  std::iota(temp.begin(), temp.end(), 0);
  supported_nodes_vector.push_back(temp);

  for (const auto& group : supported_nodes_vector) {
    if (!group.empty()) {
      std::unordered_set<size_t> node_set;
      node_set.reserve(group.size());
      for (const auto& index : group) {
        node_set.insert(node_index[index]);
      }
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
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
        outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
      }

      // Assign inputs and outputs to subgraph's meta_def
      auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
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

      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

common::Status NnapiExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  ORT_UNUSED_PARAMETER(node_compute_funcs);
  for (const auto* fused_node : fused_nodes) {
    std::vector<int> input_indexes;
    std::vector<int> input_dim_sizes;
    std::vector<int> output_indexes;
    std::vector<int> output_dim_sizes;
    std::vector<std::vector<int64_t>> output_shapes;

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
    onnxruntime::Model model(graph_body.Name(), true, ModelMetaData(),
                             IOnnxRuntimeOpSchemaRegistryList(), graph_body.DomainToVersionMap());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
    *(model_proto.mutable_graph()) = graph_body.ToGraphProto();
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

    dnn::OnnxReader onnx_reader;
    dnn::ModelBuilder model_builder;
    onnx_reader.ReadOnnx(model_proto, model_builder);
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
      ORT_ENFORCE(model->GetOutputs().size() == num_outputs, "Inconsistent output sizes");
      for (size_t i = 0; i < num_outputs; i++) {
        const auto output_name = model->GetOutputs()[i];
        const auto output_shape = model->GetShape(output_name);
        std::vector<int64_t> int64_output_shape(output_shape.begin(), output_shape.end());
        if (int64_output_shape.size() == 4) {
            // NHWC to NCHW
            std::swap(int64_output_shape[1], int64_output_shape[3]);
            std::swap(int64_output_shape[2], int64_output_shape[3]);
        }
        auto *output_tensor = ort.KernelContext_GetOutput(context, i, int64_output_shape.data(), int64_output_shape.size());
        model->SetOutputBuffer(i, ort.GetTensorMutableData<float>(output_tensor));
      }
      std::vector<float *> inputs;
      for (size_t i = 0; i < num_inputs; i++) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
        float* input = const_cast<float *>(ort.GetTensorData<float>(input_tensor));
        inputs.push_back(input);
      }
      model->Predict(inputs);
      return 0;
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime
