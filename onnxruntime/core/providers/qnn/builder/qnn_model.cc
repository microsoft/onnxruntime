// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_model.h"

#include <iostream>
#include "QnnOpDef.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

bool QnnModel::GetGraphInfoFromModel(QnnModelWrapper& model_wrapper) {
  bool rt = true;

  graph_info_ = std::make_unique<GraphInfo>(model_wrapper.GetQnnGraph(),
                                            model_wrapper.GetQnnGraphName(),
                                            std::move(model_wrapper.GetGraphInputTensorWrappers()),
                                            std::move(model_wrapper.GetGraphOutputTensorWrappers()));
  if (graph_info_ == nullptr) {
    LOGS(logger_, ERROR) << "GetGraphInfoFromModel() failed to allocate GraphInfo.";
    return false;
  }

  return rt;
}

Status QnnModel::SetGraphInputOutputInfo(const GraphViewer& graph_viewer,
                                         const onnxruntime::Node& fused_node) {
  auto graph_initializers = graph_viewer.GetAllInitializedTensors();
  for (auto graph_ini : graph_initializers) {
    initializer_inputs_.emplace(graph_ini.first);
  }
  auto input_defs = fused_node.InputDefs();
  ORT_RETURN_IF_ERROR(ParseGraphInputOrOutput(input_defs, inputs_info_, model_input_index_map_, true));

  auto output_defs = fused_node.OutputDefs();
  ORT_RETURN_IF_ERROR(ParseGraphInputOrOutput(output_defs, outputs_info_, model_output_index_map_));

  return Status::OK();
}

Status QnnModel::ParseGraphInputOrOutput(ConstPointerContainer<std::vector<NodeArg*>>& input_output_defs,
                                         std::unordered_map<std::string, OnnxTensorInfo>& input_output_info_table,
                                         std::unordered_map<std::string, size_t>& input_output_index_map,
                                         bool is_input) {
  for (size_t i = 0, end = input_output_defs.size(), index = 0; i < end; ++i) {
    const auto& name = input_output_defs[i]->Name();
    if (is_input) {
      if (IsGraphInitializerInput(name)) {
        continue;  // exclude initializer inputs
      }
    }
    // Validate input/output shape
    LOGS(logger_, VERBOSE) << (is_input ? "input " : "output ") << i << " " << name;
    input_output_index_map.emplace(name, index++);
    const auto* shape_proto = input_output_defs[i]->Shape();  // consider use qnn_model_wrapper.GetOnnxShape
    ORT_RETURN_IF(shape_proto == nullptr, "shape_proto cannot be null for output: ", name);

    const auto& dims = shape_proto->dim();
    std::vector<int64_t> shape;
    shape.reserve(dims.size());
    for (const auto& dim : dims) {
      ORT_RETURN_IF_NOT(dim.has_dim_value(), "Dynamic shape is not supported yet, for output: ", name);
      shape.push_back(dim.dim_value());
    }
    const auto* type_proto = input_output_defs[i]->TypeAsProto();
    int32_t data_type = type_proto->tensor_type().elem_type();
    // use index i so that for graph input, it has initializers included
    input_output_info_table.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(i, data_type, std::move(shape)));
  }

  return Status::OK();
}

const NodeUnit& QnnModel::GetNodeUnit(const Node* node,
                                      const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map) const {
  const auto node_unit_it = node_unit_map.find(node);
  ORT_ENFORCE(node_unit_it != node_unit_map.end(), "Node does not have corresponding NodeUnit.");
  return *node_unit_it->second;
}

Status QnnModel::ComposeGraph(const GraphViewer& graph_viewer,
                              const onnxruntime::Node& fused_node) {
  LOGS(logger_, VERBOSE) << "ComposeGraph Graph name: " << graph_viewer.Name();

  // Holder for the NodeUnits in the graph, this will guarantee the NodeUnits is
  // valid throughout the lifetime of the ModelBuilder
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph_viewer);

  const auto& graph_name = graph_viewer.Name();
  ORT_RETURN_IF_ERROR(SetGraphInputOutputInfo(graph_viewer, fused_node));

  QnnModelWrapper qnn_model_wrapper = QnnModelWrapper(graph_viewer, logger_,
                                                      qnn_backend_manager_->GetQnnInterface(),
                                                      qnn_backend_manager_->GetQnnBackendHandle(),
                                                      model_input_index_map_,
                                                      model_output_index_map_,
                                                      initializer_inputs_,
                                                      qnn_backend_manager_->GetQnnBackendType());
  bool rt = true;
  rt = qnn_model_wrapper.CreateQnnGraph(qnn_backend_manager_->GetQnnContext(), graph_name);
  if (!rt) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to initialize qnn_model_wrapper.");
  }

  // Op builer
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer.GetNode(node_indices[i]));

    // Check whether it's part of NodeUnit
    const NodeUnit& node_unit = GetNodeUnit(node, node_unit_map);
    // Q, DQ nodes in the node unit only carry the quantization parameters
    // Add the QNN node when it is the target node (It's a normal node or a singel Q/DQ node)
    const std::string& op_type = node_unit.OpType();
    LOGS(logger_, VERBOSE) << " node name: [" << node->Name()
                           << "] node optype: [" << op_type
                           << "] as part of the NodeUnit type: [" << node_unit.OpType()
                           << "] name: [" << node_unit.Name()
                           << "]";
    if (node != &node_unit.GetNode()) {
      continue;
    }

    if (const auto* op_builder = GetOpBuilder(op_type)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(qnn_model_wrapper, node_unit, logger_));
    }
  }

  ORT_RETURN_IF_NOT(qnn_model_wrapper.ComposeQnnGraph(), "Failed to compose Qnn graph.");

  rt = GetGraphInfoFromModel(qnn_model_wrapper);
  if (!rt) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetGraphInfoFromModel failed.");
  }
  LOGS(logger_, VERBOSE) << "GetGraphInfoFromModel completed.";
  return Status::OK();
}

Status QnnModel::FinalizeGraphs() {
  LOGS(logger_, VERBOSE) << "FinalizeGraphs started.";
  Qnn_ErrorHandle_t status = qnn_backend_manager_->GetQnnInterface().graphFinalize(graph_info_->Graph(),
                                                                                   qnn_backend_manager_->GetQnnProfileHandle(),
                                                                                   nullptr);
  if (QNN_GRAPH_NO_ERROR != status) {
    LOGS(logger_, ERROR) << "Failed to finalize QNN graph.";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to finalize QNN graph.");
  }

  ORT_RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo());

  LOGS(logger_, VERBOSE) << "FinalizeGraphs completed.";
  return Status::OK();
}

Status QnnModel::SetupQnnInputOutput() {
  LOGS(logger_, VERBOSE) << "Setting up QNN input/output for graph: " << graph_info_->Name();

  auto result = SetupTensors(qnn_inputs_, graph_info_->InputTensors());

  if (Status::OK() != result) {
    LOGS(logger_, ERROR) << "Failed to setup QNN input output tensors for graph: " << graph_info_->Name();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to setup QNN input tensors!");
  }

  result = SetupTensors(qnn_outputs_, graph_info_->OutputTensors(), false);
  if (Status::OK() != result) {
    LOGS(logger_, ERROR) << "Failed to setup QNN input output tensors for graph: " << graph_info_->Name();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to setup QNN output tensors!");
  }

  return Status::OK();
}

Status QnnModel::ExecuteGraph(const Ort::KernelContext& context) {
  LOGS(logger_, VERBOSE) << "QnnModel::ExecuteGraphs";
  const size_t num_inputs = context.GetInputCount();
  const size_t num_outputs = context.GetOutputCount();
  ORT_RETURN_IF_NOT(qnn_inputs_.size() <= num_inputs, "Inconsistent input sizes");
  ORT_RETURN_IF_NOT(qnn_outputs_.size() == num_outputs, "Inconsistent output sizes");

  using namespace qnn::utils;
  auto TensorDataSize = [&](auto ort_tensor) -> size_t {
    auto tensor_type_and_shape = ort_tensor.GetTensorTypeAndShapeInfo();
    size_t length = tensor_type_and_shape.GetElementCount();
    ONNXTensorElementDataType element_type = tensor_type_and_shape.GetElementType();
    size_t element_size = GetElementSizeByType(element_type);
    return element_size * length;
  };

  for (auto& qnn_input_tensor : qnn_inputs_) {
    const std::string& model_input_name(GetQnnTensorName(qnn_input_tensor));
    auto index = GetOrtInputIndex(model_input_name);
    LOGS(logger_, VERBOSE) << "model_input = " << model_input_name << " index = " << index;
    auto ort_input_tensor = context.GetInput(index);
    auto qnn_tensor_size = GetQnnTensorClientBuf(qnn_input_tensor).dataSize;
    auto ort_tensor_size = TensorDataSize(ort_input_tensor);
    LOGS(logger_, VERBOSE) << "Qnn tensor size: " << qnn_tensor_size << "Ort tensor size: " << ort_tensor_size;
    ORT_ENFORCE(qnn_tensor_size == ort_tensor_size,
                "ORT Tensor data size does not match QNN tensor data size.");
    SetQnnTensorClientBufData(qnn_input_tensor,
                              const_cast<void*>(ort_input_tensor.GetTensorData<void>()));
  }

  for (auto& qnn_output_tensor : qnn_outputs_) {
    const std::string& model_output_name(GetQnnTensorName(qnn_output_tensor));
    auto index = GetOutputIndex(model_output_name);
    LOGS(logger_, VERBOSE) << "model_output = " << model_output_name << " index = " << index;
    const auto& output_info = GetOutputInfo(model_output_name);
    const std::vector<int64_t>& output_shape = output_info->shape_;
    auto output_tensor = context.GetOutput(index, output_shape.data(), output_shape.size());
    auto qnn_tensor_size = GetQnnTensorClientBuf(qnn_output_tensor).dataSize;
    auto ort_tensor_size = TensorDataSize(output_tensor);
    LOGS(logger_, VERBOSE) << "Qnn tensor size: " << qnn_tensor_size << "Ort tensor size: " << ort_tensor_size;
    ORT_ENFORCE(qnn_tensor_size == ort_tensor_size,
                "ORT Tensor data size does not match QNN tensor data size");
    SetQnnTensorClientBufData(qnn_output_tensor,
                              const_cast<void*>(output_tensor.GetTensorData<void>()));
  }

  LOGS(logger_, VERBOSE) << "Start execute QNN graph:" << graph_info_->Name();
  auto qnn_interface = qnn_backend_manager_->GetQnnInterface();
  auto profile_backend_handle = qnn_backend_manager_->GetQnnProfileHandle();
  Qnn_ErrorHandle_t execute_status = QNN_GRAPH_NO_ERROR;
  execute_status = qnn_interface.graphExecute(graph_info_->Graph(),
                                              qnn_inputs_.data(),
                                              static_cast<uint32_t>(qnn_inputs_.size()),
                                              qnn_outputs_.data(),
                                              static_cast<uint32_t>(qnn_outputs_.size()),
                                              profile_backend_handle,
                                              nullptr);

  ORT_RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo());
  if (QNN_GRAPH_NO_ERROR != execute_status) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN graph execute error. Error code: ", execute_status);
  }

  return Status::OK();
}

Status QnnModel::GetQnnTensorDataLength(const std::vector<uint32_t>& dims,
                                        Qnn_DataType_t data_type,
                                        size_t& data_length) const {
  ORT_RETURN_IF(dims.empty(), "Tensor dimensions is nullptr");

  data_length = utils::GetElementSizeByType(data_type);

  for (size_t r = 0; r < dims.size(); r++) {
    data_length *= dims[r];
  }

  return Status::OK();
}

// Setup details for Qnn_Tensor_t for execution
// based on information in QnnTensorWrapper
Status QnnModel::SetupTensors(std::vector<Qnn_Tensor_t>& qnn_tensors,
                              const std::vector<QnnTensorWrapper>& tensor_wrappers,
                              bool is_input) {
  size_t tensor_count = tensor_wrappers.size();
  ORT_RETURN_IF(0 == tensor_count, "Zero tensor size!");
  qnn_tensors.resize(tensor_count);

  for (auto& tensor_wrapper : tensor_wrappers) {
    size_t length = 0;
    using namespace qnn::utils;
    ORT_RETURN_IF_ERROR(GetQnnTensorDataLength(tensor_wrapper.GetTensorDims(),
                                               tensor_wrapper.GetTensorDataType(),
                                               length));
    auto tensor_name = tensor_wrapper.GetName();
    auto index = is_input ? GetGraphInputIndex(tensor_name) : GetOutputIndex(tensor_name);
    qnn_tensors[index] = tensor_wrapper.GetQnnTensor();
    SetQnnTensorClientBufSize(qnn_tensors[index], static_cast<uint32_t>(length));
  }
  return Status::OK();
}

Status QnnModel::DeserializeGraphInfoFromBinaryInfo(const QnnSystemContext_GraphInfo_t& qnn_sys_ctx_graph_info) {
  std::vector<QnnTensorWrapper> input_tensor_wrappers;
  std::vector<QnnTensorWrapper> output_tensor_wrappers;

  std::string graph_name;
  if (qnn_sys_ctx_graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    graph_name.assign(qnn_sys_ctx_graph_info.graphInfoV1.graphName);
    auto graph_input_num = qnn_sys_ctx_graph_info.graphInfoV1.numGraphInputs;
    auto graph_output_num = qnn_sys_ctx_graph_info.graphInfoV1.numGraphOutputs;
    ORT_RETURN_IF(nullptr == qnn_sys_ctx_graph_info.graphInfoV1.graphInputs, "Graph from cached context doesn't have any inputs.");
    ORT_RETURN_IF(nullptr == qnn_sys_ctx_graph_info.graphInfoV1.graphOutputs, "Graph from cached context doesn't have any outputs.");

    // Copy graph input
    Qnn_Tensor_t* input_tensors = qnn_sys_ctx_graph_info.graphInfoV1.graphInputs;
    for (size_t i = 0; i < graph_input_num; ++i) {
      QnnTensorWrapper tensorwrapper(input_tensors[i]);
      input_tensor_wrappers.push_back(std::move(tensorwrapper));
    }

    // Copy graph output
    Qnn_Tensor_t* output_tensors = qnn_sys_ctx_graph_info.graphInfoV1.graphOutputs;
    for (size_t i = 0; i < graph_output_num; ++i) {
      QnnTensorWrapper tensorwrapper(output_tensors[i]);
      output_tensor_wrappers.push_back(std::move(tensorwrapper));
    }
  }
  Qnn_GraphHandle_t graph;
  auto qnn_interface = qnn_backend_manager_->GetQnnInterface();
  qnn_interface.graphRetrieve(qnn_backend_manager_->GetQnnContext(),
                              graph_name.c_str(), &graph);

  graph_info_ = std::make_unique<GraphInfo>(graph,
                                            graph_name,
                                            std::move(input_tensor_wrappers),
                                            std::move(output_tensor_wrappers));
  ORT_RETURN_IF(graph_info_ == nullptr, "Failed to allocate GraphInfo");

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
