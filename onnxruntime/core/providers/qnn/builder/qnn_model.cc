// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_model.h"

#include <iostream>
#include "QnnOpDef.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/utils.h"

namespace onnxruntime {
namespace qnn {

bool QnnModel::GetGraphInfoFromModel(QnnModelWrapper& model_wrapper) {
  bool rt = true;

  graph_info_ = std::make_unique<GraphInfo>(model_wrapper.GetQnnGraph(),
                                            model_wrapper.GetQnnGraphName(),
                                            model_wrapper.GetGraphInputTensorWrappers(),
                                            model_wrapper.GetGraphOutputTensorWrappers(),
                                            model_wrapper.GetModelTensorsMap());
  if (graph_info_ == nullptr) {
    LOGS(logger_, ERROR) << "GetGraphInfoFromModel() failed to allocate graphsInfo. make_unique returned nullptr.";
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
  ORT_RETURN_IF_ERROR(ParseGraphInputOrOutput(input_defs, inputs_info_, model_input_index_map_, model_input_index_map_without_initializers_, true));

  auto output_defs = fused_node.OutputDefs();
  std::unordered_map<std::string, size_t> dummy;
  ORT_RETURN_IF_ERROR(ParseGraphInputOrOutput(output_defs, outputs_info_, model_output_index_map_, dummy));

  return Status::OK();
}

Status QnnModel::ParseGraphInputOrOutput(ConstPointerContainer<std::vector<NodeArg*>>& input_output_defs,
                                         std::unordered_map<std::string, OnnxTensorInfo>& input_output_info_table,
                                         std::unordered_map<std::string, size_t>& input_output_index_map,
                                         std::unordered_map<std::string, size_t>& input_output_index_map_without_initializers,
                                         bool is_input) {
  for (size_t i = 0, end = input_output_defs.size(), index = 0; i < end; ++i) {
    const auto& name = input_output_defs[i]->Name();
    if (is_input) {
      if (IsGraphInitializerInput(name)) {
        continue;  // exclude initializer inputs
      } else {
        input_output_index_map_without_initializers.emplace(name, index++);
      }
    }
    // Validate input/output shape
    LOGS(logger_, VERBOSE) << (is_input ? "input " : "output ") << i << " " << name;
    input_output_index_map.emplace(name, i);
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
                              const onnxruntime::Node& fused_node,
                              bool debug) {
  LOGS_DEFAULT(VERBOSE) << "ComposeGraph Graph name: " << graph_viewer.Name();
  const onnxruntime::AllocatorPtr& cpu_allocator = GetAllocator();

  // Holder for the NodeUnits in the graph, this will guarantee the NodeUnits is
  // valid throughout the lifetime of the ModelBuilder
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph_viewer);

  auto graph_name = graph_viewer.Name();
  ORT_RETURN_IF_ERROR(SetGraphInputOutputInfo(graph_viewer, fused_node));

  QNN_INTERFACE_VER_TYPE qnn_interface = qnn_backend_manager_->GetQnnInterface();
  Qnn_ContextHandle_t context = qnn_backend_manager_->GetQnnContext();
  QnnModelWrapper qnn_model_wrapper = QnnModelWrapper(graph_viewer, logger_, qnn_interface,
                                                      model_input_index_map_,
                                                      model_output_index_map_,
                                                      inputs_info_, outputs_info_,
                                                      initializer_inputs_, cpu_allocator);
  bool rt = true;
  rt = qnn_model_wrapper.Initialize(context, graph_name.c_str(), debug);
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
    LOGS_DEFAULT(VERBOSE) << " node name: [" << node->Name()
                          << " node optype: [" << node->OpType()
                          << "] as part of the NodeUnit type: [" << node_unit.OpType()
                          << "] name: [" << node_unit.Name()
                          << "]";
    if (node != &node_unit.GetNode()) {
      continue;
    }

    if (const auto* op_builder = GetOpBuilder(node->OpType())) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(qnn_model_wrapper, node_unit, logger_,
                                                        is_quantized_model_));
    }
  }

  LOGS(logger_, VERBOSE) << "GetGraphInfoFromModel started.";
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

Status QnnModel::ExecuteGraph(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  LOGS(logger_, VERBOSE) << "QnnModel::ExecuteGraphs";
  const size_t num_inputs = ort.KernelContext_GetInputCount(context);
  const size_t num_outputs = ort.KernelContext_GetOutputCount(context);
  ORT_RETURN_IF_NOT(model_input_index_map_.size() <= num_inputs, "Inconsistent input sizes");
  ORT_RETURN_IF_NOT(model_output_index_map_.size() == num_outputs, "Inconsistent output sizes");

  auto data_size = [&](const OrtValue* ort_tensor) -> size_t {
    auto* tensor_type_and_shape = ort.GetTensorTypeAndShape(ort_tensor);
    size_t length = ort.GetTensorShapeElementCount(tensor_type_and_shape);
    ONNXTensorElementDataType element_type = ort.GetTensorElementType(tensor_type_and_shape);
    size_t element_size = utils::GetElementSizeByType(element_type);
    return element_size * length;
  };

  for (const auto& tensor_wrapper : graph_info_->InputTensors()) {
    const std::string& model_input = tensor_wrapper.GetName();
    auto index = GetInputIndex(model_input);
    LOGS(logger_, VERBOSE) << "model_input = " << model_input << " index = " << index;
    const OrtValue* input_tensor = ort.KernelContext_GetInput(context, index);
    index = model_input_index_map_without_initializers_[model_input];
    ORT_ENFORCE(qnn_inputs_[index].clientBuf.dataSize == data_size(input_tensor),
                "ORT Tensor data size does not match QNN tensor data size");
    qnn_inputs_[index].clientBuf.data = const_cast<void*>(ort.GetTensorData<void>(input_tensor));
  }

  for (const auto& tensor_wrapper : graph_info_->OutputTensors()) {
    const std::string& tensor_name = tensor_wrapper.GetName();
    auto index = GetOutputIndex(tensor_name);
    LOGS(logger_, VERBOSE) << "model_output = " << tensor_name << " index = " << index;
    const auto& output_info = GetOutputInfo(tensor_name);
    const std::vector<int64_t>& output_shape = output_info->shape_;
    auto* output_tensor = ort.KernelContext_GetOutput(context, index,
                                                      output_shape.data(), output_shape.size());
    ORT_ENFORCE(qnn_outputs_[index].clientBuf.dataSize == data_size(output_tensor),
                "ORT Tensor data size does not match QNN tensor data size");
    qnn_outputs_[index].clientBuf.data = ort.GetTensorMutableData<void>(output_tensor);
  }

  LOGS(logger_, VERBOSE) << "Start execute QNN graph:" << graph_info_->Name();
  auto qnn_interface = qnn_backend_manager_->GetQnnInterface();
  auto profile_backend_handle = qnn_backend_manager_->GetQnnProfileHandle();
  Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
  executeStatus = qnn_interface.graphExecute(graph_info_->Graph(),
                                             qnn_inputs_.data(),
                                             static_cast<uint32_t>(qnn_inputs_.size()),
                                             qnn_outputs_.data(),
                                             static_cast<uint32_t>(qnn_outputs_.size()),
                                             profile_backend_handle,
                                             nullptr);

  ORT_RETURN_IF_ERROR(qnn_backend_manager_->ExtractBackendProfilingInfo());
  if (QNN_GRAPH_NO_ERROR != executeStatus) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN graph execute error.");
  }

  return Status::OK();
}

Status QnnModel::GetQnnTensorDataLength(uint32_t* data_dimensions,
                                        uint32_t rank,
                                        Qnn_DataType_t data_type,
                                        size_t& data_length) const {
  ORT_RETURN_IF(nullptr == data_dimensions, "Tensor dimensions is nullptr");

  data_length = utils::GetElementSizeByType(data_type);

  for (size_t r = 0; r < rank; r++) {
    data_length *= data_dimensions[r];
  }

  return Status::OK();
}

// Setup details for Qnn_Tensor_t for execution
// based on information in QnnTensorWrapper provided by model.so.
Status QnnModel::SetupTensors(std::vector<Qnn_Tensor_t>& qnn_tensors,
                              const std::vector<QnnTensorWrapper>& tensor_wrappers,
                              bool is_input) {
  size_t tensor_count = tensor_wrappers.size();
  ORT_RETURN_IF(0 == tensor_count, "Zero tensor size!");
  qnn_tensors.resize(tensor_count);

  for (auto& tensor_wrapper : tensor_wrappers) {
    const auto& wrapper_tensor = tensor_wrapper.GetQnnTensor();
    size_t length = 0;
    ORT_RETURN_IF_ERROR(GetQnnTensorDataLength(wrapper_tensor.maxDimensions,
                                               wrapper_tensor.rank,
                                               wrapper_tensor.dataType, length));
    auto name = tensor_wrapper.GetName();
    auto index = is_input ? model_input_index_map_without_initializers_[name] : model_output_index_map_[name];
    qnn_tensors[index] = wrapper_tensor;
    qnn_tensors[index].currentDimensions = qnn_tensors[index].maxDimensions;
    qnn_tensors[index].clientBuf.dataSize = static_cast<uint32_t>(length);
  }
  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
