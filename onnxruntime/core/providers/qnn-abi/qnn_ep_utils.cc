// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_ep_utils.h"
#include <iostream>
#include <string>

namespace onnxruntime {
namespace QDQ {

void OrtSelectors::RegisterSelector(const OrtOpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                                    std::unique_ptr<OrtNodeGroupSelector> selector_in) {
  auto entry = std::make_unique<OrtOpVersionsAndSelector>(
      ops_and_versions_in,
      std::move(selector_in));

  selectors_set_.push_back(std::move(entry));
}

// Helper function to get the number of actual values (inputs or outputs) for a node
int NumActualValues(const OrtNode* node, const OrtApi& ort_api, bool input) {
  size_t num_defs = 0;
  OrtStatus* status = nullptr;

  if (input) {
    status = ort_api.Node_GetNumInputs(node, &num_defs);
  } else {
    status = ort_api.Node_GetNumOutputs(node, &num_defs);
  }

  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return 0;
  }

  return static_cast<int>(num_defs);
}

// Helper function to get the data type of a node's input or output
int32_t GetNodeIODataType(const OrtNode* node, const OrtApi& ort_api, bool is_input, int index) {
  // Get the inputs or outputs as OrtValueInfo instances
  OrtStatus* status = nullptr;

  size_t num_defs = 0;

  if (is_input) {
    status = ort_api.Node_GetNumInputs(node, &num_defs);
  } else {
    status = ort_api.Node_GetNumOutputs(node, &num_defs);
  }

  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return -1;
  }

  if (index >= static_cast<int>(num_defs)) {
    return -1;
  }

  std::vector<const OrtValueInfo*> io_array(num_defs);
  if (is_input) {
    status = ort_api.Node_GetInputs(node, io_array.data(), io_array.size());
  } else {
    status = ort_api.Node_GetOutputs(node, io_array.data(), io_array.size());
  }
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return -1;
  }

  // Get the OrtValueInfo at the specified index
  const OrtValueInfo* value_info = io_array[index];

  // Get the type info from the value info
  const OrtTypeInfo* type_info = nullptr;
  status = ort_api.GetValueInfoTypeInfo(value_info, &type_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return -1;
  }

  // Get the tensor element data type from the type info
  ONNXTensorElementDataType element_type;
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  status = ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info);
  if (status != nullptr || tensor_info == nullptr) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return -1;
  }

  status = ort_api.GetTensorElementType(tensor_info, &element_type);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return -1;
  }

  return static_cast<int32_t>(element_type);
}

// Helper function to check if a data type is a 16-bit integer type
bool Is16BitIntType(int32_t data_type) {
  return (data_type == 5) ||  // INT16
         (data_type == 17);   // UINT16
}

// Helper function to check if a data type is a 4-bit integer type
bool Is4BitIntType(int32_t data_type) {
  return (data_type == 20) ||  // INT4
         (data_type == 21);    // UINT4
}

// Helper function to get a constant initializer from a node's input
const OrtValue* GetConstantInitializer(const OrtGraph* graph, const OrtApi& ort_api, const char* name) {
  const OrtValue* initializer = nullptr;

  // Get all initializers in the graph
  size_t num_initializers = 0;
  OrtStatus* status = ort_api.Graph_GetNumInitializers(graph, &num_initializers);
  if (status == nullptr) {
    std::vector<const OrtValueInfo*> initializers(num_initializers);
    if (graph->GetInitializers(initializers).IsOK()) {
      // Find the initializer with the given name
      for (size_t i = 0; i < num_initializers; ++i) {
        const OrtValueInfo* value_info = initializers[i];
        const char* initializer_name = nullptr;
        status = ort_api.GetValueInfoName(value_info, &initializer_name);
        if (status == nullptr && strcmp(initializer_name, name) == 0) {
          // Found the initializer, get its value
          if (value_info->GetInitializerValue(initializer).IsOK()) {
            break;
          }
        }
        if (status != nullptr) {
          ort_api.ReleaseStatus(status);
          status = nullptr;
        }
      }
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
      }
    }
  }

  return initializer;
}

// Helper function to check if a Q or DQ node's scale is a positive constant scalar
bool IsQOrDQScalePositiveConstantScalar(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* q_node) {
  // Get the scale input (index 1) of the Q/DQ node
  size_t num_inputs = 0;
  OrtStatus* status = ort_api.Node_GetNumInputs(q_node, &num_inputs);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }
  if (num_inputs < 2) {
    return false;
  }

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  status = ort_api.Node_GetInputs(q_node, inputs.data(), inputs.size());

  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Get the scale input name
  const OrtValueInfo* scale_value_info = inputs[1];
  const char* scale_name = nullptr;
  // Use the correct API function to get the name of a value info
  status = ort_api.GetValueInfoName(scale_value_info, &scale_name);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Get the scale initializer
  const OrtValue* scale_initializer = GetConstantInitializer(graph, ort_api, scale_name);
  if (scale_initializer == nullptr) {
    return false;
  }

  // Check if the scale is a scalar
  OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  status = ort_api.GetTensorTypeAndShape(scale_initializer, &tensor_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  size_t num_dims = 0;
  status = ort_api.GetDimensionsCount(tensor_info, &num_dims);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }
  if (num_dims != 0) {  // Scalar has 0 dimensions
    return false;
  }

  // Check if the scale is positive
  ONNXTensorElementDataType element_type;
  status = ort_api.GetTensorElementType(tensor_info, &element_type);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);
    return false;
  }

  ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);

  // Check the value based on the data type
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    float* scale_data = nullptr;
    status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(scale_initializer), (void**)&scale_data);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      return false;
    }
    return *scale_data > 0.0f;
  } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    double* scale_data = nullptr;
    status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(scale_initializer), (void**)&scale_data);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      return false;
    }
    return *scale_data > 0.0;
  }

  return false;
}

// Helper function to check if a node group can be created
bool CanCreateNodeGroup(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                        const OrtNode* redundant_clip_node,
                        const std::vector<const OrtNode*>& dq_nodes,
                        const std::vector<const OrtNode*>& q_nodes) {
  // Avoid unused parameter warnings
  ORT_UNUSED_PARAMETER(redundant_clip_node);
  ORT_UNUSED_PARAMETER(graph);

  if (dq_nodes.empty()) {
    return false;
  }

  // Check if the number of DQ inputs matches the number of inputs that exist
  int num_inputs = NumActualValues(node, ort_api, true);
  if (num_inputs < static_cast<int>(dq_nodes.size())) {
    return false;
  }

  // Check if Q nodes are allowed to be empty
  if (q_nodes.empty()) {
    return false;
  }

  // Check if the number of Q outputs matches the number of outputs that exist
  int num_outputs = NumActualValues(node, ort_api, false);
  if (num_outputs < static_cast<int>(q_nodes.size())) {
    return false;
  }

  // Get the outputs as OrtValueInfo instances
  size_t num_outputs_actual = 0;
  OrtStatus* status = ort_api.Node_GetNumOutputs(node, &num_outputs_actual);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  std::vector<const OrtValueInfo*> outputs(num_outputs_actual);
  status = ort_api.Node_GetOutputs(node, outputs.data(), outputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Check if any of the outputs are graph outputs
  bool produces_graph_output = false;
  for (size_t i = 0; i < num_outputs_actual; i++) {
    const OrtValueInfo* value_info = outputs[i];
    bool is_graph_output = false;
    status = ort_api.ValueInfo_IsGraphOutput(value_info, &is_graph_output);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }

    if (is_graph_output) {
      produces_graph_output = true;
      break;
    }
  }

  // Count the total number of consumers for all outputs
  size_t total_consumers = 0;
  for (size_t i = 0; i < num_outputs_actual; i++) {
    const OrtValueInfo* value_info = outputs[i];
    size_t num_consumers = 0;
    status = ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }

    total_consumers += num_consumers;
  }

  return (num_outputs == static_cast<int>(q_nodes.size())) &&
         (q_nodes.size() == total_consumers) &&
         !produces_graph_output;
}

// Helper function to check if a QDQ pair is supported
bool IsQDQPairSupported(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* q_node, const OrtNode* dq_node) {
  // Check if both nodes have the same scale
  size_t q_num_inputs = 0;
  OrtStatus* status = ort_api.Node_GetNumInputs(q_node, &q_num_inputs);
  if (status != nullptr || q_num_inputs < 2) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return false;
  }

  std::vector<const OrtValueInfo*> q_inputs(q_num_inputs);
  status = ort_api.Node_GetInputs(q_node, q_inputs.data(), q_inputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  size_t dq_num_inputs = 0;
  status = ort_api.Node_GetNumInputs(dq_node, &dq_num_inputs);
  if (status != nullptr || dq_num_inputs < 2) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return false;
  }

  std::vector<const OrtValueInfo*> dq_inputs(dq_num_inputs);
  status = ort_api.Node_GetInputs(dq_node, dq_inputs.data(), dq_inputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Get the scale input names
  const OrtValueInfo* q_scale_value_info = q_inputs[1];
  const OrtValueInfo* dq_scale_value_info = dq_inputs[1];

  const char* q_scale_name = nullptr;
  status = ort_api.GetValueInfoName(q_scale_value_info, &q_scale_name);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  const char* dq_scale_name = nullptr;
  status = ort_api.GetValueInfoName(dq_scale_value_info, &dq_scale_name);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Check if the scale names are the same (indicating they're the same initializer)
  bool same_scale = (strcmp(q_scale_name, dq_scale_name) == 0);

  // If the scales are different, check if they have the same value
  if (!same_scale) {
    const OrtValue* q_scale_initializer = GetConstantInitializer(graph, ort_api, q_scale_name);
    const OrtValue* dq_scale_initializer = GetConstantInitializer(graph, ort_api, dq_scale_name);

    if (q_scale_initializer == nullptr || dq_scale_initializer == nullptr) {
      return false;
    }

    // Check if both scales have the same data type and shape
    OrtTensorTypeAndShapeInfo* q_tensor_info = nullptr;
    status = ort_api.GetTensorTypeAndShape(q_scale_initializer, &q_tensor_info);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      return false;
    }

    OrtTensorTypeAndShapeInfo* dq_tensor_info = nullptr;
    status = ort_api.GetTensorTypeAndShape(dq_scale_initializer, &dq_tensor_info);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      return false;
    }

    ONNXTensorElementDataType q_element_type, dq_element_type;
    status = ort_api.GetTensorElementType(q_tensor_info, &q_element_type);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
      return false;
    }

    status = ort_api.GetTensorElementType(dq_tensor_info, &dq_element_type);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
      return false;
    }

    if (q_element_type != dq_element_type) {
      ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
      return false;
    }

    // Compare the scale values
    if (q_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      float* q_scale_data = nullptr;
      float* dq_scale_data = nullptr;
      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(q_scale_initializer), (void**)&q_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(dq_scale_initializer), (void**)&dq_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      same_scale = (*q_scale_data == *dq_scale_data);
    } else if (q_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      double* q_scale_data = nullptr;
      double* dq_scale_data = nullptr;
      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(q_scale_initializer), (void**)&q_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      status = ort_api.GetTensorMutableData(const_cast<OrtValue*>(dq_scale_initializer), (void**)&dq_scale_data);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
        ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
        return false;
      }

      same_scale = (*q_scale_data == *dq_scale_data);
    }

    ort_api.ReleaseTensorTypeAndShapeInfo(q_tensor_info);
    ort_api.ReleaseTensorTypeAndShapeInfo(dq_tensor_info);
  }

  return same_scale;
}

bool OrtNodeGroupSelector::CheckQDQNodes(const OrtGraph* /*graph*/, const OrtApi& ort_api, const OrtNode* node,
                                         const OrtNode* /*redundant_clip_node*/,
                                         const std::vector<const OrtNode*>& dq_nodes,
                                         const std::vector<const OrtNode*>& q_nodes,
                                         int num_dq_inputs,
                                         bool is_empty_q_nodes_allowed) const {
  if (num_dq_inputs == -1) {
    num_dq_inputs = NumActualValues(node, ort_api, true);
  }

  // Check if the number of DQ inputs matches the expected number
  if (num_dq_inputs != static_cast<int>(dq_nodes.size())) {
    return false;
  }

  // Check if Q nodes are allowed to be empty
  if (q_nodes.empty()) {
    return is_empty_q_nodes_allowed;
  }

  // Check if the number of Q outputs matches the number of outputs that exist
  int num_outputs = NumActualValues(node, ort_api, false);

  // Get the outputs as OrtValueInfo instances
  size_t num_outputs_actual = 0;
  OrtStatus* status = ort_api.Node_GetNumOutputs(node, &num_outputs_actual);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  std::vector<const OrtValueInfo*> outputs(num_outputs_actual);
  status = ort_api.Node_GetOutputs(node, outputs.data(), outputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Check if any of the outputs are graph outputs
  bool produces_graph_output = false;

  for (size_t i = 0; i < num_outputs_actual; i++) {
    const OrtValueInfo* value_info = outputs[i];
    bool is_graph_output = false;
    status = ort_api.ValueInfo_IsGraphOutput(value_info, &is_graph_output);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }

    if (is_graph_output) {
      produces_graph_output = true;
      break;
    }
  }

  // Count the total number of consumers for all outputs
  size_t total_consumers = 0;
  for (size_t i = 0; i < num_outputs_actual; i++) {
    const OrtValueInfo* value_info = outputs[i];
    size_t num_consumers = 0;
    status = ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }

    total_consumers += num_consumers;
  }

  return (num_outputs == static_cast<int>(q_nodes.size())) &&
         (q_nodes.size() == total_consumers) &&
         !produces_graph_output;
}

bool OrtDropQDQNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                        const OrtNode* redundant_clip_node,
                                        const std::vector<const OrtNode*>& dq_nodes,
                                        const std::vector<const OrtNode*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  if (dt_input != dt_output) {
    return false;
  }

  if (!allow_16bit_ && Is16BitIntType(dt_input)) {
    return false;
  }

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  const OrtNode* dq_node = dq_nodes.front();
  const OrtNode* q_node = q_nodes.front();

  if (!allow_nonpositive_scale_) {
    // Check if the Q node's scale is a positive constant scalar
    if (!IsQOrDQScalePositiveConstantScalar(graph, ort_api, q_node)) {
      return false;
    }
  }

  // Check if the QDQ pair is supported (same scale)
  return IsQDQPairSupported(graph, ort_api, q_node, dq_node);
}

// Implementation of Check() for OrtDropDQNodeGroupSelector
bool OrtDropDQNodeGroupSelector::Check(const OrtGraph* /*graph*/, const OrtApi& ort_api, const OrtNode* /*node*/,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& /*q_nodes*/) const {
  // For drop DQ operations, we check if the node has exactly one DQ input
  if (redundant_clip_node) {
    return false;
  }

  constexpr int num_dq_inputs = 1;
  if (num_dq_inputs != static_cast<int>(dq_nodes.size())) {
    return false;
  }

  // Check if the DQ input has the expected data type
  const OrtNode* dq_node = dq_nodes.front();
  int32_t dt_input = GetNodeIODataType(dq_node, ort_api, true, 0);

  // Allow 16-bit int types only if explicitly allowed
  if (Is16BitIntType(dt_input)) {
    return false;
  }

  // Allow 4-bit int types only if explicitly allowed
  if (Is4BitIntType(dt_input)) {
    return false;
  }

  return true;
}

// Implementation of Check() for OrtUnaryNodeGroupSelector
bool OrtUnaryNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                      const OrtNode* redundant_clip_node,
                                      const std::vector<const OrtNode*>& dq_nodes,
                                      const std::vector<const OrtNode*>& q_nodes) const {
  // For unary operations, we check if the node has exactly one DQ input and one Q output
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  // Check if the input and output data types match
  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  if (dt_input != dt_output) {
    return false;
  }

  // Allow 16-bit int types only if explicitly allowed
  if (Is16BitIntType(dt_input)) {
    return false;
  }

  // Allow 4-bit int types only if explicitly allowed
  if (Is4BitIntType(dt_input)) {
    return false;
  }

  return true;
}

// Implementation of Check() for OrtBinaryNodeGroupSelector
bool OrtBinaryNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  // For binary operations, we check if the node has exactly two DQ inputs and one Q output
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 2)) {
    return false;
  }

  // Check if the input and output data types match
  int32_t dt_input_1 = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_input_2 = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);
  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  // All input and output types must match
  if (dt_input_1 != dt_input_2 || dt_input_1 != dt_output) {
    return false;
  }

  // Allow 16-bit int types only if explicitly allowed
  if (Is16BitIntType(dt_input_1)) {
    return false;
  }

  // Allow 4-bit int types only if explicitly allowed
  if (Is4BitIntType(dt_input_1)) {
    return false;
  }

  return true;
}

// Implementation of Check() for OrtVariadicNodeGroupSelector
bool OrtVariadicNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                         const OrtNode* redundant_clip_node,
                                         const std::vector<const OrtNode*>& dq_nodes,
                                         const std::vector<const OrtNode*>& q_nodes) const {
  // For variadic operations, we check if the node has at least one DQ input and one Q output
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  // Check if all DQ inputs have the same data type
  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  for (size_t i = 1; i < dq_nodes.size(); ++i) {
    if (dt_input != GetNodeIODataType(dq_nodes[i], ort_api, true, 0)) {
      return false;
    }
  }

  // Check if all Q outputs have the same data type
  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);
  for (size_t i = 1; i < q_nodes.size(); ++i) {
    if (dt_output != GetNodeIODataType(q_nodes[i], ort_api, false, 0)) {
      return false;
    }
  }

  // Check if the input and output data types match
  if (dt_input != dt_output) {
    return false;
  }

  // Allow 16-bit int types only if explicitly allowed
  if (Is16BitIntType(dt_input)) {
    return false;
  }

  // Allow 4-bit int types only if explicitly allowed
  if (Is4BitIntType(dt_input)) {
    return false;
  }

  return true;
}

bool OrtSplitNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                      const OrtNode* redundant_clip_node,
                                      const std::vector<const OrtNode*>& dq_nodes,
                                      const std::vector<const OrtNode*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  const OrtNode* dq_node = dq_nodes.front();
  int32_t dt_input = GetNodeIODataType(dq_node, ort_api, true, 0);

  if (!allow_4bit_ && Is4BitIntType(dt_input)) {
    return false;
  }

  // All Q outputs should have same data type and (optionally) equal quantization parameters as the input.
  for (size_t q_idx = 0; q_idx < q_nodes.size(); q_idx++) {
    const OrtNode* q_node = q_nodes[q_idx];

    int32_t dt_output = GetNodeIODataType(q_node, ort_api, false, 0);
    if (dt_input != dt_output) {
      return false;
    }

    if (req_equal_quant_params_ && !IsQDQPairSupported(graph, ort_api, q_node, dq_node)) {
      return false;
    }
  }

  return true;
}

bool OrtConvNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                     const OrtNode* redundant_clip_node,
                                     const std::vector<const OrtNode*>& dq_nodes,
                                     const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  // Input and output types need to be same
  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_weight = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);
  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  if (dt_input != dt_output) {
    return false;
  }

  if (!allow_4bit_weight_ && Is4BitIntType(dt_weight)) {
    return false;
  }

  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (!int8_allowed_ || dt_weight != dt_input) {
      return false;
    }
  }

  if (dq_nodes.size() == 3) {  // has bias
    int32_t dt_bias = GetNodeIODataType(dq_nodes[2], ort_api, true, 0);
    if (dt_bias != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
      return false;
    }
  }

  // 16-bit int types must be explicitly allowed
  if (!allow_16bit_ && (Is16BitIntType(dt_input) || Is16BitIntType(dt_weight))) {
    return false;
  }

  return true;
}

bool OrtEinsumNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, /*num_dq_inputs=*/-1,
                     /*is_empty_q_nodes_allowed=*/true)) {
    return false;
  }
  size_t num_dq_inputs = dq_nodes.size();
  for (size_t i = 0; i < num_dq_inputs; ++i) {
    int32_t dt_input = GetNodeIODataType(dq_nodes[i], ort_api, true, 0);

    // Check if INT8 is allowed
    if (!allow_int8_ && dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
      return false;
    }

    // Check if 16-bit int types are allowed
    if (!allow_16bit_ && Is16BitIntType(dt_input)) {
      return false;
    }

    // Check if 4-bit int types are allowed
    if (!allow_4bit_ && Is4BitIntType(dt_input)) {
      return false;
    }
  }

  if (!q_nodes.empty()) {
    int32_t dt_input0 = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
    int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

    // Check if input and output data types match
    if (dt_input0 != dt_output) {
      return false;
    }
  }

  return true;
}

bool OrtReciprocalNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                           const OrtNode* redundant_clip_node,
                                           const std::vector<const OrtNode*>& dq_nodes,
                                           const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, /*num_dq_inputs=*/-1,
                     /*is_empty_q_nodes_allowed=*/true)) {
    return false;
  }
  size_t num_dq_inputs = dq_nodes.size();
  for (size_t i = 0; i < num_dq_inputs; ++i) {
    int32_t dt_input = GetNodeIODataType(dq_nodes[i], ort_api, true, 0);
    if (!allow_int8_ && dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
      return false;
    }
    if (!allow_16bit_ && Is16BitIntType(dt_input)) {
      return false;
    }
    if (!allow_4bit_ && Is4BitIntType(dt_input)) {
      return false;
    }
  }
  if (!q_nodes.empty()) {
    int32_t dt_input0 = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
    int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);
    if (dt_input0 != dt_output) {
      return false;
    }
  }
  return true;
}

bool OrtMatMulNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  if (dq_nodes.size() != 2) {
    return false;
  }

  // Get input data types
  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_weight = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);

  // Check if INT8 is allowed
  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (!int8_allowed_ || dt_weight != dt_input) {
      return false;
    }
  }

  // 16-bit int types must be explicitly allowed
  if (!allow_16bit_ && (Is16BitIntType(dt_input) || Is16BitIntType(dt_weight))) {
    return false;
  }

  // 4-bit int types must be explicitly allowed
  if (!allow_4bit_ && (Is4BitIntType(dt_input) || Is4BitIntType(dt_weight))) {
    return false;
  }

  // Potential match for QLinearMatMul or MatMulIntegerToFloat
  bool qlinear = !q_nodes.empty();

  if (qlinear) {
    // QLinearMatMul
    if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
      return false;
    }

    int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);
    return dt_input == dt_output;
  } else {
    // Can be converted to MatMulIntegerToFloat if EP supports that
    return matmulintegertofloat_allowed_;
  }
}

bool OrtDQMatMulNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                         const OrtNode* redundant_clip_node,
                                         const std::vector<const OrtNode*>& dq_nodes,
                                         const std::vector<const OrtNode*>& q_nodes) const {
  if (redundant_clip_node) {
    return false;
  }

  // Should not have any Q nodes
  if (!q_nodes.empty()) {
    return false;
  }

  // MatMul has only 1 DQ input
  if (dq_nodes.size() != 1) {
    return false;
  }

  // Check if DQ node has only one output edge and is not a graph output
  size_t num_dq_outputs = 0;
  OrtStatus* status = ort_api.Node_GetNumOutputs(dq_nodes[0], &num_dq_outputs);
  if (status != nullptr || num_dq_outputs != 1) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return false;
  }

  std::vector<const OrtValueInfo*> dq_outputs(num_dq_outputs);
  status = ort_api.Node_GetOutputs(dq_nodes[0], dq_outputs.data(), dq_outputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  const OrtValueInfo* dq_output_value_info = dq_outputs[0];

  // Check if DQ output is a graph output
  bool is_graph_output = false;
  status = ort_api.ValueInfo_IsGraphOutput(dq_output_value_info, &is_graph_output);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  if (is_graph_output) {
    return false;
  }

  // Check if DQ node has only one consumer
  size_t num_consumers = 0;
  status = ort_api.ValueInfo_GetValueNumConsumers(dq_output_value_info, &num_consumers);
  if (status != nullptr || num_consumers != 1) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return false;
  }

  // Check if DQ is connected to MatMul's second input
  // This requires checking the inputs of the MatMul node
  size_t num_inputs = 0;
  status = ort_api.Node_GetNumInputs(node, &num_inputs);
  if (status != nullptr || num_inputs < 2) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return false;
  }

  std::vector<const OrtValueInfo*> node_inputs(num_inputs);
  status = ort_api.Node_GetInputs(node, node_inputs.data(), node_inputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  const OrtValueInfo* second_input_value_info = node_inputs[1];

  // Get the producer of the second input
  const OrtNode* second_input_producer = nullptr;
  status = ort_api.ValueInfo_GetValueProducer(second_input_value_info, &second_input_producer, nullptr);
  if (status != nullptr || second_input_producer != dq_nodes[0]) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return false;
  }

  // Get DQ node inputs to check weight and scale types
  size_t num_dq_inputs = 0;
  status = ort_api.Node_GetNumInputs(dq_nodes[0], &num_dq_inputs);
  if (status != nullptr || num_dq_inputs < 2) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    return false;
  }

  std::vector<const OrtValueInfo*> dq_inputs(num_dq_inputs);
  status = ort_api.Node_GetInputs(dq_nodes[0], dq_inputs.data(), dq_inputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Get weight and scale data types
  const OrtValueInfo* weight_value_info = dq_inputs[0];
  const OrtValueInfo* scale_value_info = dq_inputs[1];

  int32_t dt_weight = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_scales = GetNodeIODataType(dq_nodes[0], ort_api, true, 1);

  // Check if scales are float or float16
  if (dt_scales != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT &&
      dt_scales != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
    return false;
  }

  // Check if weight is 4-bit integer type
  if (!Is4BitIntType(dt_weight)) {
    return false;
  }

  // Get DQ node attributes to check blockwise quantization parameters
  // Check if axis attribute is 0

  // Check if axis attribute is 0
  OrtNodeAttrHelper attr_helper(ort_api, *dq_nodes[0]);
  int64_t axis_value = attr_helper.Get("axis", int64_t(-1));
  if (axis_value != 0) {
    return false;
  }

  // Check if block_size attribute exists and is valid
  int64_t block_size = attr_helper.Get("block_size", int64_t(0));
  if (block_size == 0) {
    return false;
  }

  // Check if block_size is a power of 2 and >= 16
  if (block_size < 16 || ((block_size - 1) & block_size)) {
    return false;
  }

  // Get names of weight, scale, and zero point
  const char* weight_name = nullptr;
  status = ort_api.GetValueInfoName(weight_value_info, &weight_name);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  const char* scale_name = nullptr;
  status = ort_api.GetValueInfoName(scale_value_info, &scale_name);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  // Check for zero point (optional)
  const OrtValueInfo* zero_point_value_info = nullptr;
  const char* zero_point_name = nullptr;
  if (num_dq_inputs > 2) {
    zero_point_value_info = dq_inputs[2];
    status = ort_api.GetValueInfoName(zero_point_value_info, &zero_point_name);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      return false;
    }
  }

  // Check if weight, scale, and zero point are constants
  const OrtValue* weight_initializer = GetConstantInitializer(graph, ort_api, weight_name);
  const OrtValue* scale_initializer = GetConstantInitializer(graph, ort_api, scale_name);
  const OrtValue* zp_initializer = zero_point_name ? GetConstantInitializer(graph, ort_api, zero_point_name) : nullptr;

  if (!weight_initializer || !scale_initializer) {
    return false;
  }

  if (zero_point_name && !zp_initializer) {
    return false;
  }

  // Check tensor shapes
  OrtTensorTypeAndShapeInfo* weight_tensor_info = nullptr;
  status = ort_api.GetTensorTypeAndShape(weight_initializer, &weight_tensor_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return false;
  }

  OrtTensorTypeAndShapeInfo* scale_tensor_info = nullptr;
  status = ort_api.GetTensorTypeAndShape(scale_initializer, &scale_tensor_info);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
    return false;
  }

  OrtTensorTypeAndShapeInfo* zp_tensor_info = nullptr;
  if (zp_initializer) {
    status = ort_api.GetTensorTypeAndShape(zp_initializer, &zp_tensor_info);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
      return false;
    }
  }

  // Check if tensors have rank 2
  size_t weight_dims_count = 0;
  status = ort_api.GetDimensionsCount(weight_tensor_info, &weight_dims_count);
  if (status != nullptr || weight_dims_count != 2) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
    ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
    if (zp_tensor_info) ort_api.ReleaseTensorTypeAndShapeInfo(zp_tensor_info);
    return false;
  }

  size_t scale_dims_count = 0;
  status = ort_api.GetDimensionsCount(scale_tensor_info, &scale_dims_count);
  if (status != nullptr || scale_dims_count != 2) {
    if (status != nullptr) ort_api.ReleaseStatus(status);
    ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
    ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
    if (zp_tensor_info) ort_api.ReleaseTensorTypeAndShapeInfo(zp_tensor_info);
    return false;
  }

  if (zp_tensor_info) {
    size_t zp_dims_count = 0;
    status = ort_api.GetDimensionsCount(zp_tensor_info, &zp_dims_count);
    if (status != nullptr || zp_dims_count != 2) {
      if (status != nullptr) ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(zp_tensor_info);
      return false;
    }
  }

  // Get dimensions of tensors
  int64_t weight_dims[2];
  status = ort_api.GetDimensions(weight_tensor_info, weight_dims, 2);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
    ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
    if (zp_tensor_info) ort_api.ReleaseTensorTypeAndShapeInfo(zp_tensor_info);
    return false;
  }

  int64_t scale_dims[2];
  status = ort_api.GetDimensions(scale_tensor_info, scale_dims, 2);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
    ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
    if (zp_tensor_info) ort_api.ReleaseTensorTypeAndShapeInfo(zp_tensor_info);
    return false;
  }

  int64_t zp_dims[2] = {0, 0};
  if (zp_tensor_info) {
    status = ort_api.GetDimensions(zp_tensor_info, zp_dims, 2);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
      ort_api.ReleaseTensorTypeAndShapeInfo(zp_tensor_info);
      return false;
    }
  }

  // Check shape compatibility
  bool shapes_compatible = ((weight_dims[0] + block_size - 1) / block_size == scale_dims[0]) &&
                           (weight_dims[1] == scale_dims[1]);

  if (zp_tensor_info) {
    shapes_compatible = shapes_compatible &&
                        (zp_dims[0] == scale_dims[0]) &&
                        (zp_dims[1] == scale_dims[1]);
  }

  // Clean up resources
  ort_api.ReleaseTensorTypeAndShapeInfo(weight_tensor_info);
  ort_api.ReleaseTensorTypeAndShapeInfo(scale_tensor_info);
  if (zp_tensor_info) ort_api.ReleaseTensorTypeAndShapeInfo(zp_tensor_info);

  return shapes_compatible;
}

bool OrtGemmNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                     const OrtNode* redundant_clip_node,
                                     const std::vector<const OrtNode*>& dq_nodes,
                                     const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, -1 /*num_dq_inputs*/,
                     true /*is_empty_q_nodes_allowed*/)) {
    return false;
  }

  // Check if we have at least 2 DQ nodes (A and B inputs)
  if (dq_nodes.size() < 2) {
    return false;
  }

  // Get input data types for A and B
  int32_t dt_A = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_B = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);

  // If A is INT8, B must also be INT8
  if (dt_A == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
    if (dt_A != dt_B) {  // if A is signed int, B must be signed int
      return false;
    }
  }

  // If there are Q nodes, check if activation and output have the same type
  if (!q_nodes.empty()) {
    int32_t dt_Y = GetNodeIODataType(q_nodes[0], ort_api, false, 0);
    if (dt_A != dt_Y) {  // activation and output must be same type
      return false;
    }
  }

  // 16-bit int types must be explicitly allowed
  if (!allow_16bit_ && (Is16BitIntType(dt_A) || Is16BitIntType(dt_B))) {
    return false;
  }

  // 4-bit int types must be explicitly allowed
  if (!allow_4bit_ && (Is4BitIntType(dt_A) || Is4BitIntType(dt_B))) {
    return false;
  }

  // If there's no bias (less than 3 DQ nodes), we're done
  if (dq_nodes.size() < 3) {
    return true;
  }

  // Check if beta attribute is 1.0 (required for bias)
  OrtNodeAttrHelper attr_helper(ort_api, *node);
  float beta_value = attr_helper.Get("beta", 0.0f);

  // Beta needs to be 1.0 for bias
  if (beta_value != 1.0f) {
    return false;
  }

  // Check if bias has the correct data type (INT32)
  int32_t dt_bias = GetNodeIODataType(dq_nodes[2], ort_api, true, 0);
  return dt_bias == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
}

bool OrtWhereNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                      const OrtNode* redundant_clip_node,
                                      const std::vector<const OrtNode*>& dq_nodes,
                                      const std::vector<const OrtNode*>& q_nodes) const {
  // Where has 1 boolean input and 2 dq inputs
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 2)) {
    return false;
  }

  // Check if all DQ inputs have the same data type
  const int32_t dt_input_1 = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  const int32_t dt_input_2 = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);

  // Check if all Q outputs have the same data type
  const int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  // All input and output types must match
  if (dt_input_1 != dt_input_2 || dt_input_1 != dt_output) {
    return false;
  }

  // Allow 16-bit int types only if explicitly allowed
  if (Is16BitIntType(dt_input_1)) {
    return false;
  }

  // Allow 4-bit int types only if explicitly allowed
  if (Is4BitIntType(dt_input_1)) {
    return false;
  }

  return true;
}

bool OrtPadNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                    const OrtNode* redundant_clip_node,
                                    const std::vector<const OrtNode*>& dq_nodes,
                                    const std::vector<const OrtNode*>& q_nodes) const {
  // Pad can have 1 or 2 dq input, the optional input constant_value can be quantized or non-quantized.
  // QNN supports data input quantized with constant_value input non-quantized.
  int num_dq_inputs = static_cast<int>(dq_nodes.size());
  if (num_dq_inputs > 2) {
    return false;
  }

  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, num_dq_inputs)) {
    return false;
  }

  const int32_t dt_input_1 = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  const int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  if (dq_nodes.size() > 1) {
    const int32_t dt_input_2 = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);
    return dt_input_1 == dt_input_2 && dt_input_1 == dt_output;
  } else {
    return dt_input_1 == dt_output;
  }
}

bool OrtInstanceAndLayerNormalizationNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                              const OrtNode* redundant_clip_node,
                                                              const std::vector<const OrtNode*>& dq_nodes,
                                                              const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes)) {
    return false;
  }

  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_bias = 0;
  bool has_bias = false;

  // bias is optional for LayerNorm
  if (dq_nodes.size() > 2) {
    has_bias = true;
    dt_bias = GetNodeIODataType(dq_nodes[2], ort_api, true, 0);
  }

  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  // Input, output, need to be the same type. The bias is int32.
  // Scale can be different with input for a16w8 case
  return (dt_input == dt_output) &&
         (has_bias ? dt_bias == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32 : true);  // 6 is INT32 in ONNX_NAMESPACE::TensorProto_DataType
}

bool OrtBatchNormalizationNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                   const OrtNode* redundant_clip_node,
                                                   const std::vector<const OrtNode*>& dq_nodes,
                                                   const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 3)) {
    return false;
  }

  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_scale = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);
  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  if (dt_input != dt_output) {
    return false;
  }

  // INT8 is 3 in ONNX_NAMESPACE::TensorProto_DataType
  if (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {  // INT8
    if (!int8_allowed_ || dt_scale != dt_input) {
      return false;
    }
  }

  return true;
}

bool OrtLogicalComparisonNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                                  const OrtNode* redundant_clip_node,
                                                  const std::vector<const OrtNode*>& dq_nodes,
                                                  const std::vector<const OrtNode*>& q_nodes) const {
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, -1, true)) {
    return false;
  }

  int32_t dt_input_1 = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_input_2 = GetNodeIODataType(dq_nodes[1], ort_api, true, 0);
  return dt_input_1 == dt_input_2;
}

bool OrtTopKNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                     const OrtNode* redundant_clip_node,
                                     const std::vector<const OrtNode*>& dq_nodes,
                                     const std::vector<const OrtNode*>& q_nodes) const {
  // Not support for now. Need to handle the indices output if we want to support it.
  if (redundant_clip_node) {
    return false;
  }

  constexpr int num_dq_inputs = 1;
  constexpr int num_q_outputs = 1;
  if (num_dq_inputs != gsl::narrow_cast<int>(dq_nodes.size())) {
    return false;
  }

  if (!CanCreateNodeGroup(graph, ort_api, node, nullptr, dq_nodes, q_nodes)) {
    return false;
  }

  if (num_q_outputs != gsl::narrow_cast<int>(q_nodes.size())) {
    return false;
  }

  const OrtNode* dq_node = dq_nodes.front();
  const OrtNode* q_node = q_nodes.front();

  int32_t dt_input = GetNodeIODataType(dq_node, ort_api, true, 0);
  int32_t dt_output = GetNodeIODataType(q_node, ort_api, false, 0);

  if (dt_input != dt_output) {
    return false;
  }

  // Check if the QDQ pair is supported (same scale)
  return IsQDQPairSupported(graph, ort_api, q_node, dq_node);
}

bool OrtCumSumNodeGroupSelector::Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                                       const OrtNode* redundant_clip_node,
                                       const std::vector<const OrtNode*>& dq_nodes,
                                       const std::vector<const OrtNode*>& q_nodes) const {
  // Only the first input has DQ node
  if (!CheckQDQNodes(graph, ort_api, node, redundant_clip_node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  int32_t dt_input = GetNodeIODataType(dq_nodes[0], ort_api, true, 0);
  int32_t dt_output = GetNodeIODataType(q_nodes[0], ort_api, false, 0);

  if (dt_input != dt_output) {
    return false;
  }

  return true;
}

// Helper function to get QDQ selection for a node
std::optional<OrtNodeGroup> GetOrtQDQSelection(const OrtGraph* graph, const OrtApi& ort_api,
                                               const OrtNode* node, const OrtNodeGroupSelector* selector) {
  // Find DQ nodes that feed into this node
  std::vector<const OrtNode*> dq_nodes;

  // Get the inputs as OrtValueInfo instances
  size_t num_inputs = 0;
  OrtStatus* status = ort_api.Node_GetNumInputs(node, &num_inputs);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::nullopt;
  }

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  status = ort_api.Node_GetInputs(node, inputs.data(), inputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::nullopt;
  }

  // For each input, get the producer node
  for (size_t i = 0; i < num_inputs; ++i) {
    const OrtValueInfo* value_info = inputs[i];
    if (value_info == nullptr) {
      continue;
    }

    // Get the producer node
    const OrtNode* producer_node = nullptr;
    status = ort_api.ValueInfo_GetValueProducer(value_info, &producer_node, nullptr);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }

    if (producer_node == nullptr) {
      continue;
    }

    // Check if this is a DQ node
    if (producer_node->GetOpType() == "DequantizeLinear") {
      dq_nodes.push_back(producer_node);
    }
  }

  // For redundant clip node, currently only support node with only one output, which is consumed by Clip/Relu->Q.
  const OrtNode* clip_node = nullptr;

  // Get the outputs to check count
  size_t output_count = 0;
  status = ort_api.Node_GetNumOutputs(node, &output_count);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::nullopt;
  }

  if (output_count == 1) {
    // Get the outputs as OrtValueInfo instances
    std::vector<const OrtValueInfo*> outputs(output_count);
    status = ort_api.Node_GetOutputs(node, outputs.data(), outputs.size());
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      return std::nullopt;
    }

    // For each output, get the consumer nodes
    const OrtValueInfo* value_info = outputs[0];

    // Get the number of consumers
    size_t num_consumers = 0;
    status = ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      return std::nullopt;
    }

    if (num_consumers == 1) {
      // Get the consumer node
      const OrtNode* next_node = nullptr;
      int64_t input_index = 0;  // This value is not used, but necessary for the API call
      status = ort_api.ValueInfo_GetValueConsumers(value_info, &next_node, &input_index, 1);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        return std::nullopt;
      }

      // Check if it's a Relu or Clip node
      if (next_node->GetOpType() == "Relu" || next_node->GetOpType() == "Clip") {
        // Get the outputs of the next node to check count
        size_t next_output_count = 0;
        status = ort_api.Node_GetNumOutputs(next_node, &next_output_count);
        if (status != nullptr) {
          ort_api.ReleaseStatus(status);
          return std::nullopt;
        }

        if (next_output_count == 1) {
          // Get the outputs of the next node
          std::vector<const OrtValueInfo*> next_outputs(next_output_count);
          status = ort_api.Node_GetOutputs(next_node, next_outputs.data(), next_outputs.size());
          if (status != nullptr) {
            ort_api.ReleaseStatus(status);
            return std::nullopt;
          }

          // Check if any of the outputs are graph outputs
          bool produces_graph_output = false;
          for (size_t i = 0; i < next_output_count; i++) {
            const OrtValueInfo* next_value_info = next_outputs[i];
            bool is_graph_output = false;
            status = ort_api.ValueInfo_IsGraphOutput(next_value_info, &is_graph_output);
            if (status != nullptr) {
              ort_api.ReleaseStatus(status);
              continue;
            }

            if (is_graph_output) {
              produces_graph_output = true;
              break;
            }
          }

          // Get the number of consumers of the next node's output
          size_t next_num_consumers = 0;
          if (next_output_count > 0) {
            const OrtValueInfo* next_value_info = next_outputs[0];
            status = ort_api.ValueInfo_GetValueNumConsumers(next_value_info, &next_num_consumers);
            if (status != nullptr) {
              ort_api.ReleaseStatus(status);
              return std::nullopt;
            }
          }

          if (next_num_consumers == 1 && !produces_graph_output) {
            clip_node = next_node;
          }
        }
      }
    }
  }

  // Find Q nodes that consume from this node or the clip node
  std::vector<const OrtNode*> q_nodes;

  // Get the outputs as OrtValueInfo instances
  size_t num_outputs = 0;
  status = ort_api.Node_GetNumOutputs(clip_node ? clip_node : node, &num_outputs);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::nullopt;
  }

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  status = ort_api.Node_GetOutputs(clip_node ? clip_node : node, outputs.data(), outputs.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::nullopt;
  }

  // For each output, get the consumer nodes
  for (size_t i = 0; i < num_outputs; ++i) {
    const OrtValueInfo* value_info = outputs[i];
    if (value_info == nullptr) {
      continue;
    }

    // Get the number of consumers
    size_t num_consumers = 0;
    status = ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }

    if (num_consumers > 0) {
      // Allocate arrays for consumer nodes and input indices
      std::vector<const OrtNode*> consumer_nodes_vec(num_consumers);
      std::vector<int64_t> input_indices_vec(num_consumers);

      // Get the consumer nodes
      status = ort_api.ValueInfo_GetValueConsumers(value_info, consumer_nodes_vec.data(), input_indices_vec.data(), num_consumers);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        continue;
      }

      // Check each consumer node
      for (size_t j = 0; j < num_consumers; ++j) {
        const OrtNode* consumer_node = consumer_nodes_vec[j];

        // Check if this is a Q node
        if (consumer_node->GetOpType() == "QuantizeLinear") {
          q_nodes.push_back(consumer_node);
        }
      }
    }
  }

  // Check if the node group is supported by the selector
  if (selector->Check(graph, ort_api, node, clip_node, dq_nodes, q_nodes)) {
    // Create a NodeGroup
    OrtNodeGroup node_group;

    node_group.target_node = node;

    if (clip_node) {
      node_group.redundant_clip_node = clip_node;
    }

    // Add DQ node indices
    node_group.dq_nodes.reserve(dq_nodes.size());
    for (const OrtNode* dq_node : dq_nodes) {
      node_group.dq_nodes.push_back(dq_node);
    }

    // Add Q node indices
    node_group.q_nodes.reserve(q_nodes.size());
    for (const OrtNode* q_node : q_nodes) {
      node_group.q_nodes.push_back(q_node);
    }

    return node_group;
  }

  return std::nullopt;
}

// Implementation of OrtSelectorManager constructor and related functions
OrtSelectorManager::OrtSelectorManager() {
  CreateSelectors();
  InitializeSelectorsMap();
}

void OrtSelectorManager::CreateSelectors() {
  // Register selectors for different op types

  // Register misc ops
  OrtOpVersionsAndSelector::OpVersionsMap misc_ops = {
      {"Gather", {}},
      {"GatherElements", {}},
      {"Reshape", {}},
      {"Expand", {}},
      {"Flatten", {}},
      {"Transpose", {}},
      {"MaxPool", {12}},
      {"Resize", {}},
      {"Squeeze", {}},
      {"Unsqueeze", {}},
      {"Tile", {}}};
  ort_selectors_.RegisterSelector(misc_ops, std::make_unique<OrtDropQDQNodeGroupSelector>());

  // Register drop DQ ops
  OrtOpVersionsAndSelector::OpVersionsMap drop_dq_ops = {
      {"ArgMax", {}},
      {"ArgMin", {}}};
  ort_selectors_.RegisterSelector(drop_dq_ops, std::make_unique<OrtDropDQNodeGroupSelector>());

  // Register unary ops
  OrtOpVersionsAndSelector::OpVersionsMap unary_ops = {
      {"AveragePool", {}},
      {"GlobalAveragePool", {}},
      {"GlobalMaxPool", {}},
      {"LeakyRelu", {}},
      {"ReduceMean", {}},
      {"ReduceMin", {}},
      {"ReduceMax", {}},
      {"ReduceProd", {}},
      {"ReduceSum", {}},
      {"Relu", {}},
      {"Gelu", {}},
      {"Elu", {}},
      {"HardSigmoid", {}},
      {"HardSwish", {}},
      {"Sigmoid", {}},
      {"Slice", {}},
      {"LogSoftmax", {}},
      {"Softmax", {}},
      {"Sqrt", {}},
      {"Atan", {}},
      {"Asin", {}},
      {"Sin", {}},
      {"Cos", {}},
      {"Sign", {}},
      {"Tanh", {}},
      {"Exp", {}},
      {"Log", {}},
      {"LRN", {}},
      {"Ceil", {}},
      {"Floor", {}},
      {"Round", {}},
      {"Abs", {}},
      {"Neg", {}},
      {"DepthToSpace", {}},
      {"SpaceToDepth", {}},
      {"Clip", {}},
      {"LpNormalization", {}}};
  ort_selectors_.RegisterSelector(unary_ops, std::make_unique<OrtUnaryNodeGroupSelector>());

  // Register binary ops
  OrtOpVersionsAndSelector::OpVersionsMap binary_ops = {
      {"Add", {}},
      {"Div", {}},
      {"Mul", {}},
      {"Pow", {}},
      {"Sub", {}},
      {"PRelu", {}},
      {"GridSample", {}}};
  ort_selectors_.RegisterSelector(binary_ops, std::make_unique<OrtBinaryNodeGroupSelector>());

  // Register variadic ops
  OrtOpVersionsAndSelector::OpVersionsMap variadic_ops = {
      {"Concat", {}},
      {"Max", {}},
      {"Min", {}}};
  ort_selectors_.RegisterSelector(variadic_ops, std::make_unique<OrtVariadicNodeGroupSelector>());

  // Register split ops
  OrtOpVersionsAndSelector::OpVersionsMap split_ops = {
      {"Split", {}}};
  ort_selectors_.RegisterSelector(split_ops, std::make_unique<OrtSplitNodeGroupSelector>());

  // Register conv ops
  OrtOpVersionsAndSelector::OpVersionsMap conv_ops = {
      {"Conv", {}}};
  ort_selectors_.RegisterSelector(conv_ops, std::make_unique<OrtConvNodeGroupSelector>());

  // Register conv transpose ops
  OrtOpVersionsAndSelector::OpVersionsMap conv_transpose_ops = {
      {"ConvTranspose", {}}};
  ort_selectors_.RegisterSelector(conv_transpose_ops, std::make_unique<OrtConvNodeGroupSelector>());

  // Register einsum ops
  OrtOpVersionsAndSelector::OpVersionsMap einsum_ops = {
      {"Einsum", {}}};
  ort_selectors_.RegisterSelector(einsum_ops, std::make_unique<OrtEinsumNodeGroupSelector>());

  // Register reciprocal ops
  OrtOpVersionsAndSelector::OpVersionsMap reciprocal_ops = {
      {"Reciprocal", {}}};
  ort_selectors_.RegisterSelector(reciprocal_ops, std::make_unique<OrtReciprocalNodeGroupSelector>());

  // Register matmul ops
  OrtOpVersionsAndSelector::OpVersionsMap matmul_ops = {
      {"MatMul", {}}};
  ort_selectors_.RegisterSelector(matmul_ops, std::make_unique<OrtMatMulNodeGroupSelector>());

  // Register gemm ops
  OrtOpVersionsAndSelector::OpVersionsMap gemm_ops = {
      {"Gemm", {}}};
  ort_selectors_.RegisterSelector(gemm_ops, std::make_unique<OrtGemmNodeGroupSelector>());

  // Register instance and layer normalization ops
  OrtOpVersionsAndSelector::OpVersionsMap instance_layer_norm_ops = {
      {"InstanceNormalization", {}},
      {"LayerNormalization", {}}};
  ort_selectors_.RegisterSelector(instance_layer_norm_ops, std::make_unique<OrtInstanceAndLayerNormalizationNodeGroupSelector>());

  // Register batch normalization ops
  OrtOpVersionsAndSelector::OpVersionsMap batch_norm_ops = {
      {"BatchNormalization", {}}};
  ort_selectors_.RegisterSelector(batch_norm_ops, std::make_unique<OrtBatchNormalizationNodeGroupSelector>());

  // Register logical comparison ops
  OrtOpVersionsAndSelector::OpVersionsMap logical_comparison_ops = {
      {"Equal", {}},
      {"Greater", {}},
      {"GreaterOrEqual", {}},
      {"Less", {}},
      {"LessOrEqual", {}}};
  ort_selectors_.RegisterSelector(logical_comparison_ops, std::make_unique<OrtLogicalComparisonNodeGroupSelector>());

  // Register where ops
  OrtOpVersionsAndSelector::OpVersionsMap where_ops = {
      {"Where", {}}};
  ort_selectors_.RegisterSelector(where_ops, std::make_unique<OrtWhereNodeGroupSelector>());

  // Register pad ops
  OrtOpVersionsAndSelector::OpVersionsMap pad_ops = {
      {"Pad", {}}};
  ort_selectors_.RegisterSelector(pad_ops, std::make_unique<OrtPadNodeGroupSelector>());

  // Register topk ops
  OrtOpVersionsAndSelector::OpVersionsMap topk_ops = {
      {"TopK", {}}};
  ort_selectors_.RegisterSelector(topk_ops, std::make_unique<OrtTopKNodeGroupSelector>());

  // Register cumsum ops
  OrtOpVersionsAndSelector::OpVersionsMap cumsum_ops = {
      {"CumSum", {}}};
  ort_selectors_.RegisterSelector(cumsum_ops, std::make_unique<OrtCumSumNodeGroupSelector>());
}

void OrtSelectorManager::InitializeSelectorsMap() {
  for (const auto& entry : ort_selectors_.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      op_type_to_selectors_map_.insert({op_info.first, &*entry});
    }
  }
}

// Implementation of GetQDQSelections for OrtGraph
std::vector<OrtNodeGroup> OrtSelectorManager::GetOrtQDQSelections(const OrtGraph* graph, const OrtApi& ort_api, const logging::Logger& logger) const {
  std::vector<OrtNodeGroup> qdq_selections;

  // Get all nodes from the graph
  size_t num_nodes = 0;
  auto status = ort_api.Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return qdq_selections;
  }
  std::vector<const OrtNode*> nodes(num_nodes);
  status = ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return qdq_selections;
  }

  // Process each node
  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = nodes[i];

    // Get node op type
    const char* op_type = node->GetOpType().c_str();

    // Get node domain
    const char* domain = nullptr;
    status = ort_api.Node_GetDomain(node, &domain);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }

    // Check domain (similar to the GraphViewer version)
    std::string domain_str(domain);
    if (domain_str != kOnnxDomain && domain_str != kMSInternalNHWCDomain && domain_str != kMSDomain) {
      continue;
    }

    // Find selector for this op type
    auto op_rule = op_type_to_selectors_map_.find(op_type);
    if (op_rule == op_type_to_selectors_map_.cend()) {
      continue;
    }

    const auto& op_versions_and_selector = *op_rule->second;

    // Check the supported versions if specified
    const auto& versions = op_versions_and_selector.op_versions_map.find(op_type)->second;
    if (!versions.empty()) {
      // Get node version
      int since_version = 0;
      status = ort_api.Node_GetSinceVersion(node, &since_version);
      if (status != nullptr) {
        ort_api.ReleaseStatus(status);
        continue;
      }

      if (std::find(versions.cbegin(), versions.cend(), since_version) == versions.cend()) {
        LOGS(logger, VERBOSE) << "Op version is not supported for " << op_type;
        continue;
      }
    }

    // Get QDQ selection for this node
    const auto qdq_node_group_selection = GetOrtQDQSelection(graph, ort_api, node, op_versions_and_selector.selector.get());
    if (qdq_node_group_selection.has_value()) {
      const auto& qdq_group = *qdq_node_group_selection;
      qdq_selections.push_back(qdq_group);
    }
  }
  return qdq_selections;
}

}  // namespace QDQ

namespace utils {

std::vector<std::vector<const OrtNode*>> CreateSupportedPartitionNodeGroups(
    const OrtGraph* graph,
    const OrtApi& ort_api,
    const std::vector<const OrtNode*>& supported_nodes,
    const std::string& ep_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map) {
  std::vector<std::vector<const OrtNode*>> supported_groups{};

  size_t num_nodes = 0;
  auto status = ort_api.Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return {};
  }
  std::vector<const OrtNode*> graph_nodes(num_nodes);
  status = ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return {};
  }

  // #inputs from unprocessed nodes (in-degree) per node.
  std::unordered_map<size_t, size_t> in_degree{};
  // Nodes that are ready to process.
  std::deque<const OrtNode*> nodes_to_process{};
  // Nodes that will be processed when considering the next partition node group.
  std::deque<const OrtNode*> nodes_to_process_with_next_group{};

  // Initialize in-degrees and find root nodes.
  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = graph_nodes[node_idx];
    const OrtNodeUnit* node_unit = node_unit_map.at(node);

    if (&node_unit->GetNode() != node) {
      // Only process the target node.
      continue;
    }

    size_t degree = node_unit->GetInputEdgesCount(ort_api);
    in_degree.insert({node_unit->Index(), degree});
    if (degree == 0) {
      nodes_to_process.push_back(node);
    }
  }

  std::vector<const OrtNode*> supported_group{};
  // The partition node group's border is the aggregate of its nodes' output nodes.
  InlinedHashSet<const OrtNode*> supported_group_border{};

  auto close_group = [&]() {
    if (!supported_group.empty()) {
      supported_groups.emplace_back(std::move(supported_group));
      supported_group.clear();
      supported_group_border.clear();
    }
  };

  size_t num_nodes_processed = 0;

  while (!nodes_to_process.empty() || !nodes_to_process_with_next_group.empty()) {
    if (nodes_to_process.empty()) {
      // We have processed all the nodes that we can while building this partition node group, start a new one.
      close_group();
      nodes_to_process.swap(nodes_to_process_with_next_group);
      continue;
    }

    const OrtNode* node = nodes_to_process.front();
    nodes_to_process.pop_front();

    const OrtNodeUnit* node_unit = node_unit_map.at(node);
    const bool is_qdq_node_unit = node_unit->UnitType() == OrtNodeUnit::Type::QDQGroup;

    // A node that is already assigned to an EP other than current EP is unsupported.
    const char* node_ep_name;
    status = ort_api.Node_GetEpName(node, &node_ep_name);
    if (status != nullptr) {
      ort_api.ReleaseStatus(status);
      continue;
    }
    const bool is_node_supported = (
        (std::string(node_ep_name).empty() || node_ep_name == ep_type) &&
        std::find(supported_nodes.cbegin(), supported_nodes.cend(), node) != supported_nodes.cend());

    if (!is_node_supported && Contains(supported_group_border, node)) {
      // An unsupported node on the border will be processed after the current partition node group.
      nodes_to_process_with_next_group.push_back(node);
      continue;
    }

    if (is_node_supported) {
      if (is_qdq_node_unit) {
        // Add DQ -> node -> Q for the node unit and must be in topological order.
        for (const OrtNode* dq : node_unit->GetDQNodes()) {
          supported_group.push_back(dq);
        }

        supported_group.push_back(node);
        const OrtNode* redundent_clip_node = node_unit->GetRedundantClipNode();
        if (redundent_clip_node) {
          supported_group.push_back(redundent_clip_node);
          supported_group_border.erase(redundent_clip_node);
        }

        for (const OrtNode* q : node_unit->GetQNodes()) {
          supported_group.push_back(q);
        }
      } else {
        supported_group.push_back(node);
      }

      // Remove node from the border.
      supported_group_border.erase(node);
    }

    // For each downstream node:
    //   1: Add the downstream node to the border if the current node is supported.
    //   2: Adjust in-degrees of the nodes consuming the current node's outputs, and add any new nodes to process.
    for (const OrtNode* output_node : node_unit->GetOutputNodes(ort_api)) {
      const OrtNodeUnit* downstream_node_unit = node_unit_map.at(output_node);
      const OrtNode* downstream_node = &downstream_node_unit->GetNode();

      if (is_node_supported) {
        supported_group_border.insert(downstream_node);
      }

      auto& downstream_node_in_degree = in_degree[downstream_node_unit->Index()];
      --downstream_node_in_degree;

      if (downstream_node_in_degree == 0) {
        nodes_to_process.push_back(downstream_node);
      }
    }

    ++num_nodes_processed;
  }

  close_group();

  ORT_ENFORCE(num_nodes_processed == in_degree.size(),
              "Processed ", num_nodes_processed, " nodes. Expected to process ", in_degree.size());

  return supported_groups;
}

}  // namespace utils

class QnnEp;

// Implementation of GetQDQNodeUnits for OrtGraph
std::pair<std::vector<std::unique_ptr<OrtNodeUnit>>, std::unordered_map<const OrtNode*, const OrtNodeUnit*>>
GetAllOrtNodeUnits(OrtApi ort_api, const OrtGraph* graph, const logging::Logger& logger) {
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;

  // Get all nodes from the graph
  size_t num_nodes = 0;
  auto status = ort_api.Graph_GetNumNodes(graph, &num_nodes);
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
  }
  std::vector<const OrtNode*> nodes(num_nodes);
  status = ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size());
  if (status != nullptr) {
    ort_api.ReleaseStatus(status);
    return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
  }

  const auto add_node_unit_to_map = [&](const std::vector<const OrtNode*>& _nodes, const OrtNodeUnit* node_unit) {
    for (const OrtNode* node : _nodes) {
      node_unit_map[node] = node_unit;
    }
  };

  // Get QDQ NodeUnits first
  QDQ::OrtSelectorManager selector_mgr;

  const auto qdq_selections = selector_mgr.GetOrtQDQSelections(graph, ort_api, logger);
  for (const auto& qdq_selection : qdq_selections) {
    auto qdq_unit = std::make_unique<OrtNodeUnit>(graph, qdq_selection, ort_api);

    // Fill the node to node_unit map for all nodes in the QDQ Group
    add_node_unit_to_map(qdq_selection.dq_nodes, qdq_unit.get());
    add_node_unit_to_map(qdq_selection.q_nodes, qdq_unit.get());
    add_node_unit_to_map({qdq_selection.target_node}, qdq_unit.get());
    if (qdq_selection.redundant_clip_node) {
      add_node_unit_to_map({qdq_selection.redundant_clip_node}, qdq_unit.get());
    }

    node_unit_holder.push_back(std::move(qdq_unit));
  }

  // Get the left over single-node OrtNodeUnit.
  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = nodes[node_idx];

    // This is already part of a QDQ OrtNodeUnit.
    if (node_unit_map.find(node) != node_unit_map.cend())
      continue;

    auto node_unit = std::make_unique<OrtNodeUnit>(node, ort_api);
    node_unit_map[node] = node_unit.get();
    node_unit_holder.push_back(std::move(node_unit));
  }

  return std::make_pair(std::move(node_unit_holder), std::move(node_unit_map));
}

}  // namespace onnxruntime
