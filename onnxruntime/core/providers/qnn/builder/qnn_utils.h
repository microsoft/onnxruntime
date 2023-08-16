// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "QnnTypes.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "nlohmann/json.hpp"

#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <unordered_set>

namespace onnxruntime {
namespace qnn {
class QnnOpConfigWrapper;

namespace utils {
size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type);

// TODO: make these work with Wrappers?
std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param);
std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor);
std::ostream& operator<<(std::ostream& out, const QnnOpConfigWrapper& op_conf_wrapper);

// Class that allows building a JSON representation of a QNN graph.
// The JSON graph is built in a format that can be loaded with Qualcomm's QNN Netron visualizer.
class QnnJSONGraph {
 public:
  QnnJSONGraph();

  /**
   * Add QNN operator to JSON graph.
   *
   * /param op_conf_wrapper QNN operator to add.
   */
  void AddOp(const QnnOpConfigWrapper& op_conf_wrapper);

  /**
   * Finalizes JSON graph (i.e., adds top-level graph metadata) and returns a reference
   * to the JSON object.
   *
   * /return A const reference to the finalized JSON graph object.
   */
  const nlohmann::json& Finalize();

 private:
  void AddOpTensors(gsl::span<const Qnn_Tensor_t> tensors);

  nlohmann::json json_;
  std::unordered_set<std::string> seen_tensors_;   // Tracks tensors already added to JSON graph.
  std::unordered_set<std::string> seen_op_types_;  // Tracks unique operator types.
};

Status GetQnnDataType(const bool is_quantized_node, const ONNX_NAMESPACE::TypeProto* type_proto,
                      Qnn_DataType_t& tensor_data_type);

bool OnnxDataTypeToQnnDataType(const int32_t data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized = false);

inline void InitializeQuantizeParam(Qnn_QuantizeParams_t& quantize_param, bool is_quantized_model, float scale = 0.0f, int32_t offset = 0) {
  quantize_param.encodingDefinition = is_quantized_model ? QNN_DEFINITION_DEFINED : QNN_DEFINITION_UNDEFINED;
  quantize_param.quantizationEncoding = is_quantized_model ? QNN_QUANTIZATION_ENCODING_SCALE_OFFSET : QNN_QUANTIZATION_ENCODING_UNDEFINED;
  quantize_param.scaleOffsetEncoding.scale = scale;
  quantize_param.scaleOffsetEncoding.offset = offset;
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
