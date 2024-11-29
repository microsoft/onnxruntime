// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_json_graph.h"

#include "core/framework/data_types.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Returns a JSON array from a gsl::span.
template <typename T>
inline nlohmann::json JSONFromSpan(gsl::span<const T> elems) {
  nlohmann::json json_array = nlohmann::json::array();

  for (auto elem : elems) {
    json_array.push_back(elem);
  }

  return json_array;
}

// Fills json array with elements from the raw source buffer.
// Returns the number of bytes copied from the raw source buffer.
template <typename T>
inline uint32_t FillJSONArrayFromRawData(nlohmann::json* json_array, const void* ptr, uint32_t num_elems) {
  gsl::span<const T> elems{reinterpret_cast<const T*>(ptr), static_cast<size_t>(num_elems)};
  for (auto elem : elems) {
    json_array->push_back(elem);
  }

  return num_elems * sizeof(T);
}

template <>
inline uint32_t FillJSONArrayFromRawData<MLFloat16>(nlohmann::json* json_array, const void* ptr, uint32_t num_elems) {
  gsl::span<const MLFloat16> elems{reinterpret_cast<const MLFloat16*>(ptr), static_cast<size_t>(num_elems)};
  for (auto elem : elems) {
    json_array->push_back(elem.ToFloat());
  }

  return num_elems * sizeof(MLFloat16);
}

// Fills json array with typed elements from the raw source buffer.
// Returns the number of bytes copied from the raw source buffer.
uint32_t AppendQnnElemsToJSONArray(nlohmann::json* json_array, const void* data, uint32_t num_elems,
                                   Qnn_DataType_t data_type) {
  switch (data_type) {
    case QNN_DATATYPE_BOOL_8:  // Handle bool the same as int8 (0 or 1)
    case QNN_DATATYPE_INT_8:
      return FillJSONArrayFromRawData<int8_t>(json_array, data, num_elems);
    case QNN_DATATYPE_INT_16:
      return FillJSONArrayFromRawData<int16_t>(json_array, data, num_elems);
    case QNN_DATATYPE_INT_32:
      return FillJSONArrayFromRawData<int32_t>(json_array, data, num_elems);
    case QNN_DATATYPE_INT_64:
      return FillJSONArrayFromRawData<int64_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_8:
      return FillJSONArrayFromRawData<uint8_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_16:
      return FillJSONArrayFromRawData<uint16_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_32:
      return FillJSONArrayFromRawData<uint32_t>(json_array, data, num_elems);
    case QNN_DATATYPE_UINT_64:
      return FillJSONArrayFromRawData<uint64_t>(json_array, data, num_elems);
    case QNN_DATATYPE_FLOAT_32:
      return FillJSONArrayFromRawData<float>(json_array, data, num_elems);
    case QNN_DATATYPE_FLOAT_16:
      return FillJSONArrayFromRawData<MLFloat16>(json_array, data, num_elems);
    default:
      return 0;  // Do not append anything for unsupported types.
  }
}

// Returns a JSON array that contains static tensor data. The resulting JSON array is constructed hierarchically
// according to the provided dimensions/shape.
//
// Example:
// If buf = [0, 1, 2, 3, 4, 5] and dims = [1, 2, 3]
//   => returns JSON array [[[0, 1, 2], [3, 4, 5]]]
nlohmann::json GetQnnClientBufJSON(const Qnn_ClientBuffer_t& buf, Qnn_DataType_t data_type,
                                   gsl::span<const uint32_t> dims) {
  using json = nlohmann::json;
  const char* data_ptr = reinterpret_cast<const char*>(buf.data);

  // Calculate number of elements.
  uint32_t num_elems = 1;
  for (auto d : dims) {
    num_elems *= d;
  }

  if (num_elems == 0) {
    return json::array();
  }

  const uint32_t last_dim = dims.back();
  const uint32_t num_dims = gsl::narrow_cast<uint32_t>(dims.size());
  std::vector<json> curr;
  curr.reserve(num_elems / last_dim);

  // Group raw data into individual JSON arrays of size `last_dim` each.
  // Store these JSON arrays in the `curr` vector.
  for (uint32_t j = num_elems; j > 0; j -= last_dim) {
    curr.push_back(json::array());
    data_ptr += AppendQnnElemsToJSONArray(&curr.back(), data_ptr, last_dim, data_type);
  }

  // Iterate through dimension values backwards (starting at second-to-last).
  // In each iteration, we collect the JSON arrays in the `curr` vector into groups (i.e., new JSON arrays) of
  // size `dim_val`. This new/smaller collection of JSON arrays becomes the input for the next iteration.
  for (uint32_t i = num_dims - 1; i-- > 0;) {
    const uint32_t dim_val = dims[i];
    std::vector<json> next;

    for (uint32_t j = 0; j < curr.size(); ++j) {
      if (j % dim_val == 0) {
        next.push_back(json::array());
      }

      next.back().emplace_back(std::move(curr[j]));
    }

    curr = std::move(next);
  }

  assert(curr.size() == 1);
  return curr[0];
}

// Returns a JSON representation of a QNN tensor.
// Example:
//
// {
//     "id" : 1652639423,
//     "type" : 3
//     "dataFormat" : 0,
//     "data_type" : 562,
//     "dims" : [ 1, 224, 224, 3 ],
//     "quant_params" : { ... },
//     "axis_format" : "NOT_YET_DEFINED",
//     "src_axis_format" : "NOT_YET_DEFINED",
// }
nlohmann::json GetQnnTensorJSON(const Qnn_Tensor_t& tensor, bool include_static_data = false) {
  using json = nlohmann::json;
  json tensor_json = json::object();
  const Qnn_TensorType_t tensor_type = GetQnnTensorType(tensor);

  tensor_json["id"] = GetQnnTensorID(tensor);
  tensor_json["type"] = tensor_type;
  tensor_json["dataFormat"] = GetQnnTensorDataFormat(tensor);
  tensor_json["data_type"] = GetQnnTensorDataType(tensor);
  tensor_json["src_axis_format"] = "NOT_YET_DEFINED";
  tensor_json["axis_format"] = "NOT_YET_DEFINED";

  const Qnn_QuantizeParams_t& quant_params = GetQnnTensorQParams(tensor);
  tensor_json["quant_params"] = {
      {"definition", quant_params.encodingDefinition},
      {"encoding", quant_params.quantizationEncoding},
      {"scale_offset",
       {{"scale", quant_params.scaleOffsetEncoding.scale}, {"offset", quant_params.scaleOffsetEncoding.offset}}}};

  gsl::span<const uint32_t> dims{GetQnnTensorDims(tensor), GetQnnTensorRank(tensor)};
  tensor_json["dims"] = JSONFromSpan(dims);

  if (tensor_type == Qnn_TensorType_t::QNN_TENSOR_TYPE_STATIC) {
    if (include_static_data) {
      tensor_json["data"] = GetQnnClientBufJSON(GetQnnTensorClientBuf(tensor), GetQnnTensorDataType(tensor), dims);
    } else {
      uint32_t* qnn_tensor_dims = GetQnnTensorDims(tensor);
      uint32_t qnn_tensor_rank = GetQnnTensorRank(tensor);
      uint32_t element_counts =
          std::accumulate(qnn_tensor_dims, qnn_tensor_dims + qnn_tensor_rank, 1, std::multiplies<uint32_t>());
      std::stringstream ss;
      ss << element_counts;
      tensor_json["params_count"] = ss.str();
    }
  }

  return tensor_json;
}

// Returns a JSON object representation of a QNN scalar parameter. Example: { "306": 1 }
// Note that the key is the stringified data type.
nlohmann::json GetQnnScalarParamJSON(const Qnn_Scalar_t& param) {
  nlohmann::json param_json = nlohmann::json::object();
  std::stringstream ss;
  ss << static_cast<uint64_t>(param.dataType);

  switch (param.dataType) {
    case QNN_DATATYPE_BOOL_8:  // Print bool the same as int8 (0 or 1)
    case QNN_DATATYPE_INT_8:
      param_json[ss.str()] = param.int8Value;
      break;
    case QNN_DATATYPE_INT_16:
      param_json[ss.str()] = param.int16Value;
      break;
    case QNN_DATATYPE_INT_32:
      param_json[ss.str()] = param.int32Value;
      break;
    case QNN_DATATYPE_UINT_8:
      param_json[ss.str()] = param.uint8Value;
      break;
    case QNN_DATATYPE_UINT_16:
      param_json[ss.str()] = param.uint16Value;
      break;
    case QNN_DATATYPE_UINT_32:
      param_json[ss.str()] = param.uint32Value;
      break;
    case QNN_DATATYPE_FLOAT_32:
      param_json[ss.str()] = param.floatValue;
      break;
    default:
      // Do nothing for unsupported types.
      break;
  }

  return param_json;
}

// Returns a JSON array initialized with the names of the provided QNN tensors.
nlohmann::json GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t> tensors) {
  nlohmann::json names_json = nlohmann::json::array();

  for (const auto& tensor : tensors) {
    names_json.push_back(GetQnnTensorName(tensor));
  }

  return names_json;
}

// Returns a JSON representation of a QNN operator.
// Example:
// {
//     "package": "qti.aisw",
//     "type": "Conv2d",
//     "input_names": [ "Transpose_token_2012_out0", "weight_quantized", "beta_quantized" ],
//     "output_names": [ "resnetv17_relu0_fwd_QuantizeLinear" ],
//     "scalar_params": { "group": {...} },
//     "tensor_params": { "stride": {...} },
//     "macs_per_inference": ""
// }
nlohmann::json GetQnnOpJSON(const QnnOpConfigWrapper& op_config) {
  using json = nlohmann::json;
  json op_json = json::object();
  op_json["package"] = op_config.GetPackageName();
  op_json["type"] = op_config.GetTypeName();

  json tensor_params_json = json::object();
  json scalar_params_json = json::object();

  gsl::span<const Qnn_Param_t> params{op_config.GetParams(), op_config.GetParamsNum()};
  for (const auto& param : params) {
    if (param.paramType == QNN_PARAMTYPE_SCALAR) {
      scalar_params_json[param.name] = GetQnnScalarParamJSON(param.scalarParam);
    } else if (param.paramType == QNN_PARAMTYPE_TENSOR) {
      tensor_params_json[param.name][GetQnnTensorName(param.tensorParam)] = GetQnnTensorJSON(param.tensorParam, true);
    }
  }

  op_json["tensor_params"] = std::move(tensor_params_json);
  op_json["scalar_params"] = std::move(scalar_params_json);
  op_json["input_names"] =
      GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t>{op_config.GetInputTensors(), op_config.GetInputsNum()});
  op_json["output_names"] =
      GetQnnTensorNamesJSON(gsl::span<const Qnn_Tensor_t>{op_config.GetOutputTensors(), op_config.GetOutputsNum()});
  op_json["macs_per_inference"] = "";  // Metadata set by QNN converter tools. Not needed.

  return op_json;
}

}  // namespace

QnnJSONGraph::QnnJSONGraph() {
  using json = nlohmann::json;

  json_ = {// Use dummy model.cpp and model.bin files when loading JSON with QNN Netron.
           // They don't have to exist in order to visualize the graph.
           {"model.cpp", "N/A"},
           {"model.bin", "N/A"},
           {"converter_command", ""},
           {"copyright_str", "Copyright (c) Microsoft Corporation. All rights reserved."},
           {"op_types", json::array()},
           {"Total parameters", ""},
           {"Total MACs per inference", ""},
           {"graph", {{"tensors", json::object()}, {"nodes", json::object()}}}};
}

void QnnJSONGraph::AddOp(const QnnOpConfigWrapper& op_conf_wrapper) {
  // Serialize inputs and outputs.
  AddOpTensors({op_conf_wrapper.GetInputTensors(), op_conf_wrapper.GetInputsNum()});
  AddOpTensors({op_conf_wrapper.GetOutputTensors(), op_conf_wrapper.GetOutputsNum()});

  // Track unique op types (serialized in Finalize()).
  const std::string& op_type = op_conf_wrapper.GetTypeName();
  if (seen_op_types_.count(op_type) == 0) {
    seen_op_types_.insert(op_type);
  }

  // Serialize op
  json_["graph"]["nodes"][op_conf_wrapper.GetOpName()] = GetQnnOpJSON(op_conf_wrapper);
}

void QnnJSONGraph::AddOpTensors(gsl::span<const Qnn_Tensor_t> tensors) {
  for (const auto& tensor : tensors) {
    std::string name = GetQnnTensorName(tensor);  // Copies name into std::string, which is moved into seen_tensors_.
    if (seen_tensors_.count(name) == 0) {
      json_["graph"]["tensors"][name] = GetQnnTensorJSON(tensor);
      seen_tensors_.insert(std::move(name));
    }
  }
}

const nlohmann::json& QnnJSONGraph::Finalize() {
  json_["op_types"] = seen_op_types_;
  return json_;
}

}  // namespace qnn
}  // namespace onnxruntime
