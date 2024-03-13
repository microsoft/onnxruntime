// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/framework/utils.h"
#include "core/optimizer/utils.h"
#include "float.h"
// #include <deque>

#include <string>
#include <unordered_map>
#include <unordered_set>
#endif  // #if !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_arg.h"
#include "core/optimizer/initializer.h"
#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

using namespace onnxruntime;

namespace onnxruntime {
namespace optimizer_utils {

#if !defined(ORT_MINIMAL_BUILD)

bool IsFloatingPointDataType(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
         tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
         tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
}

// Check whether input is a constant scalar with expected float value.
bool IsInitializerWithExpectedValue(const Graph& graph, const NodeArg& input_arg, float expected_value, bool is_constant) {
  if (!IsScalar(input_arg)) {
    return false;
  }

  constexpr float atol = 1e-8f;
  constexpr float rtol = 1e-5f;
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (is_constant) {
    tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  } else if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    return false;
  }

  if (tensor_proto == nullptr) {
    return false;
  }

  Initializer init_const{*tensor_proto, graph.ModelPath()};
  const auto data_type = tensor_proto->data_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* val = init_const.data<float>();
    if (std::isnan(val[0]) || std::isinf(val[0])) {
      if (std::isinf(val[0]) && std::isinf(expected_value) && (std::signbit(val[0]) == std::signbit(expected_value))) {
        return true;
      }
      return false;
    }

    float diff = std::abs(val[0] - expected_value);
    if (diff > (atol + rtol * std::abs(expected_value))) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
    const double* val = init_const.data<double>();
    if (std::isnan(val[0]) || std::isinf(val[0])) return false;

    const double expected_val = static_cast<double>(expected_value);
    double diff = std::abs(val[0] - expected_val);
    if (diff > (atol + static_cast<double>(rtol) * std::abs(expected_value))) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    const MLFloat16* val = init_const.data<MLFloat16>();
    const float flt_val = math::halfToFloat(val[0].val);
    if (std::isnan(flt_val) || std::isinf(flt_val)) return false;
    const float expected_val = math::halfToFloat(math::floatToHalf(expected_value));
    float diff = std::abs(flt_val - expected_val);
    if (diff > (atol + rtol * std::abs(expected_value))) {
      return false;
    }
  } else {
    // Not expected data types.
    return false;
  }

  return true;
}

// Check whether input is a constant scalar with expected intger value.
bool IsInitializerWithExpectedValue(const Graph& graph, const NodeArg& input_arg, int64_t expected_value, bool is_constant) {
  if (!IsScalar(input_arg)) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (is_constant) {
    tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  } else if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    return false;
  }

  Initializer init_const{*tensor_proto, graph.ModelPath()};
  const auto data_type = tensor_proto->data_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    const int64_t* val = init_const.data<int64_t>();
    if (val[0] != expected_value) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    const int32_t* val = init_const.data<int32_t>();
    if (static_cast<int64_t>(val[0]) != expected_value) {
      return false;
    }
  } else {
    // Not expected data types.
    return false;
  }

  return true;
}

bool IsAttributeWithExpectedValue(const Node& node, const std::string& attr_name, int64_t expected_value) {
  const auto* attr_proto = graph_utils::GetNodeAttribute(node, attr_name);
  if ((nullptr != attr_proto) && attr_proto->has_i()) {
    return attr_proto->i() == expected_value;
  }
  return false;
}

bool IsAttributeWithExpectedValue(const Node& node, const std::string& attr_name, float expected_value, float eps) {
  const auto* attr_proto = graph_utils::GetNodeAttribute(node, attr_name);
  if ((nullptr != attr_proto) && attr_proto->has_f()) {
    return std::abs(attr_proto->f() - expected_value) < eps;
  }
  return false;
}

bool IsAttributeWithExpectedValues(const Node& node, const std::string& attr_name, const std::vector<int64_t>& expected_values) {
  const auto* attr_proto = graph_utils::GetNodeAttribute(node, attr_name);
  if ((nullptr == attr_proto) || attr_proto->ints_size() != (int)expected_values.size()) {
    return false;
  }

  for (int i = 0; i < attr_proto->ints_size(); i++) {
    if (attr_proto->ints(i) != expected_values[i]) {
      return false;
    }
  }

  return true;
}

bool AppendTensorFromInitializer(const Graph& graph, const NodeArg& input_arg, InlinedVector<int64_t>& data, bool require_constant) {
  if (require_constant && !graph_utils::IsConstantInitializer(graph, input_arg.Name(), true)) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    return false;
  }

  Initializer init_const{*tensor_proto, graph.ModelPath()};
  const auto data_type = tensor_proto->data_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    const int64_t* val = init_const.data<int64_t>();
    data.reserve(data.size() + gsl::narrow<size_t>(init_const.size()));
    data.insert(data.end(), val, val + init_const.size());
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    const int32_t* val = init_const.data<int32_t>();
    data.reserve(data.size() + gsl::narrow<size_t>(init_const.size()));
    for (size_t i = 0; i < init_const.size(); i++) {
      data.push_back(static_cast<int64_t>(val[i]));
    }
  } else {
    return false;
  }

  return true;
}

bool ValidateShape(const NodeArg& node_arg, const std::initializer_list<int64_t>& expected_dim_values) {
  auto shape = node_arg.Shape();
  if (shape == nullptr || static_cast<size_t>(shape->dim_size()) != expected_dim_values.size()) {
    return false;
  }

  int index = 0;
  for (auto& expected_dim_value : expected_dim_values) {
    if (expected_dim_value > 0) {
      auto dim = shape->dim(index);
      if (!utils::HasDimValue(dim) || expected_dim_value != dim.dim_value()) {
        return false;
      }
    }
    ++index;
  }

  return true;
}

bool CompareShape(const ONNX_NAMESPACE::TensorShapeProto& node_arg_shape, const ONNX_NAMESPACE::TensorShapeProto& node_arg_other_shape) {
  if (node_arg_shape.dim_size() != node_arg_other_shape.dim_size())
    return false;

  if (node_arg_shape.dim_size() < 1)
    return false;

  for (int i = 0; i < node_arg_shape.dim_size(); ++i) {
    const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim = node_arg_shape.dim(i);
    const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim_other = node_arg_other_shape.dim(i);
    if (!utils::HasDimValue(dim) || !utils::HasDimValue(dim_other))
      return false;
    if (dim.dim_value() != dim_other.dim_value())
      return false;
  }
  return true;
}

bool IsShapeKnownOnAllDims(const NodeArg& node_arg, int expected_dim_size) {
  auto shape = node_arg.Shape();
  if (shape == nullptr || shape->dim_size() != expected_dim_size) {
    return false;
  }

  for (int i = 0; i < expected_dim_size; i++) {
    if (!utils::HasDimValue(shape->dim(i))) {
      return false;
    }
  }

  return true;
}

int32_t IndexOfNodeInput(const Node& node, const NodeArg& node_arg) {
  int32_t index = 0;
  for (auto& input_arg : node.InputDefs()) {
    if (input_arg->Name().compare(node_arg.Name()) == 0) {
      return index;
    }
    index++;
  }

  return -1;
}

int32_t IndexOfNodeOutput(const Node& node, const NodeArg& node_arg) {
  int32_t index = 0;
  for (auto& output_arg : node.OutputDefs()) {
    if (output_arg->Name().compare(node_arg.Name()) == 0) {
      return index;
    }
    index++;
  }

  return -1;
}

// Allow certain domains/ops. We don't know anything about unknown domains/ops (e.g. custom ops),
// so we have to assume that they are not deterministic, to be on the safe side.
// We could also allow other known domains (kMSDomain, kMSNchwcDomain, kMSFeaturizersDomain),
// as long as we verify which of their operations are non-deterministic and add them in the map below.
constexpr std::array kOnnxDomainNonDeterministicOps{"RandomUniform", "RandomNormal", "RandomUniformLike",
                                                    "RandomNormalLike", "Multinomial", "Dropout"};

// List of deterministic MS domain operators. Currently used for constant folding and common subexpression elimination.
//
// TODO(adrianlizarraga): Investigate converting to lists of *non-deterministic* MS domain operators to be consistent
// with the above ONNX list. With the current approach, only MS domain Q/DQ operators
// (plus ShrunkenGather for training) are considered deterministic.
#ifdef ENABLE_TRAINING_OPS
constexpr std::array kMSDomainDeterministicOps{"ShrunkenGather", "QuantizeLinear", "DequantizeLinear",
                                               "ConcatTraining"};
#else
constexpr std::array kMSDomainDeterministicOps{"QuantizeLinear", "DequantizeLinear"};
#endif

bool IsOperationDeterministic(const std::string& domain, const std::string& op) {
  if (domain.compare(kOnnxDomain) == 0) {
    auto iter = std::find(kOnnxDomainNonDeterministicOps.begin(), kOnnxDomainNonDeterministicOps.end(), op);
    return iter == kOnnxDomainNonDeterministicOps.end();
  }

  if (domain.compare(kMSDomain) == 0) {
    auto iter = std::find(kMSDomainDeterministicOps.begin(), kMSDomainDeterministicOps.end(), op);
    return iter != kMSDomainDeterministicOps.end();
  }

  // Unknown domain. Assume the op is not deterministic.
  return false;
}

#endif  // #if !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

bool GetClipConstantMinMax(const Graph& graph, const Node& node, float& min, float& max) {
  min = std::numeric_limits<float>::lowest();
  max = std::numeric_limits<float>::max();

  // Clip opset 1 and 6 has min and max as attributes. they're inputs from opset 11 on.
  bool min_max_are_attributes = node.SinceVersion() < 11;
  bool min_max_are_constant_values = true;

  if (min_max_are_attributes) {
    min = graph_utils::GetNodeAttribute(node, "min")->f();
    max = graph_utils::GetNodeAttribute(node, "max")->f();
  } else {
    // update min/max if provided via a constant initializer
    // return true if value is default or coming from a constant initializer and update 'value'
    // return false if value is mutable
    auto update_if_constant_value =
        [&graph](const Node& node, size_t input_idx, float& value) {
          const auto& input_defs = node.InputDefs();
          const NodeArg* input = (input_defs.size() > input_idx) ? input_defs[input_idx] : nullptr;

          if (input == nullptr || !input->Exists()) {
            // optional input not specified so using default value
            return true;
          }

          bool is_constant = true;
          const ONNX_NAMESPACE::TensorProto* initializer = graph.GetConstantInitializer(input->Name(), true);
          if (initializer) {
            Initializer i(*initializer, graph.ModelPath());
            switch (initializer->data_type()) {
              case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
                value = *i.data<float>();
                break;
              // double isn't currently supported
              // case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
              //  value = static_cast<float>(*i.data<double>());
              //  break;
              case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
                value = math::halfToFloat(i.data<MLFloat16>()->val);
                break;
              default:
                ORT_THROW("Unexpected data type for Clip input of ", initializer->data_type());
            }
          } else {
            is_constant = false;
          }

          return is_constant;
        };

    // 'min' is input 1, 'max' is input 2. both are optional.
    // if the input is constant, 'min' or 'max' is updated by the call to get_if_constant_value
    min_max_are_constant_values = update_if_constant_value(node, 1, min) &&
                                  update_if_constant_value(node, 2, max);
  }

  return min_max_are_constant_values;
}

bool CheckOutputEdges(const Graph& graph, const Node& node, size_t expected_output_edges) {
  if (graph.NodeProducesGraphOutput(node)) {
    return false;
  }

  return node.GetOutputEdgesCount() == expected_output_edges;
}

bool IsScalar(const NodeArg& input_arg) {
  auto shape = input_arg.Shape();
  if (shape == nullptr) {
    // shape inferencing wasn't able to populate shape information for this NodeArg
    return false;
  }

  auto dim_size = shape->dim_size();
  return dim_size == 0 || (dim_size == 1 && shape->dim(0).has_dim_value() && shape->dim(0).dim_value() == 1);
}

template <typename T>
bool GetScalarInitializerValue(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, T& value,
                               bool is_constant) {
  if (!IsScalar(input_arg)) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (is_constant) {
    tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  } else if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    return false;
  }

  if (tensor_proto == nullptr) {
    return false;
  }

  Initializer init_const{*tensor_proto, graph.ModelPath()};
  const T* val = init_const.data<T>();
  value = *val;

  return true;
}

template bool GetScalarInitializerValue<float>(const onnxruntime::Graph& graph, const onnxruntime::NodeArg& input_arg, float& value,
                                               bool is_constant);

#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

}  // namespace optimizer_utils
}  // namespace onnxruntime
