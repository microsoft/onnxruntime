// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/framework/utils.h"
#include "core/optimizer/utils.h"
#include "float.h"
//#include <deque>

#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace onnxruntime;

namespace onnxruntime {
namespace optimizer_utils {

bool IsFloatingPointDataType(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
         tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
         tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
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

// Check whether input is a constant scalar with expected float value.
bool IsInitializerWithExpectedValue(const Graph& graph, const NodeArg& input_arg, float expected_value, bool is_constant) {
  if (!IsScalar(input_arg)) {
    return false;
  }

  const float atol = 1e-8f;
  const float rtol = 1e-5f;
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
    if (diff > (atol + rtol * std::abs(expected_value))) {
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

bool AppendTensorFromInitializer(const Graph& graph, const NodeArg& input_arg, std::vector<int64_t>& data, bool require_constant) {
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
    for (int64_t i = 0; i < init_const.size(); i++) {
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

bool CheckOutputEdges(const Graph& graph, const Node& node, size_t expected_output_edges) {
  if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
    return false;
  }

  return node.GetOutputEdgesCount() == expected_output_edges;
}

// Allow certain domains/ops. We don't know anything about unknown domains/ops (e.g. custom ops),
// so we have to assume that they are not deterministic, to be on the safe side.
// We could also allow other known domains (kMSDomain, kMSNchwcDomain, kMSFeaturizersDomain),
// as long as we verify which of their operations are non-deterministic and add them in the map below.
static const std::unordered_map<std::string, std::unordered_set<std::string>> kNonDeterministicOps =
    {
        {kOnnxDomain, {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"}},
};

bool IsOperationDeterministic(const std::string& domain, const std::string& op) {
  auto itDomain = kNonDeterministicOps.find(domain);
  if (itDomain == kNonDeterministicOps.end()) {
    // Unknown domain. Assume the op is not deterministic.
    return false;
  }

  return itDomain->second.count(op) == 0;
}

}  // namespace optimizer_utils
}  // namespace onnxruntime
