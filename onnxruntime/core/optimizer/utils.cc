// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/make_unique.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/framework/utils.h"
#include "core/optimizer/utils.h"
#include "float.h"
//#include <deque>

using namespace onnxruntime;

namespace onnxruntime {
namespace optimizer_utils {

bool IsFloatingPointDataType(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT
      || tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
      || tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
}

inline bool IsScalar(const NodeArg& input_arg) {
  auto shape = input_arg.Shape();
  if (shape == nullptr) {
    // shape inferencing wasn't able to populate shape information for this NodeArg
    return false;
  }

  auto dim_size = shape->dim_size();
  if (dim_size != 0) {
    // only check scalar.
    return false;
  }

  return true;
}

// Check whether input is a constant scalar with expected float value.
bool CheckConstantInput(const Graph& graph, const NodeArg& input_arg, float expected_value) {
  if (!IsScalar(input_arg)) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  if (tensor_proto == nullptr) {
    return false;
  }

  auto init_const = onnxruntime::make_unique<Initializer>(*tensor_proto);
  const auto data_type = tensor_proto->data_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* val = init_const->data<float>();
    float diff = std::abs(val[0] - static_cast<float>(expected_value));
    if (diff > FLT_EPSILON) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
    const double* val = init_const->data<double>();
    double diff = std::abs(val[0] - static_cast<double>(expected_value));
    if (diff > DBL_EPSILON) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    const MLFloat16* val = init_const->data<MLFloat16>();
    float diff = std::abs(math::halfToFloat(val[0].val) - static_cast<float>(expected_value));
    if (diff > FLT_EPSILON) {
      return false;
    }
  } else {
    // Not expected data types.
    return false;
  }

  return true;
}

// Check whether input is a constant scalar with expected intger value.
bool CheckConstantInput(const Graph& graph, const NodeArg& input_arg, int expected_value) {
  if (!IsScalar(input_arg)) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  if (tensor_proto == nullptr) {
    return false;
  }

  auto init_const = onnxruntime::make_unique<Initializer>(*tensor_proto);
  const auto data_type = tensor_proto->data_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    const int64_t* val = init_const->data<int64_t>();
    if (val[0] != static_cast<int64_t>(expected_value)) {
      return false;
    }
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    const int32_t* val = init_const->data<int32_t>();
    if (val[0] != static_cast<int32_t>(expected_value)) {
      return false;
    }
  } else {
    // Not expected data types.
    return false;
  }

  return true;
}

}  // namespace optimizer_utils
}  // namespace onnxruntime
