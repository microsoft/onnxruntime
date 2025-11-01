// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/common/logging/logging.h"
#include <onnx/onnx-ml.pb.h>

namespace onnxruntime {

/**
 * @class CustomDataPropagation
 * Custom data propagation for the operator to help enhance shape inference.
 */
class CustomDataPropagation {
 public:
  virtual ~CustomDataPropagation() = default;
  virtual Status infer() = 0;

 protected:
  CustomDataPropagation(const Node& node,
                        NodeArg& output_def,
                        std::function<Status(const std::string&, TensorShapeVector&)> func,
                        const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation,
                        const logging::Logger& logger) noexcept
      : node_(node),
        output_def_(output_def),
        get_initialized_input_values_func_(func),
        output_from_onnx_op_data_propagation_(output_from_onnx_op_data_propagation),
        logger_(logger) {}

  const Node& node_;
  NodeArg& output_def_;
  std::function<Status(const std::string&, TensorShapeVector&)> get_initialized_input_values_func_;
  const ONNX_NAMESPACE::TypeProto& output_from_onnx_op_data_propagation_;
  const logging::Logger& logger_;
};

}  // namespace onnxruntime
