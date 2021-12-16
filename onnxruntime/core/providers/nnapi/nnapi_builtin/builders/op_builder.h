// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/graph/basic_types.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

class Node;
class NodeUnit;

namespace common {
class Status;
}

namespace nnapi {

class ModelBuilder;

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Check if the initializers of this operator need preprocess
  // which will not be copied
  virtual void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const = 0;

  // Add the operator to NNAPI model
  virtual common::Status AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const ORT_MUST_USE_RESULT = 0;
};

// Get the lookup table with IOpBuilder delegates for different onnx operators
// Note, the lookup table should have same number of entries as the result of CreateOpSupportCheckers()
// in op_support_checker.h
const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders();

// Transpose the NHWC input to NCHW output
common::Status TransposeNHWCToNCHW(ModelBuilder& model_builder, const std::string& input, const std::string& output)
    ORT_MUST_USE_RESULT;

// Get the quantized input's scale and zero point for the given input
common::Status GetQuantizedInputScaleAndZeroPoint(const InitializedTensorSet& initializers,
                                                  const NodeUnit& node_unit, const std::string& input_name,
                                                  float& scale, int32_t& zero_point) ORT_MUST_USE_RESULT;

}  // namespace nnapi
}  // namespace onnxruntime
