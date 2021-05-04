// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This contains the utility functions which will be used to build a coreml model

#pragma once

#include <unordered_map>
#include "core/common/status.h"
#include "core/graph/basic_types.h"

namespace CoreML {
namespace Specification {
class WeightParams;
}
}  // namespace CoreML

namespace onnxruntime {
namespace coreml {

// Try to see if we can map explicit padding to auto padding for Conv/Pool
// Since usually use auto padding is more efficient
common::Status HandleAutoPad(const std::vector<int64_t> input_shape,
                             const int64_t weight_size_y,
                             const int64_t weight_size_x,
                             const std::vector<int64_t>& onnx_pads,
                             const std::vector<int64_t>& onnx_strides,
                             const std::vector<int64_t>& onnx_dilations,
                             AutoPadType auto_pad_type,
                             AutoPadType& auto_pad_type_out) ORT_MUST_USE_RESULT;

// Copy an onnx initializer data to a coreml weight
common::Status CreateCoreMLWeight(CoreML::Specification::WeightParams& weight,
                                  const ONNX_NAMESPACE::TensorProto& tensor);

// Copy the float array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight,
                        const float* data, size_t num_elements);

}  // namespace coreml
}  // namespace onnxruntime
