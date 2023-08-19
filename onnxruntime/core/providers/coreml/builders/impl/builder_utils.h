// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This contains the utility functions which will be used to build a coreml model

#pragma once

#ifdef __APPLE__

#include "core/common/gsl.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/providers/common.h"

namespace CoreML {
namespace Specification {
class WeightParams;
}
}  // namespace CoreML

namespace onnxruntime {
namespace coreml {

// Try to see if we can map explicit padding to auto padding for Conv/Pool
// Since usually use auto padding is more efficient
Status HandleAutoPad(const std::vector<int64_t> input_shape,
                     const int64_t weight_size_y,
                     const int64_t weight_size_x,
                     const std::vector<int64_t>& onnx_pads,
                     const std::vector<int64_t>& onnx_strides,
                     const std::vector<int64_t>& onnx_dilations,
                     AutoPadType auto_pad_type,
                     AutoPadType& auto_pad_type_out);

// Copy an onnx initializer data to a coreml weight
Status CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, const ONNX_NAMESPACE::TensorProto& tensor);

// Copy the float array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const float> data);

// Copy the int32_t array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int32_t> data);

// Copy the int64_t array to a coreml weight
void CreateCoreMLWeight(CoreML::Specification::WeightParams& weight, gsl::span<const int64_t> data);

}  // namespace coreml
}  // namespace onnxruntime

#endif
