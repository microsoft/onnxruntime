// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

#include "core/common/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/shared/node_unit/node_unit.h"

namespace onnxruntime::nnapi::op_builder_helpers {

template<typename T>
Status AddScalarOperand(ModelBuilder& model_builder, std::vector<uint32_t>& input_indices, T scalar_value) {
  uint32_t index = 0;
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromScalar(std::move(scalar_value), index));
  input_indices.push_back(index);
  return Status::OK();
}

Status AddNnapiTranspose(ModelBuilder& model_builder,
                         const std::string& data_input,
                         const std::string& perm_input,
                         const std::vector<int32_t>& perm,
                         const std::string& output);

Status AddNnapiReshape(ModelBuilder& model_builder,
                       const std::string& data_input,
                       const std::string& shape_input, const std::vector<int32_t>& shape_value,
                       const std::string& output, const Shape* output_shape);

Status AddNnapiSplit(ModelBuilder& model_builder,
                     const std::string& input,
                     int32_t axis,
                     const std::vector<std::string>& outputs);

Status BuildBatchedMatMul(ModelBuilder& model_builder, const NodeUnit& node_unit);

}
