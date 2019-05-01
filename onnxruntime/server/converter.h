// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <google/protobuf/stubs/status.h>

#include "core/framework/data_types.h"

#include "environment.h"
#include "predict.pb.h"

namespace onnxruntime {
namespace server {

onnx::TensorProto_DataType MLDataTypeToTensorProtoDataType(const onnxruntime::DataTypeImpl* cpp_type);

// Convert MLValue to TensorProto. Some fields are ignored:
//   * name field: could not get from MLValue
//   * doc_string: could not get from MLValue
//   * segment field: we do not expect very large tensors in the prediction output
//   * external_data field: we do not expect very large tensors in the prediction output
// Note: If any input data is in raw_data field, all outputs tensor data will be put into raw_data field.
common::Status MLValueToTensorProto(const onnxruntime::MLValue& ml_value, bool using_raw_data,
                                    std::unique_ptr<onnxruntime::logging::Logger> logger,
                                    /* out */ onnx::TensorProto& tensor_proto);

}  // namespace server
}  // namespace onnxruntime
