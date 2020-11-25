// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_cxx_api.h"

#include "environment.h"
#include "predict.pb.h"

namespace onnxruntime {
namespace server {

onnx::TensorProto_DataType MLDataTypeToTensorProtoDataType(ONNXTensorElementDataType cpp_type);

// Convert MLValue to TensorProto. Some fields are ignored:
//   * name field: could not get from MLValue
//   * doc_string: could not get from MLValue
//   * segment field: we do not expect very large tensors in the prediction output
//   * external_data field: we do not expect very large tensors in the prediction output
// Note: If any input data is in raw_data field, all outputs tensor data will be put into raw_data field.
void MLValueToTensorProto(Ort::Value& ml_value, bool using_raw_data,
                          const std::shared_ptr<spdlog::logger>& logger,
                          /* out */ onnx::TensorProto& tensor_proto);

}  // namespace server
}  // namespace onnxruntime
